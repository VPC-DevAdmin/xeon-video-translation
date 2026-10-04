"""One clock for the assistant's audio and video, with clips scheduled at future times.

The old `Playback` admitted at most two seconds of media and showed a frozen portrait
when nothing was queued. The assistant plays a prepared acknowledgement right away,
then idle motion, then a reply whose chunks are scheduled on a deadline chosen from
the renderer's measured rate. So this timeline accepts clips with explicit start
times far ahead, loops idle footage when nothing is active, and counts stalls
(video frames due inside a promised reply window whose chunk has not landed).

Idle footage is a list of segments, each a continuous recording of the renderer's
motion. Growth appends to the last segment while the renderer still continues it and
starts a new one after a reply moved the motion state; the loop runs over all
segments and hides every boundary (and the wrap) behind a half-second dissolve. The
loop position is a cursor that advances with the frames shown, so appending footage
never moves the frame on screen. Switching between idle and a clip dissolves too."""

from __future__ import annotations

import bisect
import time

import numpy as np


class Timeline:
    def __init__(self, fps: int = 25, still=None, idle_frames=None, audio_rate: int = 48000, transition_seconds: float = 0.5):
        self.epoch = time.monotonic()
        self.fps = int(fps)
        self.audio_rate = int(audio_rate)
        self.still = still
        self.idle_segments: list = []
        if idle_frames is not None and len(idle_frames):
            self.idle_segments.append(np.asarray(idle_frames))
        self.idle_crossfade = int(round(0.5 * self.fps))        # frames blended at each idle boundary
        self.transition_frames = int(round(transition_seconds * self.fps))   # dissolve when idle and clips alternate
        self.generation = 0
        self.clips: list = []           # (start, end, audio48k, frames), sorted by start
        self.promises: list = []        # (start, end) windows a reply has committed to
        self.first_audio_seconds = None
        self.stalls = 0
        self.frames_sent = 0
        self.frames_skipped = 0
        self.idle_frames_sent = 0
        self.scheduled_seconds = 0.0
        self._cursor = 0                # position in the idle loop
        self._last_index = None
        self._last = None               # (unblended image, source) shown at the previous frame
        self._blend = None              # (image to dissolve from, frame index of the switch)

    def now(self) -> float:
        return time.monotonic() - self.epoch

    # ------------------------------------------------------------------ idle
    def add_idle(self, frames, continuous: bool) -> None:
        """Append idle footage: to the last segment when the renderer's motion continued
        it, otherwise as a new segment whose boundary the loop will dissolve across."""
        frames = np.asarray(frames)
        if not len(frames):
            return
        if continuous and self.idle_segments:
            self.idle_segments[-1] = np.concatenate([self.idle_segments[-1], frames])
        else:
            self.idle_segments.append(frames)

    @property
    def idle_frame_count(self) -> int:
        return sum(len(s) for s in self.idle_segments)

    def idle_seconds(self) -> float:
        return self.idle_frame_count / self.fps

    # ------------------------------------------------------------------ clips
    def interrupt(self) -> int:
        self.generation += 1
        self.clips.clear()
        self.promises.clear()
        return self.generation

    def schedule(self, start: float, audio, frames, generation: int):
        """Place a clip at `start` (timeline seconds). Returns (start, end) or None when stale."""
        if generation != self.generation or not len(frames):
            return None
        audio = np.asarray(audio)
        duration = max(len(audio) / self.audio_rate, len(frames) / self.fps)
        end = start + duration
        starts = [c[0] for c in self.clips]
        self.clips.insert(bisect.bisect_right(starts, start), (start, end, audio, frames))
        self.scheduled_seconds += duration
        self.promises = [(a, b) for a, b in self.promises if not (a < end and start < b)]
        return start, end

    def promise(self, start: float, end: float, generation: int) -> None:
        if generation == self.generation:
            self.promises.append((start, end))

    def last_end(self) -> float:
        return max((c[1] for c in self.clips), default=self.now())

    def active(self, seconds: float):
        while self.clips and self.clips[0][1] < seconds - 0.5:
            self.clips.pop(0)
        for clip in self.clips:
            if clip[0] <= seconds < clip[1]:
                return clip
            if clip[0] > seconds:
                break
        return None

    def promised(self, seconds: float) -> bool:
        return any(a <= seconds < b for a, b in self.promises)

    # ----------------------------------------------------------------- output
    def frame_at(self, seconds: float):
        """The frame to show at `seconds` and its source ("clip", "idle" or "still")."""
        index = int(seconds * self.fps)
        if self._last_index is not None:
            self._cursor += max(0, index - self._last_index)
        self._last_index = index
        clip = self.active(seconds)
        if clip is not None:
            frames = clip[3]
            image, source = frames[min(len(frames) - 1, int((seconds - clip[0]) * self.fps))], "clip"
        else:
            if self.promised(seconds):
                self.stalls += 1
            if self.idle_segments:
                image, source = idle_loop_frame(self.idle_segments, self._cursor, self.idle_crossfade), "idle"
            else:
                image, source = self.still, "still"
        if self._last is not None and source != self._last[1] and self.transition_frames and image is not None:
            self._blend = (self._last[0], index)
        self._last = (image, source)
        if self._blend is not None and image is not None:
            since = index - self._blend[1]
            if 0 <= since < self.transition_frames and self._blend[0] is not None and self._blend[0].shape == image.shape:
                weight = 1.0 - (since + 1) / (self.transition_frames + 1)
                image = (weight * self._blend[0].astype(np.float32) + (1.0 - weight) * image.astype(np.float32)).astype(image.dtype)
            else:
                self._blend = None
        return image, source

    def audio_packet(self, seconds: float, samples: int):
        out = np.zeros(samples, dtype=np.int16)
        self.active(seconds)
        for start, end, audio, _ in self.clips:
            lo = max(0, round((start - seconds) * self.audio_rate))
            hi = min(samples, round((end - seconds) * self.audio_rate))
            if hi > lo:
                offset = max(0, round((seconds + lo / self.audio_rate - start) * self.audio_rate))
                portion = audio[offset: offset + hi - lo]
                out[lo: lo + len(portion)] = portion
        return out


def idle_loop_frame(segments, position: int, crossfade: int):
    """Frame at `position` of a forward loop over continuous `segments`.

    Each segment plays its first n-K frames. The K frames that follow (its natural
    continuation) are dissolved into the first K frames of the next segment, and the
    last segment dissolves into the first, so every boundary is a half-second
    crossfade instead of a jump or a reversed motion. Appending frames to the last
    segment or adding a segment only lengthens the loop; earlier positions keep
    their frames, which is what lets the loop grow while it plays."""
    segments = [s for s in segments if len(s)]
    if not segments:
        return None
    held = [max(0, min(int(crossfade), len(s) // 2 - 1)) for s in segments]
    lengths = [len(s) - k for s, k in zip(segments, held)]
    j = position % sum(lengths)
    index = 0
    for index, length in enumerate(lengths):
        if j < length:
            break
        j -= length
    frame = segments[index][j]
    previous, k = segments[index - 1], held[index - 1]          # index 0 wraps to the last segment
    if j >= k:
        return frame
    weight = 1.0 - (j + 1) / (k + 1)                            # 1 -> continuation, 0 -> head
    tail = previous[len(previous) - k + j].astype(np.float32)
    return (weight * tail + (1.0 - weight) * frame.astype(np.float32)).astype(frame.dtype)


def idle_frame(frames, index: int, crossfade: int):
    """Single-segment loop (kept for callers and tests of the one-recording case)."""
    return idle_loop_frame([np.asarray(frames)], index, crossfade)


def head_start_required(reply_seconds: float, render_ratio: float, first_chunk_seconds: float,
                        margin: float = 1.0) -> float:
    """Smallest head start that keeps a reply of `reply_seconds` from stalling when the
    renderer produces `render_ratio` seconds of video per second (r) and the first chunk
    takes `first_chunk_seconds`. r >= 1 needs only the first chunk and the margin."""
    deficit = max(0.0, reply_seconds * (1.0 - min(render_ratio, 1.0)))
    return first_chunk_seconds + deficit + margin
