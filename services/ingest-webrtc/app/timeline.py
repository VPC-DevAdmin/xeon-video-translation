"""One clock for the assistant's audio and video, with clips scheduled at future times.

The old `Playback` admitted at most two seconds of media and showed a frozen portrait
when nothing was queued. The assistant plays a prepared acknowledgement right away,
then idle motion, then a reply whose chunks are scheduled on a deadline chosen from
the renderer's measured rate. So this timeline accepts clips with explicit start
times far ahead, loops idle footage when nothing is active, and counts stalls
(video frames due inside a promised reply window whose chunk has not landed)."""

from __future__ import annotations

import bisect
import time

import numpy as np


class Timeline:
    def __init__(self, fps: int = 25, still=None, idle_frames=None, audio_rate: int = 48000):
        self.epoch = time.monotonic()
        self.fps = int(fps)
        self.audio_rate = int(audio_rate)
        self.still = still
        self.idle_frames = idle_frames if idle_frames is not None and len(idle_frames) else None
        self.generation = 0
        self.clips: list = []           # (start, end, audio48k, frames), sorted by start
        self.promises: list = []        # (start, end) windows a reply has committed to
        self.first_audio_seconds = None
        self.stalls = 0
        self.frames_sent = 0
        self.frames_skipped = 0
        self.idle_frames_sent = 0
        self.scheduled_seconds = 0.0

    def now(self) -> float:
        return time.monotonic() - self.epoch

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

    def frame_at(self, seconds: float):
        clip = self.active(seconds)
        if clip is not None:
            frames = clip[3]
            return frames[min(len(frames) - 1, int((seconds - clip[0]) * self.fps))], "clip"
        if self.promised(seconds):
            self.stalls += 1
        if self.idle_frames is not None:
            n = len(self.idle_frames)
            if n == 1:
                return self.idle_frames[0], "idle"
            period = 2 * n - 2                      # ping-pong loop, no jump at the wrap
            index = int(seconds * self.fps) % period
            if index >= n:
                index = period - index
            return self.idle_frames[index], "idle"
        return self.still, "still"

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


def head_start_required(reply_seconds: float, render_ratio: float, first_chunk_seconds: float,
                        margin: float = 1.0) -> float:
    """Smallest head start that keeps a reply of `reply_seconds` from stalling when the
    renderer produces `render_ratio` seconds of video per second (r) and the first chunk
    takes `first_chunk_seconds`. r >= 1 needs only the first chunk and the margin."""
    deficit = max(0.0, reply_seconds * (1.0 - min(render_ratio, 1.0)))
    return first_chunk_seconds + deficit + margin
