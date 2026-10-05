"""One clock for the assistant's audio and video, with clips scheduled at future times.

The old `Playback` admitted at most two seconds of media and showed a frozen portrait
when nothing was queued. The assistant plays a prepared acknowledgement right away,
then idle motion, then a reply whose chunks are scheduled on a deadline chosen from
the renderer's measured rate. So this timeline accepts clips with explicit start
times far ahead, loops idle footage when nothing is active, and counts stalls
(video frames due inside a promised reply window whose chunk has not landed).

Idle footage is a list of segments, each a continuous recording of the renderer's
motion. Growth appends to the last segment while the renderer still continues it and
starts a new one after a reply moved the motion state. Finalized loops cut at a
naturally matching frame. The loop
position is a cursor that advances with the frames shown, so appending footage never
moves the frame on screen. Prepared face clips use direct joins instead of dissolves.

There can be several idle loops ("front": listening; "working": looking down at a
tablet) and the active one is switched at scheduled times, so the filler plan of a
turn can send the persona to its tablet and back. Filler clips carry a tag so the
plan can cut them off (with a short fade) the moment the reply is ready.

Anchors: every clip starts from the renderer's rest pose (it is rendered from a reset)
and ends after naturally rendered trailing silence, and every idle loop starts from
that same rest pose. The idle loop is one continuous take cut at the frame that best
matches its start. Whole-face dissolves remain available only as an explicit legacy
option for comparison."""

from __future__ import annotations

import bisect
import time

import numpy as np


def settle_frames(last_frame, anchor, count: int):
    """`count` frames dissolving from `last_frame` into `anchor`, the last one the anchor
    itself: appended to footage so it ends in the common rest pose."""
    last = np.asarray(last_frame).astype(np.float32)
    target = np.asarray(anchor).astype(np.float32)
    out = []
    for i in range(count):
        w = (i + 1) / count
        out.append(((1.0 - w) * last + w * target).astype(np.asarray(anchor).dtype))
    return np.stack(out) if out else np.zeros((0,) + np.asarray(anchor).shape, np.asarray(anchor).dtype)


def best_loop_end(frames, start: int, step: int = 4) -> int:
    """Index j >= start whose frame is closest to frame 0, so frames[0:j+1] loops back to
    its start through a naturally similar pose instead of a cut or a dissolve."""
    frames = np.asarray(frames)
    head = frames[0, ::step, ::step].astype(np.float32)
    best, best_diff = len(frames), None
    for j in range(start, len(frames)):
        diff = float(np.abs(frames[j, ::step, ::step].astype(np.float32) - head).mean())
        if best_diff is None or diff < best_diff:
            best, best_diff = j, diff
    return best


def closest_anchor_end(frames, anchor, first: int, step: int = 6) -> int:
    """Exclusive end index in a silent tail, favoring a rest-like, low-motion frame.

    This selects an actual rendered frame. Pixel dissolves of a moving face create
    two sets of facial features, so a prepared clip should finish on natural output.
    """
    frames = np.asarray(frames)
    target = np.asarray(anchor) if anchor is not None else None
    if not len(frames) or target is None or target.shape != frames.shape[1:]:
        return len(frames)
    first = min(max(0, int(first)), len(frames) - 1)
    sampled_target = target[::step, ::step].astype(np.int16)
    best, best_score = len(frames) - 1, float("inf")
    for index in range(first, len(frames)):
        frame = frames[index, ::step, ::step].astype(np.int16)
        pose = np.abs(frame - sampled_target).mean()
        motion = np.abs(frame - frames[index - 1, ::step, ::step].astype(np.int16)).mean() if index else 0.0
        score = float(pose + 0.2 * motion)
        if score < best_score:
            best, best_score = index, score
    return best + 1


class IdleLoop:
    """One continuous take of idle footage played as a loop.

    An optional legacy crossfade can hide a growing take's wrap. Once the take is
    finalized it is cut at the frame that best matches its first frame (`best_loop_end`),
    so the wrap is a natural continuation without blended facial features. Frame 0 is the renderer's rest
    pose (every take starts from a reset), which is also where clips start and end."""

    def __init__(self, crossfade: int, frames=None):
        self.crossfade = int(crossfade)
        self.frames = np.asarray(frames) if frames is not None and len(frames) else None
        self.pending: list = []         # footage waiting until the loop is outside its wrap blend
        self.cursor = 0
        self.finalized = False
        self.wrap_blend = 2

    @property
    def ready(self) -> bool:
        return self.frames is not None

    @property
    def frame_count(self) -> int:
        return (len(self.frames) if self.frames is not None else 0) + sum(len(f) for f in self.pending)

    @property
    def anchor(self):
        """The rest pose the take starts from."""
        return self.frames[0] if self.frames is not None else None

    @property
    def segments(self) -> list:
        return [self.frames] if self.frames is not None else []

    def effective_crossfade(self) -> int:
        return self.wrap_blend if self.finalized else self.crossfade

    def add(self, frames, continuous: bool = True) -> None:
        """Append footage that continues the take."""
        frames = np.asarray(frames)
        if not len(frames) or self.finalized:
            return
        self.pending.append(frames)
        self.publish()

    def publish(self) -> None:
        if not self.pending:
            return
        if self.frames is not None:
            segment, j, k = idle_loop_locate([self.frames], self.cursor, self.effective_crossfade())
            if j < k:
                return                                   # the wrap blend is on screen; its tail must not change now
            self.frames = np.concatenate([self.frames] + self.pending)
        else:
            self.frames = np.concatenate(self.pending)
        self.pending.clear()

    def finalize(self, min_seconds: float, fps: int) -> bool:
        """Cut the take at its best loop point (at least `min_seconds` in) and stop growing."""
        if self.finalized:
            return True
        self.publish()
        if self.frames is None or len(self.frames) < int(min_seconds * fps) + 2:
            return False
        # Do not choose an endpoint before the frame currently being displayed.
        # Shortening the loop past the cursor would remap it and create a visible jump.
        first = max(int(min_seconds * fps), min(self.cursor + 1, len(self.frames) - 1))
        j = best_loop_end(self.frames, first)
        # Include the frame chosen as the match. Slicing at j discarded that frame,
        # leaving the loop to jump from a less similar predecessor to frame zero.
        self.frames = self.frames[:j + 1]
        self.pending.clear()
        self.finalized = True
        self.wrap_blend = 0
        self.cursor %= idle_loop_length([self.frames], self.effective_crossfade())
        return True

    def restart(self) -> None:
        """Drop the take (used when the renderer's motion no longer continues it)."""
        self.frames, self.pending, self.cursor, self.finalized = None, [], 0, False

    def advance(self, delta: int) -> None:
        """Move the cursor by `delta` frames, kept inside the loop as it is now; footage
        published afterwards only extends the loop beyond it."""
        if self.frames is not None:
            self.cursor = (self.cursor + max(0, delta)) % idle_loop_length([self.frames], self.effective_crossfade())
        self.publish()

    def frame(self):
        return idle_loop_frame([self.frames], self.cursor, self.effective_crossfade())


class Timeline:
    def __init__(self, fps: int = 25, still=None, idle_frames=None, audio_rate: int = 48000,
                 transition_seconds: float = 0.0, idle_crossfade_seconds: float = 0.0):
        self.epoch = time.monotonic()
        self.fps = int(fps)
        self.audio_rate = int(audio_rate)
        self.still = still
        self.idle_crossfade = int(round(idle_crossfade_seconds * self.fps))  # legacy wrap dissolve; off for faces
        self.loops: dict = {"front": IdleLoop(self.idle_crossfade, idle_frames)}
        self.mode_switches: list = []   # (seconds, loop name), sorted; "front" before the first
        self.transition_frames = int(round(transition_seconds * self.fps))   # optional legacy dissolve; off for face clips
        self.generation = 0
        self.clips: list = []           # (start, end, audio48k, frames, tag), sorted by start
        self.promises: list = []        # (start, end) windows a reply has committed to
        self.first_audio_seconds = None
        self.stalls = 0
        self.frames_sent = 0
        self.frames_skipped = 0
        self.idle_frames_sent = 0
        self.scheduled_seconds = 0.0
        self._last_index = None
        self._last = None               # (unblended image, source) shown at the previous frame
        self._blend = None              # (image to dissolve from, frame index of the switch)

    def now(self) -> float:
        return time.monotonic() - self.epoch

    # ------------------------------------------------------------------ idle
    def loop(self, name: str) -> IdleLoop:
        if name not in self.loops:
            self.loops[name] = IdleLoop(self.idle_crossfade)
        return self.loops[name]

    @property
    def idle_segments(self) -> list:
        return self.loops["front"].segments

    @idle_segments.setter
    def idle_segments(self, segments) -> None:
        segments = [np.asarray(x) for x in segments if len(x)]
        self.loops["front"].frames = np.concatenate(segments) if segments else None

    def add_idle(self, frames, continuous: bool = True, name: str = "front") -> None:
        self.loop(name).add(frames, continuous)

    @property
    def idle_frame_count(self) -> int:
        return self.loops["front"].frame_count

    def idle_seconds(self, name: str = "front") -> float:
        return self.loop(name).frame_count / self.fps

    def set_mode(self, seconds: float, name: str, generation: int) -> None:
        """From `seconds` on, show the idle loop `name` whenever no clip is active."""
        if generation != self.generation:
            return
        self.mode_switches = sorted([(t, m) for t, m in self.mode_switches if t < seconds] + [(seconds, name)])

    def mode_at(self, seconds: float) -> str:
        name = "front"
        for t, m in self.mode_switches:
            if t <= seconds:
                name = m
            else:
                break
        return name

    # ------------------------------------------------------------------ clips
    def interrupt(self) -> int:
        self.generation += 1
        self.clips.clear()
        self.promises.clear()
        self.mode_switches.clear()
        return self.generation

    def schedule(self, start: float, audio, frames, generation: int, tag: str = "reply"):
        """Place a clip at `start` (timeline seconds). Returns (start, end) or None when stale."""
        if generation != self.generation or not len(frames):
            return None
        audio = np.asarray(audio)
        duration = max(len(audio) / self.audio_rate, len(frames) / self.fps)
        end = start + duration
        starts = [c[0] for c in self.clips]
        self.clips.insert(bisect.bisect_right(starts, start), (start, end, audio, frames, tag))
        self.scheduled_seconds += duration
        self.promises = [(a, b) for a, b in self.promises if not (a < end and start < b)]
        return start, end

    def truncate(self, seconds: float, tag: str, fade_seconds: float = 0.08, settle_to=None, settle_count: int = 6,
                 min_keep_seconds: float = 0.5):
        """Cut clips carrying `tag` at `seconds`: later ones are dropped, the one playing
        across it ends there with a short audio fade and, when `settle_to` is given, a
        few frames settling into that anchor frame. A clip that would have played for
        less than `min_keep_seconds` is dropped rather than cut (a syllable and a fade
        would sound like a stutter). Returns (affected, end) where `end` is when the cut
        footage finishes (`seconds` when nothing was playing)."""
        kept, affected, end_at = [], 0, seconds
        for start, end, audio, frames, clip_tag in self.clips:
            if clip_tag != tag or end <= seconds:
                kept.append((start, end, audio, frames, clip_tag))
                continue
            affected += 1
            if start >= seconds or seconds - start < min_keep_seconds:
                continue
            samples = int((seconds - start) * self.audio_rate)
            audio = np.array(audio[:samples], copy=True)
            fade = min(len(audio), int(fade_seconds * self.audio_rate))
            if fade:
                audio[-fade:] = (audio[-fade:].astype(np.float32) * np.linspace(1.0, 0.0, fade, dtype=np.float32)).astype(audio.dtype)
            frames = frames[:max(1, int(round((seconds - start) * self.fps)))]
            end_at = seconds
            if settle_to is not None and settle_count and frames.shape[1:] == np.asarray(settle_to).shape:
                frames = np.concatenate([frames, settle_frames(frames[-1], settle_to, settle_count)])
                end_at = seconds + settle_count / self.fps
            kept.append((start, end_at, audio, frames, clip_tag))
        self.clips = kept
        return affected, end_at

    def clips_tagged(self, tag: str) -> list:
        return [c for c in self.clips if c[4] == tag]

    def promise(self, start: float, end: float, generation: int) -> None:
        if generation == self.generation:
            self.promises.append((start, end))

    def last_end(self) -> float:
        return max((c[1] for c in self.clips), default=self.now())

    def active(self, seconds: float):
        while self.clips and self.clips[0][1] < seconds - 0.5:
            self.clips.pop(0)
        active = None
        for clip in self.clips:
            if clip[0] <= seconds < clip[1]:
                # A reply supersedes a filler at its scheduled start. Audio packets
                # already use this later-clip priority; video must do the same.
                active = clip
            if clip[0] > seconds:
                break
        return active

    def promised(self, seconds: float) -> bool:
        return any(a <= seconds < b for a, b in self.promises)

    # ----------------------------------------------------------------- output
    def frame_at(self, seconds: float):
        """The frame to show at `seconds` and its source ("clip", "idle" or "still")."""
        index = int(seconds * self.fps + 1e-6)          # seconds come from index / fps; floor must not lose a frame
        delta = index - self._last_index if self._last_index is not None else 0
        self._last_index = index
        for loop in self.loops.values():
            loop.advance(delta)
        clip = self.active(seconds)
        if clip is not None:
            frames = clip[3]
            image, source = frames[min(len(frames) - 1, int((seconds - clip[0]) * self.fps))], "clip:turn" if clip[4] == "turn" else "clip"
        else:
            if self.promised(seconds):
                self.stalls += 1
            name = self.mode_at(seconds)
            loop = self.loops.get(name) if self.loops.get(name) is not None and self.loops[name].ready else self.loops["front"]
            if loop.ready:
                image, source = loop.frame(), f"idle:{name if loop is self.loops.get(name) else 'front'}"
            else:
                image, source = self.still, "still"
        if self._last is not None and source != self._last[1] and image is not None:
            if source.startswith("idle"):
                # Back from a clip: clips end settled in the rest pose, which is where every
                # idle loop starts, so restart the loop there and the join is a hard cut.
                for name, loop in self.loops.items():
                    if source == f"idle:{name}":
                        loop.cursor = 0
                        image = loop.frame()
            if self.transition_frames:
                frames_for_switch = min(self.transition_frames, 4) if any("turn" in name for name in (self._last[1], source)) else self.transition_frames
                self._blend = (self._last[0], index, frames_for_switch)
        self._last = (image, source)
        if self._blend is not None and image is not None:
            since = index - self._blend[1]
            length = self._blend[2]
            if 0 <= since < length and self._blend[0] is not None and self._blend[0].shape == image.shape:
                weight = 1.0 - (since + 1) / (length + 1)
                image = (weight * self._blend[0].astype(np.float32) + (1.0 - weight) * image.astype(np.float32)).astype(image.dtype)
            else:
                self._blend = None
        return image, source

    def audio_packet(self, seconds: float, samples: int):
        out = np.zeros(samples, dtype=np.int16)
        self.active(seconds)
        for start, end, audio, _, _ in self.clips:
            lo = max(0, round((start - seconds) * self.audio_rate))
            hi = min(samples, round((end - seconds) * self.audio_rate))
            if hi > lo:
                offset = max(0, round((seconds + lo / self.audio_rate - start) * self.audio_rate))
                portion = audio[offset: offset + hi - lo]
                out[lo: lo + len(portion)] = portion
        return out


def _held(segments, crossfade: int) -> list[int]:
    return [max(0, min(int(crossfade), len(s) // 2 - 1)) for s in segments]


def idle_loop_length(segments, crossfade: int) -> int:
    """Frames in one pass of the loop (each segment minus the frames held for its dissolve)."""
    return sum(len(s) - k for s, k in zip(segments, _held(segments, crossfade)))


def idle_loop_locate(segments, position: int, crossfade: int):
    """(segment index, frame within its play region, frames of the previous segment's
    continuation that dissolve into this one) for a loop position."""
    held = _held(segments, crossfade)
    lengths = [len(s) - k for s, k in zip(segments, held)]
    j = position % sum(lengths)
    index = 0
    for index, length in enumerate(lengths):
        if j < length:
            break
        j -= length
    return index, j, held[index - 1]                             # index 0 wraps to the last segment


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
    index, j, k = idle_loop_locate(segments, position, crossfade)
    frame = segments[index][j]
    previous = segments[index - 1]
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
