"""Bounded audio/video playout on one monotonic clock; interruption flushes both."""

import asyncio
import time
from collections import deque
from fractions import Fraction
import numpy as np
import av
from aiortc import MediaStreamTrack


class Playback:
    def __init__(self, image):
        self.epoch = time.monotonic()
        self.image = image
        self.clips = deque()
        self.generation = 0
        self.first_audio_seconds = None
        self.last_end = 0.0
        self.video_frames_sent = 0
        self.video_frames_skipped = 0

    def interrupt(self):
        self.generation += 1
        self.clips.clear()
        return self.generation

    async def enqueue(self, audio, frames, generation):
        if not len(audio) or not len(frames):
            raise ValueError("avatar audio and frames must be nonempty")
        while self.clips and self.clips[-1][1] > time.monotonic() - self.epoch + 2:
            if generation != self.generation:
                return
            await asyncio.sleep(0.02)
        if generation != self.generation:
            return
        now = time.monotonic() - self.epoch + 0.08
        start = max(now, self.clips[-1][1] if self.clips else now)
        self.clips.append((start, start + len(audio) / 48000, audio, frames))
        self.last_end = start + len(audio) / 48000
        return {
            "start": start,
            "end": self.last_end,
            "base": self.first_audio_seconds or 0.0,
        }

    def active(self, seconds):
        while self.clips and self.clips[0][1] < seconds - 0.2:
            self.clips.popleft()
        return next((c for c in self.clips if c[0] <= seconds < c[1]), None)


class AudioOutput(MediaStreamTrack):
    kind = "audio"

    def __init__(self, playback):
        super().__init__()
        self.playback, self.index = playback, 0

    async def recv(self):
        self.index = max(self.index, int((time.monotonic() - self.playback.epoch) * 50))
        pts = self.index * 960
        seconds = pts / 48000
        if self.playback.first_audio_seconds is None:
            self.playback.first_audio_seconds = seconds
        await asyncio.sleep(max(0, self.playback.epoch + seconds - time.monotonic()))
        samples = np.zeros(960, dtype=np.int16)
        # Sample each overlapping clip, including boundaries within this packet.
        self.playback.active(seconds)
        for start, end, audio, _ in self.playback.clips:
            lo = max(0, round((start - seconds) * 48000))
            hi = min(960, round((end - seconds) * 48000))
            if hi > lo:
                offset = max(0, round((seconds + lo / 48000 - start) * 48000))
                portion = audio[offset : offset + hi - lo]
                samples[lo : lo + len(portion)] = portion
        frame = av.AudioFrame.from_ndarray(
            samples[None, :], format="s16", layout="mono"
        )
        frame.sample_rate, frame.pts, frame.time_base = 48000, pts, Fraction(1, 48000)
        self.index += 1
        return frame


class VideoOutput(MediaStreamTrack):
    kind = "video"

    def __init__(self, playback):
        super().__init__()
        self.playback, self.index = playback, 0

    async def recv(self):
        scheduled = max(self.index, int((time.monotonic() - self.playback.epoch) * 25))
        if self.playback.video_frames_sent:
            self.playback.video_frames_skipped += scheduled - self.index
        self.index = scheduled
        self.playback.video_frames_sent += 1
        seconds = self.index / 25
        await asyncio.sleep(max(0, self.playback.epoch + seconds - time.monotonic()))
        clip = self.playback.active(seconds)
        image = self.playback.image
        if clip:
            frame_index = min(len(clip[3]) - 1, int((seconds - clip[0]) * 25))
            image = clip[3][frame_index]
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = self.index * 3600, Fraction(1, 90000)
        self.index += 1
        return frame
