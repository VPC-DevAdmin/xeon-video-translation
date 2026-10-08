"""Audio chunking and session bookkeeping for the FlashHead render service (no torch)."""

from __future__ import annotations

import base64
from collections import deque

import numpy as np


class ChunkSpec:
    """How much audio one render call consumes and how many frames it yields."""

    def __init__(self, frame_num: int, motion_frames: int, fps: int, sample_rate: int, cached_seconds: int):
        self.frames = int(frame_num - motion_frames)          # new frames per chunk (28 for Pro)
        self.fps = int(fps)
        self.sample_rate = int(sample_rate)
        self.samples = self.frames * self.sample_rate // self.fps   # 17920 at 16 kHz
        self.seconds = self.frames / self.fps
        self.cached_samples = int(cached_seconds) * self.sample_rate
        self.cached_frames = int(cached_seconds) * self.fps
        self.frame_num = int(frame_num)

    def as_dict(self) -> dict:
        return {"frames_per_chunk": self.frames, "samples_per_chunk": self.samples,
                "seconds_per_chunk": round(self.seconds, 4), "fps": self.fps, "sample_rate": self.sample_rate}


def decode_pcm(pcm_b64: str) -> np.ndarray:
    """Base64 PCM16 mono -> float32 in [-1, 1]."""
    raw = base64.b64decode(pcm_b64)
    if len(raw) % 2:
        raise ValueError("PCM16 payload has an odd byte count")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def fit_chunk(samples: np.ndarray, spec: ChunkSpec) -> tuple[np.ndarray, int]:
    """Pad or trim audio to exactly one chunk; returns (audio, frames_covered_by_real_audio)."""
    if len(samples) > spec.samples:
        samples = samples[: spec.samples]
    covered = int(np.ceil(len(samples) * spec.fps / spec.sample_rate)) if len(samples) else 0
    covered = min(spec.frames, covered)
    if len(samples) < spec.samples:
        samples = np.pad(samples, (0, spec.samples - len(samples)))
    return samples.astype(np.float32, copy=False), covered


class AudioContext:
    """The rolling window of recent audio FlashHead conditions each chunk on."""

    def __init__(self, spec: ChunkSpec):
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self.cache = deque(np.zeros(self.spec.cached_samples, np.float32), maxlen=self.spec.cached_samples)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self.cache.extend(chunk)
        return np.asarray(self.cache, dtype=np.float32)

    @property
    def window(self) -> tuple[int, int]:
        end = self.spec.cached_frames
        return end - self.spec.frame_num, end
