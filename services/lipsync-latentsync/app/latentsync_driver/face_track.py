"""Shared face track: one landmark pass per source clip, reused by every window.

The windowed renderer hands LatentSync 16 s cuts of the same source and each
cut used to run face detection on its own frames. Detection is the same work
whether it happens per window or once, but doing it once per *source* gives
three things the per-window pass cannot:

* a landmark trajectory smoothed over the whole clip, so the affine warp has
  no discontinuity at window seams;
* a cache keyed on the source content, so the second mode run on the same
  clip (fast then quality), a revision, or a resumed job skips detection
  entirely;
* a detection pass that streams frames instead of holding a window in RAM.

Tracks live under ``MODEL_CACHE_DIR/cache/latentsync_tracks/<key>.npz`` with
the 3-point landmarks (left eye, right eye, nose) per 25 fps frame. Only
geometry is cached, never pixels; the warp itself still runs on the window's
own frames because they are what gets pasted back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TRACK_VERSION = "insightface-buffalo_l-3pt-v1"
_SUBDIR = Path("cache") / "latentsync_tracks"
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def source_digest(path: Path) -> str:
    """Whole-file SHA-256: equal size and prefix do not imply equal video."""
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def track_key(path: Path, fps: int, smooth_window: int) -> str:
    return hashlib.sha256(
        f"{source_digest(path)}|{TRACK_VERSION}|fps={fps}|smooth={smooth_window}".encode()
    ).hexdigest()


def cache_path(model_cache_dir: Path, key: str) -> Path:
    return Path(model_cache_dir) / _SUBDIR / f"{key}.npz"


def fill_missing(per_frame: list) -> tuple[list, list[int]]:
    """Forward-fill None detections, back-fill leading gaps. Mirrors the
    pipeline's gap handling so cached and uncached runs agree."""
    missing = [i for i, l in enumerate(per_frame) if l is None]
    if not missing:
        return per_frame, missing
    last = None
    filled = []
    for lmk in per_frame:
        if lmk is not None:
            last = lmk
        filled.append(last)
    first = next((l for l in filled if l is not None), None)
    if first is None:
        return [], missing
    return [first if l is None else l for l in filled], missing


def smooth(landmarks: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay (order 2) along time for each of the 6 coordinates.

    Same filter and parameters as the pipeline's per-window smoother, now
    applied across the whole clip so window seams share one trajectory.
    """
    n = len(landmarks)
    if window <= 1 or n < 3:
        return landmarks
    eff = min(int(window), n)
    if eff % 2 == 0:
        eff -= 1
    if eff <= 2:
        return landmarks
    from scipy.signal import savgol_filter

    flat = landmarks.reshape(n, -1).astype(np.float64)
    return savgol_filter(flat, eff, 2, axis=0, mode="interp").reshape(n, 3, 2).astype(np.float32)


def slice_for_window(track: np.ndarray, offset: int, frames: int) -> np.ndarray:
    """Landmarks for window frames [offset, offset+frames) of the source track.

    The window encoder pads past the end of the source by cloning the last
    frame (tpad), so a short slice is extended with the last landmark.
    """
    if offset < 0:
        raise ValueError("window offset must be non-negative")
    if frames <= 0:
        return track[:0]
    if offset >= len(track):
        if len(track) == 0:
            raise ValueError("empty face track")
        return np.repeat(track[-1:], frames, axis=0)
    part = track[offset : offset + frames]
    if len(part) < frames:
        part = np.concatenate([part, np.repeat(part[-1:], frames - len(part), axis=0)])
    return part


def build(source: Path, fps: int, extract, *, smooth_window: int, max_miss_ratio: float,
          frame_budget_bytes: int, progress=None) -> dict:
    """Stream the source at `fps` and run `extract(frame) -> (3,2) | None`."""
    from gpu_runtime.media import iter_frames

    started = time.perf_counter()
    per_frame = []
    for frame in iter_frames(source, fps=fps, max_frame_bytes=frame_budget_bytes):
        per_frame.append(extract(frame))
        if progress and len(per_frame) % 100 == 0:
            progress(len(per_frame))
    total = len(per_frame)
    if total == 0:
        raise RuntimeError(f"no frames decoded from {source}")
    filled, missing = fill_missing(per_frame)
    if not filled:
        raise RuntimeError(f"no face detected in any of {total} frames of {source}")
    ratio = len(missing) / total
    if ratio > max_miss_ratio:
        raise RuntimeError(
            f"{len(missing)}/{total} frames ({ratio:.1%}) lack a detected face, exceeding "
            f"LATENTSYNC_MAX_MISSING_FACE_RATIO={max_miss_ratio}"
        )
    landmarks = smooth(np.stack(filled).astype(np.float32), smooth_window)
    return {
        "landmarks": landmarks,
        "fps": fps,
        "frames": total,
        "missing": len(missing),
        "seconds": time.perf_counter() - started,
    }


def load_or_build(source: Path, *, model_cache_dir: Path, fps: int, extract,
                  smooth_window: int, max_miss_ratio: float, frame_budget_bytes: int) -> np.ndarray:
    """Return the (N,3,2) landmark track for `source`, building it once."""
    source = Path(source)
    key = track_key(source, fps, smooth_window)
    path = cache_path(model_cache_dir, key)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if path.exists():
            try:
                with np.load(path) as data:
                    if str(data["version"]) == TRACK_VERSION:
                        log.info("face track cache hit: %s (%d frames)", path.name, len(data["landmarks"]))
                        return data["landmarks"]
            except Exception as exc:  # corrupt cache: rebuild
                log.warning("face track cache unreadable (%s); rebuilding", exc)
        result = build(source, fps, extract, smooth_window=smooth_window,
                       max_miss_ratio=max_miss_ratio, frame_budget_bytes=frame_budget_bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, landmarks=result["landmarks"], version=np.array(TRACK_VERSION),
                 fps=np.array(fps), frames=np.array(result["frames"]),
                 meta=np.array(json.dumps({"missing": result["missing"], "seconds": result["seconds"],
                                           "source_name": source.name})))
        os.replace(tmp, path)
        log.info("face track built: %d frames at %d fps in %.1fs (%d without a face) -> %s",
                 result["frames"], fps, result["seconds"], result["missing"], path.name)
        return result["landmarks"]
