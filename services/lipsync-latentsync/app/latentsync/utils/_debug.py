"""Debug-dump helper for the LatentSync pipeline.

Enabled via ``LATENTSYNC_DEBUG_DUMP=1``. Saves intermediates from face
detection, affine transform, restore_img, and the final composite as
PNG files to ``LATENTSYNC_DEBUG_DIR`` (default ``/jobs/latentsync_debug/``).

Dumps are rate-limited to ``LATENTSYNC_DEBUG_FRAME_LIMIT`` per stage
(default 3) so a long run doesn't fill the disk — the first few frames
are almost always enough to diagnose geometry or precision issues.

Use case: after a run that produced bad output, enable this, re-run,
pull the PNGs out of /jobs/latentsync_debug/ and inspect. Nothing
downstream of the debug dump reads its output — it's purely
observational, never affects the live pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _bool_env(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


_DEBUG_ENABLED = _bool_env("LATENTSYNC_DEBUG_DUMP")
_DEBUG_DIR = Path(os.environ.get("LATENTSYNC_DEBUG_DIR", "/jobs/latentsync_debug"))
_DEBUG_FRAME_LIMIT = int(os.environ.get("LATENTSYNC_DEBUG_FRAME_LIMIT", "3"))

# Per-name counter so each debug category gets its own rate limit.
_counters: dict[str, int] = {}


def debug_enabled() -> bool:
    return _DEBUG_ENABLED


def _to_uint8_hwc(image) -> np.ndarray:
    """Convert any common tensor/array layout to HxWxC uint8 for cv2.imwrite."""
    # torch → numpy
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    # Float ranges: [0,1] → [0,255]; [-1,1] → [0,255]; [0,255] → clip.
    if image.dtype.kind == "f":
        lo, hi = float(image.min()), float(image.max())
        if lo >= 0.0 and hi <= 1.0 + 1e-3:
            image = image * 255.0
        elif lo >= -1.0 - 1e-3 and hi <= 1.0 + 1e-3:
            image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Drop leading batch dim.
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    # CHW → HWC when channels are 1/3/4.
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    # Single-channel: squeeze so cv2 writes grayscale.
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    return image


def dump(name: str, image, frame_idx: int | None = None) -> None:
    """Write `image` to the debug dir as ``<name>_<idx>.png``.

    No-op when LATENTSYNC_DEBUG_DUMP is unset, or when this `name` has
    already been written LATENTSYNC_DEBUG_FRAME_LIMIT times.
    """
    if not _DEBUG_ENABLED:
        return

    count = _counters.get(name, 0)
    if count >= _DEBUG_FRAME_LIMIT:
        return
    _counters[name] = count + 1

    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        arr = _to_uint8_hwc(image)
        idx = frame_idx if frame_idx is not None else count
        path = _DEBUG_DIR / f"{name}_{idx:03d}.png"
        # cv2 expects BGR for 3-channel images; we're being loose here
        # since this is debug output and most inputs are already BGR
        # (OpenCV native) or RGB-but-close-enough for visual inspection.
        cv2.imwrite(str(path), arr)
    except Exception as e:
        # Debug dump must never break the live pipeline.
        print(f"[latentsync debug] failed to dump {name}: {e}")


def dump_annotated_frame(
    name: str,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    landmarks: np.ndarray | None = None,
    frame_idx: int | None = None,
) -> None:
    """Save `frame` with the bbox and landmark points drawn on top.

    Fastest signal for "is face detection landing in the right place
    and are the landmarks on actual features?" The reviewer called this
    out as the most likely failure mode for challenging source video
    (car interiors, side lighting, partial occlusion).
    """
    if not _DEBUG_ENABLED:
        return

    count = _counters.get(name, 0)
    if count >= _DEBUG_FRAME_LIMIT:
        return
    _counters[name] = count + 1

    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        annotated = np.ascontiguousarray(frame.copy())
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if landmarks is not None:
            for pt in np.asarray(landmarks).astype(int):
                cv2.circle(annotated, tuple(pt), 2, (0, 0, 255), -1)
        idx = frame_idx if frame_idx is not None else count
        path = _DEBUG_DIR / f"{name}_{idx:03d}.png"
        cv2.imwrite(str(path), annotated)
    except Exception as e:
        print(f"[latentsync debug] failed to dump {name}: {e}")
