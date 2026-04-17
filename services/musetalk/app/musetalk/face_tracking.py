"""Face detection + landmark extraction using `face-alignment`.

Replaces the upstream MuseTalk preprocessing that used `mmpose` / DWPose.
Same 68-point 2D landmark layout (FAN), so downstream MuseTalk code that
derives face-box and crop-box coordinates is a drop-in substitute.

On CPU, `face-alignment` runs SFD for detection and FAN for landmarks at
roughly 0.5-2 FPS depending on frame size. The detector is the dominant
cost — we accept that for this demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FrameDetection:
    landmarks: np.ndarray | None          # (68, 2) or None if no face
    face_box: tuple[int, int, int, int] | None  # (x1, y1, x2, y2), tight bbox of landmarks


def _face_alignment_cls():
    """Lazy import: face_alignment pulls torch/torchvision on import."""
    import face_alignment
    return face_alignment


def build_aligner(device: str = "cpu"):
    fa_mod = _face_alignment_cls()
    return fa_mod.FaceAlignment(
        fa_mod.LandmarksType.TWO_D,
        flip_input=False,
        device=device,
    )


def detect_frame(aligner, frame_bgr: np.ndarray) -> FrameDetection:
    """Run FAN on a single BGR frame. Returns landmarks + tight bbox."""
    # face-alignment expects RGB.
    rgb = frame_bgr[:, :, ::-1]
    lms_list = aligner.get_landmarks_from_image(rgb)
    if not lms_list:
        return FrameDetection(landmarks=None, face_box=None)

    # Pick the largest-area landmark set if multiple detections.
    def area(lms: np.ndarray) -> float:
        x1, y1 = lms.min(axis=0)
        x2, y2 = lms.max(axis=0)
        return float((x2 - x1) * (y2 - y1))

    lms = max(lms_list, key=area)
    x1, y1 = lms.min(axis=0)
    x2, y2 = lms.max(axis=0)

    # Clip to frame bounds so downstream crops don't underflow.
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    return FrameDetection(landmarks=lms.astype(np.float32), face_box=(x1, y1, x2, y2))


def detect_batch(aligner, frames_bgr: Sequence[np.ndarray]) -> list[FrameDetection]:
    """Run detection frame-by-frame. face-alignment doesn't meaningfully
    batch on CPU so a loop is as fast as anything fancier."""
    return [detect_frame(aligner, f) for f in frames_bgr]
