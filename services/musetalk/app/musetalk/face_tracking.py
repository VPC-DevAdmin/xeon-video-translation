"""Face detection via InsightFace SCRFD.

Previously this module wrapped `face-alignment` (SFD + FAN). That worked but
was dominated by FAN's CPU cost — on the demo's 53-frame clip, face detection
alone took 9 minutes 23 seconds, 47% of the total MuseTalk wall time.

SCRFD is a single-pass anchor-free detector distributed with InsightFace.
It runs through ONNX Runtime at ~50 ms per frame on a Xeon core, roughly two
orders of magnitude faster than FAN with comparable box accuracy for our
use case (front-facing talking heads). Landmarks are available from the
`buffalo_l` pack if we need them later; this module only returns bboxes
today.

The upstream library downloads weights to `~/.insightface/models/` on first
`prepare()`. The service image mounts `HOME=/root` and keeps that path
persistent across restarts; the download script in `scripts/` can also
pre-fetch so cold starts don't stall.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    landmarks: np.ndarray | None           # reserved for later (buffalo_l 2d106); None for detection-only
    face_box: tuple[int, int, int, int] | None  # (x1, y1, x2, y2), clipped to frame bounds
    score: float | None = None             # SCRFD detection confidence, if available
    # Five-point landmarks from SCRFD: left-eye, right-eye, nose, left-mouth,
    # right-mouth. Used by the CodeFormer face-alignment path.
    kps: np.ndarray | None = None          # shape (5, 2) float32, or None


# --------------------------------------------------------------------------- #
# Aligner loading
# --------------------------------------------------------------------------- #


def _insightface_root() -> Path:
    """Cache/weights path. We anchor it under MODEL_CACHE_DIR so it survives
    container rebuilds and is discoverable alongside the other weight bundles.
    """
    root = Path(os.environ.get("MODEL_CACHE_DIR", "/models")) / "insightface"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_aligner(
    device: str = "cpu",
    det_size: tuple[int, int] = (640, 640),
    model_name: str = "buffalo_l",
):
    """Return a configured `insightface.app.FaceAnalysis` instance.

    Only the detection module is loaded — we don't need recognition embeddings
    or identity clustering. Weights land in `MODEL_CACHE_DIR/insightface/` on
    first use (~285 MB for the buffalo_l pack).
    """
    from insightface.app import FaceAnalysis

    if device != "cpu":
        log.warning("insightface aligner: only CPU is supported in this build")

    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name=model_name,
        root=str(_insightface_root()),
        providers=providers,
        allowed_modules=["detection"],
    )
    # ctx_id=-1 selects the CPU execution provider.
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


# --------------------------------------------------------------------------- #
# Per-frame detection
# --------------------------------------------------------------------------- #


def detect_frame(aligner, frame_bgr: np.ndarray) -> FrameDetection:
    """Run SCRFD on a single BGR frame. Returns largest-face bbox + score."""
    # insightface.app.FaceAnalysis.get expects BGR — same as OpenCV.
    faces = aligner.get(frame_bgr)
    if not faces:
        return FrameDetection(landmarks=None, face_box=None, score=None)

    # Pick the largest detection. Confidence-weighted would be another option,
    # but for single-subject clips "biggest face" is a safe heuristic and
    # avoids bouncing between a foreground face and an accidental reflection.
    def area(face) -> float:
        x1, y1, x2, y2 = face.bbox[:4]
        return float((x2 - x1) * (y2 - y1))

    face = max(faces, key=area)
    x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]

    # Clip to the frame. SCRFD occasionally overshoots by a few pixels on
    # faces at the edge; cropping downstream would OOM if those leak through.
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return FrameDetection(landmarks=None, face_box=None, score=None)

    score = float(face.det_score) if hasattr(face, "det_score") else None
    kps = None
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32).reshape(-1, 2)
    return FrameDetection(
        landmarks=None,
        face_box=(x1, y1, x2, y2),
        score=score,
        kps=kps,
    )


def detect_batch(aligner, frames_bgr: Sequence[np.ndarray]) -> list[FrameDetection]:
    """Run detection frame-by-frame. SCRFD doesn't meaningfully batch on CPU
    through ONNX Runtime, so a simple loop is fine. Returns one FrameDetection
    per input frame.
    """
    return [detect_frame(aligner, f) for f in frames_bgr]
