"""Per-person face tracks for clips with more than one person on screen.

The single face track (face_track.py) follows the largest face in each frame.
With two presenters of similar size that is wrong twice over: the generated
mouth lands on whoever is largest, and the track jumps between the two faces as
they move. This module tracks every face instead:

* every frame (25 fps grid) is scanned with the same detector + 106-point
  landmarks as the single track;
* every `sample_every` frames each face also gets an ArcFace embedding
  (buffalo_l w600k_r50, already in the models volume); the embeddings are
  clustered into identities, so a person keeps their number when they cross
  the frame or the other person walks in front;
* between sampled frames faces are linked to identities by position.

Per identity it keeps the 3-point landmarks the warp uses, the detector
confidence (0 where the person is not found) and a mouth-opening measure from
the lip landmarks. The backend matches voices to identities with the mouth
measure and renders each speaker's lines on that speaker's own track.

Identities are numbered left to right by their mean position, so the numbering
is stable for a given clip. The result is cached next to the single tracks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import face_track

log = logging.getLogger(__name__)

IDENTITY_VERSION = "insightface-buffalo_l-identities-v1"
_SUBDIR = Path("cache") / "latentsync_identities"
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

# InsightFace 2d106 mouth: 52 and 61 are the corners; (62, 60), (66, 54) and
# (70, 57) are upper/lower inner-lip pairs (checked against a frame on 7 Oct 2026).
MOUTH_CORNERS = (52, 61)
MOUTH_PAIRS = ((62, 60), (66, 54), (70, 57))


def mouth_opening(lmk106: np.ndarray) -> float:
    """Inner-lip gap divided by mouth width: scale-free, about 0 when closed."""
    pts = np.asarray(lmk106, dtype=np.float64)
    width = float(np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]]))
    if width <= 1e-6:
        return 0.0
    gap = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_PAIRS])
    return float(gap / width)


@dataclass
class Detection:
    bbox: np.ndarray            # x1, y1, x2, y2
    landmarks3: np.ndarray      # (3, 2)
    score: float
    mouth: float
    embedding: np.ndarray | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])


@dataclass
class Identities:
    landmarks: list[np.ndarray]           # per identity (N, 3, 2), gaps filled, smoothed
    visible: list[np.ndarray]             # per identity (N,) detector confidence, 0 = absent
    mouth: list[np.ndarray]               # per identity (N,) mouth opening, NaN = absent
    presence: list[float]                 # fraction of frames the identity is found
    centers: list[list[float]]            # mean bbox centre, source pixels
    frames: int
    fps: int
    meta: dict = field(default_factory=dict)

    def summary(self) -> list[dict]:
        return [{"id": k, "presence": round(self.presence[k], 4),
                 "center": [round(c, 1) for c in self.centers[k]]} for k in range(len(self.landmarks))]


# ------------------------------------------------------------------ clustering
def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.55, min_share: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Average-linkage clustering on cosine distance. Returns (labels, centroids);
    clusters with fewer than `min_share` of the samples are dropped (label -1)."""
    from sklearn.cluster import AgglomerativeClustering

    x = np.asarray(embeddings, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)
    if len(x) == 1:
        return np.zeros(1, dtype=int), x.copy()
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=threshold).fit_predict(x)
    keep = [c for c in np.unique(labels) if (labels == c).mean() >= min_share]
    remap = {c: i for i, c in enumerate(sorted(keep, key=lambda c: -(labels == c).sum()))}
    out = np.array([remap.get(c, -1) for c in labels])
    centroids = np.stack([x[out == i].mean(axis=0) for i in range(len(remap))]) if remap else np.zeros((0, x.shape[1]))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
    return out, centroids


def _match(cost: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one matching (rows = detections, cols = identities),
    pairs above `limit` left unmatched."""
    if cost.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= limit]


def assign(frames: list[list[Detection]], centroids: np.ndarray, sample_every: int,
           min_similarity: float = 0.3, max_jump: float = 0.75) -> list[dict[int, Detection]]:
    """Per frame, {identity: detection}. Frames with embeddings are matched by
    similarity to the identity centroids; the others by position, to where each
    identity was last seen (at most `max_jump` face widths away)."""
    k = len(centroids)
    out: list[dict[int, Detection]] = []
    last: dict[int, Detection] = {}
    for faces in frames:
        found: dict[int, Detection] = {}
        if faces and k:
            with_emb = [d for d in faces if d.embedding is not None]
            if with_emb and len(with_emb) == len(faces):
                emb = np.stack([d.embedding for d in faces])
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
                cost = 1.0 - emb @ centroids.T
                for r, c in _match(cost, 1.0 - min_similarity):
                    found[c] = faces[r]
            else:
                ids = list(last)
                if ids:
                    cost = np.array([[np.linalg.norm(d.center - last[i].center) / max(last[i].width, 1.0)
                                      for i in ids] for d in faces])
                    for r, c in _match(cost, max_jump):
                        found[ids[c]] = faces[r]
        last.update(found)
        out.append(found)
    return out


def finalize(assigned: list[dict[int, Detection]], k: int, fps: int, smooth_window: int) -> Identities:
    n = len(assigned)
    landmarks, visible, mouth, presence, centers = [], [], [], [], []
    for i in range(k):
        per = [frame.get(i) for frame in assigned]
        lm = [d.landmarks3 if d is not None else None for d in per]
        filled, missing = face_track.fill_missing(lm)
        if not filled:
            filled = [np.zeros((3, 2), np.float32)] * n
        vis = np.array([d.score if d is not None else 0.0 for d in per], dtype=np.float32)
        mo = np.array([d.mouth if d is not None else np.nan for d in per], dtype=np.float32)
        seen = [d.center for d in per if d is not None]
        landmarks.append(face_track.smooth(np.stack(filled).astype(np.float32), smooth_window))
        visible.append(vis)
        mouth.append(mo)
        presence.append(float((vis > 0).mean()) if n else 0.0)
        centers.append(np.mean(seen, axis=0).tolist() if seen else [0.0, 0.0])
    # Number identities left to right.
    order = sorted(range(k), key=lambda i: centers[i][0])
    pick = lambda xs: [xs[i] for i in order]  # noqa: E731
    return Identities(pick(landmarks), pick(visible), pick(mouth), pick(presence), pick(centers), n, fps)


# ------------------------------------------------------------------ build
class Scanner:
    """Detector + landmarks (the pipeline's FaceDetector) and ArcFace."""

    def __init__(self, detector, landmarks3_from_106):
        self.detector = detector
        self.landmarks3 = landmarks3_from_106
        self._recognizer = None

    def recognizer(self):
        if self._recognizer is None:
            from insightface.model_zoo import get_model
            from gpu_runtime import ort_cuda_provider

            root = os.environ.get("INSIGHTFACE_ROOT", os.path.join(os.environ.get("MODEL_CACHE_DIR", "/models"), "insightface"))
            path = os.path.join(root, "models", "buffalo_l", "w600k_r50.onnx")
            if not os.path.exists(path):
                raise FileNotFoundError(f"face recognition model missing: {path}")
            model = get_model(path, providers=[ort_cuda_provider(), "CPUExecutionProvider"])
            model.prepare(ctx_id=0)
            self._recognizer = model
        return self._recognizer

    def detect(self, frame_rgb: np.ndarray, embed: bool, threshold: float = 0.5) -> list[Detection]:
        faces = self.detector.app.get(frame_rgb)
        out = []
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            w, h = x2 - x1, y2 - y1
            if w < 50 or h < 80 or not (0.2 <= w / h <= 1.5) or f.det_score < threshold:
                continue
            lmk = np.asarray(f.landmark_2d_106, dtype=np.float32)
            d = Detection(np.array([x1, y1, x2, y2], np.float32), self.landmarks3(np.round(lmk)).astype(np.float32),
                          float(f.det_score), mouth_opening(lmk))
            if embed and getattr(f, "kps", None) is not None:
                from insightface.utils import face_align

                crop = face_align.norm_crop(np.ascontiguousarray(frame_rgb[..., ::-1]), landmark=f.kps, image_size=112)
                d.embedding = np.asarray(self.recognizer().get_feat(crop)).reshape(-1).astype(np.float32)
            out.append(d)
        return out


def build(source: Path, fps: int, scanner: Scanner, *, smooth_window: int, frame_budget_bytes: int,
          sample_every: int = 10, progress=None) -> Identities:
    from gpu_runtime.media import iter_frames

    started = time.perf_counter()
    frames: list[list[Detection]] = []
    for i, frame in enumerate(iter_frames(source, fps=fps, max_frame_bytes=frame_budget_bytes)):
        frames.append(scanner.detect(frame, embed=(i % sample_every == 0)))
        if progress and len(frames) % 250 == 0:
            progress(len(frames))
    if not frames:
        raise RuntimeError(f"no frames decoded from {source}")
    sampled = [d for faces in frames for d in faces if d.embedding is not None]
    if not sampled:
        raise RuntimeError(f"no face detected in {source}")
    labels, centroids = cluster_embeddings(np.stack([d.embedding for d in sampled]))
    assigned = assign(frames, centroids, sample_every)
    result = finalize(assigned, len(centroids), fps, smooth_window)
    result.meta = {"seconds": round(time.perf_counter() - started, 1), "samples": len(sampled),
                   "clusters": int(len(centroids)), "source_name": Path(source).name}
    log.info("face identities: %d people over %d frames in %.1fs (presence %s)", len(centroids), len(frames),
             result.meta["seconds"], [round(p, 2) for p in result.presence])
    return result


def cache_path(model_cache_dir: Path, key: str) -> Path:
    return Path(model_cache_dir) / _SUBDIR / f"{key}.npz"


def key_for(source: Path, fps: int, smooth_window: int) -> str:
    import hashlib

    return hashlib.sha256(f"{face_track.source_digest(source)}|{IDENTITY_VERSION}|fps={fps}|smooth={smooth_window}".encode()).hexdigest()


def load_or_build(source: Path, *, model_cache_dir: Path, fps: int, scanner_factory, smooth_window: int,
                  frame_budget_bytes: int) -> Identities:
    source = Path(source)
    key = key_for(source, fps, smooth_window)
    path = cache_path(model_cache_dir, key)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if path.exists():
            try:
                with np.load(path) as data:
                    if str(data["version"]) == IDENTITY_VERSION:
                        k = int(data["count"])
                        meta = json.loads(str(data["meta"]))
                        return Identities([data[f"landmarks_{i}"] for i in range(k)], [data[f"visible_{i}"] for i in range(k)],
                                          [data[f"mouth_{i}"] for i in range(k)], meta["presence"], meta["centers"],
                                          int(data["frames"]), fps, meta)
            except Exception as exc:
                log.warning("identity cache unreadable (%s); rebuilding", exc)
        result = build(source, fps, scanner_factory(), smooth_window=smooth_window, frame_budget_bytes=frame_budget_bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        arrays = {"version": np.array(IDENTITY_VERSION), "count": np.array(len(result.landmarks)),
                  "frames": np.array(result.frames)}
        for i in range(len(result.landmarks)):
            arrays[f"landmarks_{i}"] = result.landmarks[i]
            arrays[f"visible_{i}"] = result.visible[i]
            arrays[f"mouth_{i}"] = result.mouth[i]
        meta = dict(result.meta, presence=result.presence, centers=result.centers)
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, path)
        return result
