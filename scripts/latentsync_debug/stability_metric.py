#!/usr/bin/env python3
"""Per-frame stability metric for LatentSync debugging.

Takes a video, detects face landmarks on every frame, computes
frame-to-frame pixel displacement of a fixed set of reference points
(eye corners, nose tip, mouth corners). Reports mean/max/std.

Run on any of:
  - The source clip (baseline: how much did the subject actually move?)
  - The output clip (current state)
  - An intermediate artifact (e.g. stabilized.mp4)

Then A/B test fixes against the number instead of squinting at frames.

Usage (inside the lipsync-latentsync container):
    python /app/scripts/latentsync_debug/stability_metric.py /path/to/video.mp4

Or directly outside the container, if the host has the deps (not
likely unless you've installed them; the container is the easier path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Landmark indices (InsightFace 2D-106 model). Rough correspondences:
#   33:  inner-right eye corner (from subject's POV: their right)
#   37:  outer-right eye corner
#   87:  inner-left eye corner
#   93:  outer-left eye corner
#   54:  nose tip
#   72:  right mouth corner
#   78:  left mouth corner
# These are the points least affected by talking motion — stable
# anchors for measuring unintended jitter.
STABLE_LANDMARKS = [33, 37, 87, 93, 54, 72, 78]


def _load_face_analyzer():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        root="checkpoints/auxiliary",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(512, 512))
    return app


def _largest_face(faces):
    best = None
    best_area = 0
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    return best


def collect_landmarks(video_path: Path) -> np.ndarray:
    """Return an (N, L, 2) array of landmarks for the stable landmark
    set, where N is frame count and L is len(STABLE_LANDMARKS).

    Frames with no detected face get NaN'd so downstream stats can
    ignore them.
    """
    app = _load_face_analyzer()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    out = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        face = _largest_face(faces)
        if face is None:
            out.append(np.full((len(STABLE_LANDMARKS), 2), np.nan))
        else:
            lmk = face.landmark_2d_106  # (106, 2)
            out.append(np.array([lmk[i] for i in STABLE_LANDMARKS]))
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  frame {frame_idx}...", file=sys.stderr)
    cap.release()
    return np.stack(out, axis=0)  # (N, L, 2)


def compute_stability(landmarks: np.ndarray) -> dict:
    """Frame-to-frame displacement stats per landmark.

    For each consecutive frame pair, compute Euclidean distance for
    each of the L stable landmarks. Then summarize:
      - per-landmark mean + max + std
      - overall mean + max + 95th percentile

    NaN frames (face not detected) are skipped.
    """
    n, l, _ = landmarks.shape
    # Pairwise diffs: (N-1, L, 2)
    diffs = landmarks[1:] - landmarks[:-1]
    # Euclidean distance per landmark: (N-1, L)
    dists = np.sqrt((diffs ** 2).sum(axis=-1))

    # Mask out NaN frame pairs.
    valid = ~np.isnan(dists)

    per_landmark = []
    for i in range(l):
        v = dists[:, i][valid[:, i]]
        if len(v) == 0:
            per_landmark.append({"count": 0})
            continue
        per_landmark.append({
            "count": int(len(v)),
            "mean_px": float(v.mean()),
            "max_px": float(v.max()),
            "std_px": float(v.std()),
        })

    # Overall (all valid landmark-pairs across all frames).
    all_v = dists[valid]
    if len(all_v) == 0:
        overall = {"count": 0}
    else:
        overall = {
            "count": int(len(all_v)),
            "mean_px": float(all_v.mean()),
            "max_px": float(all_v.max()),
            "p50_px": float(np.percentile(all_v, 50)),
            "p95_px": float(np.percentile(all_v, 95)),
            "p99_px": float(np.percentile(all_v, 99)),
            "std_px": float(all_v.std()),
        }

    return {
        "frames": int(n),
        "frame_pairs": int(n - 1),
        "stable_landmarks": STABLE_LANDMARKS,
        "per_landmark": per_landmark,
        "overall": overall,
    }


def main():
    p = argparse.ArgumentParser(
        description="Measure per-frame face stability in a video.",
    )
    p.add_argument("video_path", type=Path, help="Path to an mp4 / mov")
    p.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of the human-readable summary",
    )
    args = p.parse_args()

    if not args.video_path.exists():
        print(f"not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"==> Running InsightFace landmark detection on {args.video_path.name}", file=sys.stderr)
    lmk = collect_landmarks(args.video_path)
    print(f"==> Got {lmk.shape[0]} frames, computing stability", file=sys.stderr)
    stats = compute_stability(lmk)

    if args.json:
        json.dump(stats, sys.stdout, indent=2)
        print()
        return

    # Human readable.
    o = stats["overall"]
    print()
    print(f"File: {args.video_path}")
    print(f"Frames:       {stats['frames']}")
    print(f"Frame pairs:  {stats['frame_pairs']}")
    print()
    print("Per-frame landmark displacement (pixels; lower = more stable):")
    if o.get("count"):
        print(f"  mean: {o['mean_px']:>7.2f}   p50:  {o['p50_px']:>7.2f}")
        print(f"  p95:  {o['p95_px']:>7.2f}   p99:  {o['p99_px']:>7.2f}")
        print(f"  max:  {o['max_px']:>7.2f}   std:  {o['std_px']:>7.2f}")
    else:
        print("  (no valid frame pairs — no face detected?)")
    print()
    print("Interpretation cheat-sheet:")
    print("  mean < 1.0 px   : rock-solid — face is at sub-pixel stability")
    print("  mean 1-2 px     : excellent — typical of high-quality source")
    print("  mean 2-4 px     : noticeable jitter — start investigating")
    print("  mean 4-8 px     : 'bouncy face' — definitely visible to viewers")
    print("  mean > 8 px     : subject is moving, or pipeline is broken")


if __name__ == "__main__":
    main()
