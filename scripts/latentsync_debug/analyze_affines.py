#!/usr/bin/env python3
"""Analyze dumped affine-matrix sequences for frame-to-frame drift.

Paired with LATENTSYNC_DUMP_AFFINES=1 in the lipsync-latentsync service.
On static input (literally identical source frames) every frame's
affine matrix *should* be bit-identical. If it isn't, we've found the
jitter source upstream of kornia.

Usage (from repo root):
    docker compose exec -T lipsync-latentsync \\
        python /app/repo_scripts/latentsync_debug/analyze_affines.py \\
        /jobs/affine_debug/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _describe(name: str, arr: np.ndarray) -> None:
    # arr shape: (N, ...) where N = frame count
    n = arr.shape[0]
    if n < 2:
        print(f"{name}: only {n} frame(s) — nothing to compare")
        return

    base = arr[0]
    # Element-wise absolute diff vs. frame 0
    diffs = np.stack([np.abs(arr[i] - base) for i in range(n)])  # (N, ...)
    max_per_frame = diffs.reshape(n, -1).max(axis=1)             # (N,)
    mean_per_frame = diffs.reshape(n, -1).mean(axis=1)           # (N,)
    bit_identical = (diffs == 0).all()

    print(f"=== {name} ({n} frames, element shape {arr.shape[1:]}) ===")
    print(f"  bit-identical across all frames: {bit_identical}")
    print(f"  max element diff vs frame 0:")
    print(f"    overall max: {max_per_frame.max():.6e}")
    print(f"    overall mean: {mean_per_frame.mean():.6e}")
    print(f"  first 10 per-frame max diffs:")
    for i, d in enumerate(max_per_frame[:10]):
        print(f"    frame {i:3d}: max_diff={d:.6e}")
    if n > 10:
        # Also show the frame with the largest diff
        worst = int(max_per_frame.argmax())
        print(f"  worst frame: {worst}  max_diff={max_per_frame[worst]:.6e}")
        print(f"  arr[0] = {base.flatten()[:8]}")
        print(f"  arr[{worst}] = {arr[worst].flatten()[:8]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir", type=Path, help="Dir containing affines_*.npy")
    args = p.parse_args()

    if not args.dump_dir.is_dir():
        print(f"not a dir: {args.dump_dir}", file=sys.stderr)
        sys.exit(1)

    for name in (
        "affines_pre_smooth",
        "affines_post_smooth",
        "boxes_pre_smooth",
        "boxes_post_smooth",
    ):
        f = args.dump_dir / f"{name}.npy"
        if not f.exists():
            print(f"{f} not found; skipping")
            continue
        arr = np.load(f)
        _describe(name, arr)
        print()


if __name__ == "__main__":
    main()
