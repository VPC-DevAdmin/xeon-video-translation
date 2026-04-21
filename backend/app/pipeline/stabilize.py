"""Optional pre-stabilization pass for the source video.

Removes handheld camera shake from the upload before downstream stages
(transcribe, tts, lipsync) see it. The lipsync stage in particular
benefits: a stable source means landmark detection is more consistent
frame-to-frame, which in turn means the affine matrix driving the UNet
warp-back is more stable, which means less per-frame "face breathing"
in the final composite.

When artifacts would be worse than the jitter
---------------------------------------------
Stabilization is not free. For clips with a stationary subject and
light handheld wobble (most phone selfies), it's nearly perfect. For
clips with more dramatic subject motion, it can introduce:

  - A "floaty" feel on intentional motion (head turns get smoothed)
  - Brief warping around fast motion (vidstab does a global per-frame
    transform; rapid subject motion against a static background can
    produce visible warp at the edges)
  - Snap-back when motion exceeds the smoothing window
  - Black borders on frame edges (mitigated by optzoom=1 auto-crop)

The `smoothing` parameter directly trades stability for responsiveness.
Lower = more jitter removed but intentional motion stays crisp; higher
= locked-off tripod feel but long-duration drift.

Two ffmpeg filters considered
-----------------------------

1. **vidstab** (`libvidstab`): proper motion-vector-based stabilization
   with a detect-then-transform two-pass. Much better quality, but
   requires ffmpeg compiled with `--enable-libvidstab`. Debian's ffmpeg
   package includes it starting from bookworm, so we have it.

2. **deshake** (built-in): single-pass, coarser. Falls back if
   vidstab isn't available. Don't expect the same quality; the
   auto-detection is less robust to zoom/rotation.

This module tries vidstab first and falls back to deshake on failure.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StabilizeError(RuntimeError):
    pass


@dataclass
class StabilizeResult:
    input_path: str
    output_path: str
    backend: str  # "vidstab" | "deshake" | "skipped"
    smoothing: int
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "backend": self.backend,
            "smoothing": self.smoothing,
            "duration_ms": self.duration_ms,
        }


def stabilize_video(
    input_path: Path,
    output_path: Path,
    smoothing: int = 10,
    zoom: int = 0,
    shakiness: int = 5,
) -> StabilizeResult:
    """Produce a stabilized copy of `input_path` at `output_path`.

    `smoothing`: vidstab smoothing parameter. Window is (smoothing*2+1)
      frames. Default 10 means ~21-frame centered window — catches
      handheld jitter without eating head turns.
    `zoom`: extra zoom beyond what optzoom computes. 0 = auto-zoom to
      avoid black borders. Higher = more crop, safer against edge
      artifacts at the cost of framing tightness.
    `shakiness`: vidstab detection sensitivity (1-10). 5 is a balanced
      default. Higher on shakier sources.

    Returns a StabilizeResult. Raises StabilizeError on hard failure.
    """
    import time
    started = time.perf_counter()

    if not input_path.exists():
        raise StabilizeError(f"input video missing: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Two-pass vidstab. Pass 1 writes a transforms file; pass 2 applies
    # the inverse transforms with optzoom to hide border gaps.
    transforms_file = output_path.with_suffix(output_path.suffix + ".trf")
    try:
        _run_vidstab_pass1(input_path, transforms_file, shakiness=shakiness)
        _run_vidstab_pass2(
            input_path, output_path, transforms_file,
            smoothing=smoothing, zoom=zoom,
        )
        backend = "vidstab"
        log.info(
            "stabilized via vidstab: %s -> %s (smoothing=%d shakiness=%d)",
            input_path.name, output_path.name, smoothing, shakiness,
        )
    except StabilizeError as vidstab_err:
        # Fall back to the built-in deshake filter if vidstab is unavailable
        # or the two-pass run fails for any reason. deshake is worse but
        # it's always available in any ffmpeg build.
        log.warning(
            "vidstab failed (%s); falling back to built-in deshake filter",
            vidstab_err,
        )
        try:
            _run_deshake(input_path, output_path)
            backend = "deshake"
        except Exception as deshake_err:
            raise StabilizeError(
                f"both vidstab and deshake failed; vidstab: {vidstab_err}; "
                f"deshake: {deshake_err}",
            ) from deshake_err
    finally:
        # Always clean up the transforms sidecar file.
        if transforms_file.exists():
            transforms_file.unlink()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise StabilizeError(
            f"stabilize ran ({backend}) but produced no output at {output_path}",
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    return StabilizeResult(
        input_path=input_path.name,
        output_path=output_path.name,
        backend=backend,
        smoothing=smoothing,
        duration_ms=duration_ms,
    )


def vidstab_available() -> bool:
    """Check whether this ffmpeg has libvidstab support compiled in.

    Cheapest reliable check: list the filters and look for vidstabdetect.
    """
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return False
    return "vidstabdetect" in (result.stdout or "")


def _run_vidstab_pass1(
    input_path: Path, transforms_file: Path, shakiness: int,
) -> None:
    """Pass 1: analyse motion, write transforms to sidecar file."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-vf",
        f"vidstabdetect=shakiness={shakiness}:accuracy=15:result={transforms_file}",
        # Discard the encoded output from pass 1 — we only want the .trf
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise StabilizeError(
            f"vidstabdetect failed: "
            f"{proc.stderr.decode(errors='replace')[-800:]}"
        )
    if not transforms_file.exists() or transforms_file.stat().st_size == 0:
        raise StabilizeError(
            f"vidstabdetect completed but no transforms file was written "
            f"at {transforms_file}"
        )


def _run_vidstab_pass2(
    input_path: Path,
    output_path: Path,
    transforms_file: Path,
    smoothing: int,
    zoom: int,
) -> None:
    """Pass 2: apply inverse transforms with smoothing + border hiding."""
    # optzoom=1 auto-computes the minimum zoom that keeps the frame
    # fully covered after shifts. If `zoom` is explicitly set, use that
    # as a fixed additional zoom on top.
    vf = (
        f"vidstabtransform=input={transforms_file}"
        f":smoothing={smoothing}"
        f":optzoom={1 if zoom == 0 else 0}"
        f":zoom={zoom}"
        # unsharp with mild defaults helps recover a little crispness
        # lost to the sub-pixel resampling the stabilizer does.
        f",unsharp=5:5:0.8:3:3:0.4"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-vf", vf,
        # Preserve original audio bit-for-bit. Stabilization is video-only;
        # re-encoding audio would cost quality for no benefit.
        "-c:a", "copy",
        # Re-encode video as h264. libx264 crf 18 is visually lossless.
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=1200)
    if proc.returncode != 0:
        raise StabilizeError(
            f"vidstabtransform failed: "
            f"{proc.stderr.decode(errors='replace')[-800:]}"
        )


def _run_deshake(input_path: Path, output_path: Path) -> None:
    """Fallback single-pass stabilization using ffmpeg's built-in deshake.

    Much coarser than vidstab — deshake estimates inter-frame motion
    via a single-pass block search with limited search radius. It
    handles handheld jitter adequately but loses quality against real
    motion. We use this only when vidstab isn't available.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-vf", "deshake",
        "-c:a", "copy",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=1200)
    if proc.returncode != 0:
        raise StabilizeError(
            f"deshake failed: "
            f"{proc.stderr.decode(errors='replace')[-800:]}"
        )
