"""Stage 1 — audio extraction via ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioExtractionError(RuntimeError):
    pass


def probe_duration_seconds(input_path: Path) -> float:
    """Return media duration in seconds via ffprobe. Raises if probe fails."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(input_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=30)
    except subprocess.CalledProcessError as e:
        raise AudioExtractionError(
            f"ffprobe failed: {e.stderr.decode(errors='replace')}"
        ) from e
    data = json.loads(out)
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as e:
        raise AudioExtractionError("ffprobe returned no duration") from e


def extract_audio(input_path: Path, output_path: Path, sample_rate: int = 16_000) -> Path:
    """Extract a mono PCM WAV at the given sample rate. Whisper expects 16 kHz."""
    if not input_path.exists():
        raise AudioExtractionError(f"input does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        "-f", "wav",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace')[-2000:]}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioExtractionError("ffmpeg produced no output")
    return output_path
