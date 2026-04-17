"""Stage 1 tests. Generates a synthetic clip with ffmpeg's lavfi sources so the
suite runs without bundled binary fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.pipeline.audio import (
    AudioExtractionError,
    extract_audio,
    probe_duration_seconds,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def _make_test_clip(path: Path, seconds: float = 2.0) -> None:
    """Create a tiny silent video with a 440 Hz tone for testing."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_probe_duration(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _make_test_clip(clip, seconds=2.0)
    duration = probe_duration_seconds(clip)
    assert 1.5 < duration < 2.5


def test_extract_audio(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    out = tmp_path / "audio.wav"
    _make_test_clip(clip, seconds=2.0)
    extract_audio(clip, out)
    assert out.exists()
    assert out.stat().st_size > 1024  # at minimum a real WAV header + samples


def test_extract_audio_missing_input(tmp_path: Path) -> None:
    with pytest.raises(AudioExtractionError):
        extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav")
