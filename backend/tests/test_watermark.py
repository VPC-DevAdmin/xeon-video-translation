"""Stage 6 tests. Generates tiny synthetic video + audio with ffmpeg so the
test suite doesn't need bundled fixtures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.pipeline.watermark import MuxError, mux_and_watermark

HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def _make_clip(path: Path, seconds: float = 1.0) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=160x120:d={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-tune", "stillimage",
        str(path),
    ], check=True, capture_output=True)


def _make_audio(path: Path, seconds: float = 1.0) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        str(path),
    ], check=True, capture_output=True)


def test_mux_with_watermark(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    audio = tmp_path / "in.wav"
    out = tmp_path / "final.mp4"
    _make_clip(video, 1.0)
    _make_audio(audio, 1.0)
    result = mux_and_watermark(video, audio, out, watermark=True)
    assert out.exists()
    assert out.stat().st_size > 1024
    assert result.watermark is True
    assert "AI-generated" in result.metadata_comment


def test_mux_without_watermark(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    audio = tmp_path / "in.wav"
    out = tmp_path / "final.mp4"
    _make_clip(video, 1.0)
    _make_audio(audio, 1.0)
    result = mux_and_watermark(video, audio, out, watermark=False)
    assert out.exists()
    assert result.watermark is False


def test_mux_missing_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(MuxError):
        mux_and_watermark(
            tmp_path / "nope.mp4",
            tmp_path / "also-nope.wav",
            tmp_path / "out.mp4",
        )
