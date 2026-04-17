"""Stage 5 dispatcher tests. Doesn't load Wav2Lip — only verifies the backend
routing and stub error messages.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.pipeline import lipsync

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_unknown_backend_raises(tmp_path: Path) -> None:
    with pytest.raises(lipsync.LipsyncError, match="unknown"):
        lipsync.run(
            backend="nope",
            video_in=tmp_path / "v.mp4",
            audio_in=tmp_path / "a.wav",
            output_path=tmp_path / "o.mp4",
        )


def test_musetalk_stub(tmp_path: Path) -> None:
    with pytest.raises(lipsync.LipsyncError, match="MuseTalk"):
        lipsync.run(
            backend="musetalk",
            video_in=tmp_path / "v.mp4",
            audio_in=tmp_path / "a.wav",
            output_path=tmp_path / "o.mp4",
        )


def test_latentsync_stub(tmp_path: Path) -> None:
    with pytest.raises(lipsync.LipsyncError, match="LatentSync"):
        lipsync.run(
            backend="latentsync",
            video_in=tmp_path / "v.mp4",
            audio_in=tmp_path / "a.wav",
            output_path=tmp_path / "o.mp4",
        )


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg needed to build a test clip")
def test_none_backend_copies_video(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:d=0.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src),
    ], check=True, capture_output=True)

    result = lipsync.run(
        backend="none",
        video_in=src,
        audio_in=tmp_path / "ignored.wav",  # passthrough doesn't read audio
        output_path=dst,
    )
    assert dst.exists()
    assert result.passthrough is True
    assert result.backend == "none"


def test_realtime_factor_ordering() -> None:
    # ETA sanity check: each backend slower than the last.
    assert lipsync.realtime_factor("none") == 0.0
    assert lipsync.realtime_factor("wav2lip") < lipsync.realtime_factor("musetalk")
    assert lipsync.realtime_factor("musetalk") < lipsync.realtime_factor("latentsync")
