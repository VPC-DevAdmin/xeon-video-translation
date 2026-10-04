import shutil
import subprocess
import numpy as np
import pytest
from gpu_runtime.media import read_frames


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    path = tmp_path / "clip with spaces.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x48:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_fps_color_and_owned_frames(clip, monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("GPU_REQUIRED", raising=False)
    frames, fps = read_frames(clip, 1024 * 1024, fps=25, hardware=False)
    assert len(frames) == 25 and fps == 25
    assert frames[0].shape == (48, 64, 3)
    assert frames[0][0, 0, 0] > 240 and frames[0][0, 0, 1] < 10
    frames[0][0, 0] = 0
    assert frames[1][0, 0, 0] > 240
    bgr, _ = read_frames(clip, 1024 * 1024, pixel_format="bgr24", hardware=False)
    assert len(bgr) == 30 and bgr[0][0, 0, 2] > 240


def test_budget_reaps_decoder(clip, monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("GPU_REQUIRED", raising=False)
    with pytest.raises(RuntimeError, match="exceeds decoded-frame budget"):
        read_frames(clip, 64 * 48 * 3 * 2, hardware=False)
    with pytest.raises(RuntimeError, match="one decoded frame"):
        read_frames(clip, 1, hardware=False)


def test_gpu_mode_refuses_software_decode(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="requires hardware"):
        read_frames("unused", 1024, hardware=False)


def test_rotation_matches_ffmpeg_display_orientation(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("GPU_REQUIRED", raising=False)
    source = tmp_path / "source.mp4"
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=1:d=1",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    supported = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True
    ).stdout
    command = ["ffmpeg", "-v", "error", "-y"]
    if "-display_rotation" in supported:
        command += ["-display_rotation", "90", "-i", str(source), "-c", "copy"]
    else:
        command += ["-i", str(source), "-c", "copy", "-metadata:s:v:0", "rotate=90"]
    subprocess.run(command + [str(rotated)], check=True)
    expected = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(rotated),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    frames, _ = read_frames(rotated, 1024 * 1024, hardware=False)
    assert frames[0].shape == (64, 48, 3)
    np.testing.assert_array_equal(
        frames[0], np.frombuffer(expected, np.uint8).reshape(64, 48, 3)
    )
