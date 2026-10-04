"""Bounded NVDEC reader for the existing host-frame renderer interfaces.

CUDA decodes compressed frames; FFmpeg currently performs FPS selection and
RGB conversion on the host. This removes an intermediate lossy encode in
LatentSync, but is deliberately not described as a zero-copy pipeline.
"""

import json
import os
import select
import subprocess
import tempfile
import time
from fractions import Fraction
from . import required, span


def probe_stream(path):
    info = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(info.stdout)["streams"][0]


def cuda_download_filter(stream):
    formats = {
        "yuv420p": "nv12",
        "yuvj420p": "nv12",
        "nv12": "nv12",
        "yuv420p10le": "p010le",
        "p010le": "p010le",
    }
    pixel = stream.get("pix_fmt")
    if pixel not in formats:
        raise ValueError(f"unsupported NVDEC pixel format: {pixel}")
    # Requires actual CUDA frames; software decoder fallback cannot pass.
    return "hwdownload,format=" + formats[pixel]


def display_rotation(stream):
    """Display rotation in FFmpeg's display-matrix convention (counter-clockwise
    degrees). The legacy `rotate` tag is clockwise, so it is negated; the
    display matrix wins when both are present."""
    rotation = -float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    return rotation


def cuda_filter_input(path):
    """Input flags and software filter prefix for legacy FFmpeg filter graphs.

    The transpose is applied explicitly from the probed rotation, so the
    input's display matrix must not survive into the output: FFmpeg 6 keeps
    it under -noautorotate (verified on the XE7740 image), and players would
    rotate the already-transposed frames again. -display_rotation 0 clears it.
    """
    stream = probe_stream(path)
    filters = [cuda_download_filter(stream)]
    rotation = display_rotation(stream)
    if round(rotation) % 90:
        raise ValueError("non-right-angle display rotation is unsupported")
    angle = round(rotation) % 360
    if angle == 90:
        filters.append("transpose=cclock")
    elif angle == 270:
        filters.append("transpose=clock")
    elif angle == 180:
        filters += ["hflip", "vflip"]
    return [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-noautorotate",
        "-display_rotation",
        "0",
    ], ",".join(filters)


def read_frames(
    path, budget_bytes, *, fps=None, pixel_format="rgb24", hardware=True, timeout=600
):
    """Decode every frame into host RAM, bounded by `budget_bytes` in total."""
    frames = []
    source_fps = None
    for frame in iter_frames(
        path, fps=fps, pixel_format=pixel_format, hardware=hardware, timeout=timeout,
        max_frame_bytes=budget_bytes, _report_fps=lambda value: None,
    ):
        if (len(frames) + 1) * frame.nbytes > budget_bytes:
            raise RuntimeError(
                "video exceeds decoded-frame budget; shorten the renderer window"
            )
        frames.append(frame)
    if not frames:
        raise RuntimeError("video decode produced no frames")
    return frames, fps or float(Fraction(probe_stream(path).get("avg_frame_rate", "25/1")))


def iter_frames(
    path, *, fps=None, pixel_format="rgb24", hardware=True, timeout=600,
    max_frame_bytes=None, _report_fps=None,
):
    """Yield decoded frames one at a time without retaining them.

    Same NVDEC/ffmpeg pipeline as `read_frames`; only one frame is held at a
    time, so a long source can be scanned (face tracking) within a budget
    that is a single frame rather than the whole clip.
    """
    import numpy as np

    if required() and not hardware:
        raise RuntimeError("GPU deployment requires hardware decoding")
    if pixel_format not in {"rgb24", "bgr24"}:
        raise ValueError("only packed RGB/BGR frames are supported")
    stream = probe_stream(path)
    width, height = int(stream["width"]), int(stream["height"])
    rotation = display_rotation(stream)
    if round(rotation) % 90:
        raise ValueError("non-right-angle display rotation is unsupported")
    if round(rotation) % 180:
        width, height = height, width
    frame_size = width * height * 3
    if frame_size <= 0 or (max_frame_bytes is not None and frame_size > max_frame_bytes):
        raise RuntimeError("one decoded frame exceeds the frame budget")
    source_fps = float(Fraction(stream.get("avg_frame_rate", "25/1")))
    if source_fps <= 0:
        raise ValueError("input has no valid frame rate")
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    filters = []
    if hardware:
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        filters += [cuda_download_filter(stream)]
    command += ["-noautorotate", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn"]
    angle = round(rotation) % 360
    if angle == 90:
        filters.append("transpose=cclock")
    elif angle == 270:
        filters.append("transpose=clock")
    elif angle == 180:
        filters += ["hflip", "vflip"]
    if fps:
        filters.append(f"fps={fps}")
    if filters:
        command += ["-vf", ",".join(filters)]
    command += [
        "-vsync",  # FFmpeg 4.4 (LatentSync) and current versions support passthrough.
        "0",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    pending = bytearray()
    produced = 0
    deadline = time.monotonic() + timeout
    with (
        tempfile.TemporaryFile() as errors,
        span("media.decode", decoder="nvdec" if hardware else "software"),
    ):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if (
                    remaining <= 0
                    or not select.select([process.stdout], [], [], max(0, remaining))[0]
                ):
                    raise TimeoutError("video decode timed out")
                chunk = os.read(
                    process.stdout.fileno(), min(frame_size - len(pending), 1024 * 1024)
                )
                if not chunk:
                    break
                pending.extend(chunk)
                if len(pending) == frame_size:
                    # Own each buffer; subsequent reads cannot overwrite it.
                    frame = np.frombuffer(pending, dtype=np.uint8).reshape(height, width, 3)
                    pending = bytearray()
                    produced += 1
                    yield frame
            status = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            if status or pending or not produced:
                errors.seek(0)
                raise RuntimeError(
                    "video decode failed: "
                    + errors.read()[-1000:].decode(errors="replace")
                )
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
