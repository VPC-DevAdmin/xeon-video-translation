"""Bounded overlapping media windows around existing GPU renderers.

ffmpeg streams decoding/encoding; each model request sees at most one window.
Window artifacts are checksummed and can be reused after a failed attempt.
"""

import json
import logging
import math
import subprocess
from pathlib import Path
from ..checkpoints import digest


class RenderCancelled(RuntimeError):
    pass


def _encoder_args(hardware: bool) -> list[str]:
    """Intermediate-window encode. NVENC keeps the 7-8 window cuts and the
    per-window trims off the CPU; CQ 16 is visually lossless for the renderer
    input. libx264 CRF 16 is used only in the explicit CPU deployment."""
    if hardware:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", "16", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]


def encode(args, output):
    """Use the requested encoder; a failure remains an actionable job error."""
    from ..config import settings
    from gpu_runtime import span
    if settings.resolved_device == "cuda" and settings.video_encoder != "h264_nvenc":
        raise RuntimeError("CUDA window encoding requires h264_nvenc")
    args = list(args)
    if settings.resolved_device == "cuda" and "-i" in args:
        from gpu_runtime.media import cuda_filter_input
        flags, prefix = cuda_filter_input(args[args.index("-i")+1])
        args = flags + args
        if "-vf" in args:
            pos = args.index("-vf")+1
            args[pos] = prefix + "," + args[pos]
        else:
            args += ["-vf", prefix]
    with span("window.encode", encoder=settings.video_encoder):
        return run_ffmpeg([*args, *_encoder_args(settings.video_encoder == "h264_nvenc"), output])


def _call_renderer(renderer, source, sound, rendered, offset_frames):
    """Pass the window's first source frame (25 fps grid) to renderers that
    take it, so they can slice a shared face track instead of re-detecting."""
    import inspect

    try:
        parameters = inspect.signature(renderer).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts = "window_offset_frames" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts:
        return renderer(source, sound, rendered, window_offset_frames=offset_frames)
    return renderer(source, sound, rendered)


def run_ffmpeg(args):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)],
        capture_output=True,
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])


def duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def bounded_plan(video, audio, budget_mb, size, overlap, force=False, fps=None):
    """Bound decoded RGB memory, including context and frame rounding.

    Use 80% of the worker limit to allow decoder/container rounding. This
    bounds retained source frames, not total model/process memory.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    stream = json.loads(result.stdout)["streams"][0]
    from fractions import Fraction

    try:
        source_fps = fps or float(Fraction(stream.get("avg_frame_rate", "25/1"))) or 25
    except (ValueError, ZeroDivisionError):
        source_fps = fps or 25
    frame_bytes = int(stream["width"]) * int(stream["height"]) * 3
    max_frames = int(budget_mb * 1024**2 * 0.8 // frame_bytes)
    if max_frames < 1:
        raise ValueError("one decoded frame exceeds the configured renderer budget")
    seconds = max(duration(video), duration(audio))
    if not force and math.ceil(seconds * source_fps) <= max_frames:
        return None
    # Windows are resampled to 25 fps before the renderer sees them.
    pad = min(round(overlap * 25), max(0, (max_frames - 1) // 4))
    core = min(max(1, round(size * 25)), max_frames - 2 * pad)
    return core / 25, pad / 25


def windows(seconds, size, overlap):
    total = math.ceil(seconds * 25)
    step = max(1, round(size * 25))
    pad = max(0, round(overlap * 25))
    for start in range(0, total, step):
        end = min(total, start + step)
        yield start, end, max(0, start - pad), min(total, end + pad)


def render(
    video,
    audio,
    output,
    renderer,
    size=8,
    overlap=0.4,
    progress=None,
    cancel=None,
    configuration=None,
    preparer=None,
):
    """Render `video` in overlapping windows. While window N renders, window
    N+1 is cut (ffmpeg) and, when `preparer(source, sound, offset_frames)` is
    given, handed to the renderer to decode and warp ahead of its turn."""
    import threading

    video, audio, output = map(Path, (video, audio, output))
    root = output.parent / "render-windows"
    root.mkdir(exist_ok=True)
    video_seconds = duration(video)
    audio_seconds = duration(audio)
    seconds = max(video_seconds, audio_seconds)
    fingerprint = {
        "video": digest(video),
        "audio": digest(audio),
        "size": size,
        "overlap": overlap,
        "configuration": configuration or {},
    }
    manifest = root / "manifest.json"
    try:
        previous = json.loads(manifest.read_text())
    except (OSError, ValueError):
        previous = {}
    if previous.get("fingerprint") != fingerprint:
        previous = {"fingerprint": fingerprint, "parts": {}}
    parts = list(windows(seconds, size, overlap))
    outputs = []

    def needs_render(index):
        clip = root / f"part-{index:06}.mp4"
        return not (clip.exists() and previous["parts"].get(str(index)) == digest(clip))

    def cut(index):
        """Write the 25 fps source window and its audio for `index`."""
        start, end, left, right = parts[index]
        source = root / f"source-{index:06}.mp4"
        sound = root / f"audio-{index:06}.wav"
        # Seek near each window rather than decoding the whole prefix again.
        # Beyond EOF, retain one last frame/sample before padding.
        span = (right - left) / 25
        seek = min(left / 25, max(0, video_seconds - 0.08))
        encode(
            [
                "-ss",
                seek,
                "-i",
                video,
                "-an",
                "-vf",
                f"fps=25,tpad=stop_mode=clone:stop_duration={span},trim=end_frame={right - left},setpts=PTS-STARTPTS",
                "-r",
                "25",
                "-fps_mode",
                "cfr",
            ],
            source,
        )
        if left / 25 >= audio_seconds:
            run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", span, sound])
        else:
            run_ffmpeg(
                [
                    "-ss",
                    left / 25,
                    "-i",
                    audio,
                    "-af",
                    f"apad=whole_dur={span},atrim=end={span},asetpts=PTS-STARTPTS",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    sound,
                ]
            )
        return source, sound

    def cut_and_prepare(index):
        try:
            source, sound = cut(index)
            if preparer is not None:
                preparer(source, sound, parts[index][2])
        except Exception as exc:  # the main loop will cut again and the renderer prepares itself
            logging.getLogger(__name__).warning("window %d prefetch failed: %s", index, str(exc)[-200:])

    prefetch = {}
    for index, (start, end, left, right) in enumerate(parts):
        if cancel and cancel.is_set():
            raise RenderCancelled("render cancelled between windows")
        clip = root / f"part-{index:06}.mp4"
        if needs_render(index):
            source = root / f"source-{index:06}.mp4"
            sound = root / f"audio-{index:06}.wav"
            rendered = root / f"rendered-{index:06}.mp4"
            thread = prefetch.pop(index, None)
            if thread is not None:
                thread.join()
            if not (source.exists() and sound.exists()):
                cut(index)
            # Overlap: cut and prepare the next window while this one renders.
            nxt = index + 1
            if nxt < len(parts) and needs_render(nxt) and not (cancel and cancel.is_set()):
                thread = threading.Thread(target=cut_and_prepare, args=(nxt,), name=f"window-prefetch-{nxt}", daemon=True)
                thread.start()
                prefetch[nxt] = thread
            _call_renderer(renderer, source, sound, rendered, left)
            if cancel and cancel.is_set():
                raise RenderCancelled("render cancelled after active window drained")
            encode(
                [
                    "-i",
                    rendered,
                    "-an",
                    "-vf",
                    f"fps=25,trim=start_frame={start - left}:end_frame={end - left},setpts=PTS-STARTPTS",
                    "-r",
                    "25",
                    "-fps_mode",
                    "cfr",
                ],
                clip,
            )
            if abs(duration(clip) - (end - start) / 25) > 0.08:
                raise RuntimeError("renderer returned the wrong window duration")
            previous["parts"][str(index)] = digest(clip)
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(previous))
            temporary.replace(manifest)
            for scratch in (source, sound, rendered):
                scratch.unlink(missing_ok=True)
        outputs.append(clip)
        if progress:
            progress((index + 1) / len(parts))
    listing = root / "concat.txt"
    listing.write_text("\n".join(f"file '{p.name}'" for p in outputs))
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            listing,
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            output,
        ]
    )
    for thread in prefetch.values():
        thread.join()
    for scratch in root.glob("source-*.mp4"):
        scratch.unlink(missing_ok=True)
    for scratch in list(root.glob("audio-*.wav")) + list(root.glob("rendered-*.mp4")):
        scratch.unlink(missing_ok=True)
    return {"windows": len(parts), "duration": seconds, "configuration": fingerprint}
