"""Stage 6 — mux new audio onto the video and apply the watermark.

Always runs, regardless of lipsync backend:
- If lipsync ran: video input is `lipsynced.mp4`
- If lipsync was skipped: video input is the original upload, audio input is
  `translated_audio.wav` — the output is the familiar "foreign-film dub" look.

Watermarking is non-negotiable (see docs/ethics.md). It can be disabled only
via `ENABLE_WATERMARK=false` for internal pipeline testing.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger(__name__)


class MuxError(RuntimeError):
    pass


@dataclass
class MuxResult:
    output_path: str
    watermark: bool
    metadata_comment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "watermark": self.watermark,
            "metadata_comment": self.metadata_comment,
        }


def _drawtext_filter(text: str) -> str:
    """Build an ffmpeg drawtext filter with a readable, corner-pinned badge.

    Coordinates:
      x = w - tw - 20   (20px from the right)
      y = h - th - 20   (20px from the bottom)

    Uses the basefont from fontconfig to avoid shipping a font file. If the
    container has no font, ffmpeg falls back; worst case the filter errors and
    we raise with a clear message.
    """
    # Escape colon and backslash for the filter DSL.
    safe = text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
    return (
        f"drawtext=text='{safe}'"
        f":x=w-tw-20:y=h-th-20"
        f":fontcolor=white:fontsize=24"
        f":box=1:boxcolor=black@0.55:boxborderw=8"
    )


def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            timeout=30,
        )
        return float(out.decode().strip())
    except Exception as e:
        log.warning("ffprobe failed on %s: %s", path, e)
        return None


def mux_and_watermark(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    watermark: bool | None = None,
) -> MuxResult:
    """Produce the final MP4 with muxed audio + optional watermark.

    `video_path` should already carry whatever lipsync produced (or be the
    original upload when lipsync is skipped). Audio is replaced wholesale.

    If the audio is longer than the video, the last video frame is frozen
    (via ffmpeg `tpad`) to pad up to the audio length — important because
    XTTS output commonly runs longer than the source clip for languages
    with different syllable density. Previously we truncated with
    `-shortest` and cut speech mid-word.
    """
    if not video_path.exists():
        raise MuxError(f"video input missing: {video_path}")
    if not audio_path.exists():
        raise MuxError(f"audio input missing: {audio_path}")

    wm = settings.enable_watermark if watermark is None else watermark
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comment = "AI-generated: polyglot-demo open-source video translation"

    video_dur = _probe_duration(video_path)
    audio_dur = _probe_duration(audio_path)
    pad_seconds = 0.0
    if video_dur is not None and audio_dur is not None and audio_dur > video_dur:
        pad_seconds = audio_dur - video_dur

    video_filters: list[str] = []
    if pad_seconds > 0.0:
        # Clone the final frame. Without this, libx264 would end the video
        # stream at the original last frame and the remaining audio plays
        # over nothing.
        video_filters.append(
            f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}"
        )
    if wm:
        video_filters.append(_drawtext_filter(settings.watermark_text))

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]
    if video_filters:
        cmd.extend(["-vf", ",".join(video_filters)])
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    else:
        # No filter → we can stream-copy the video track.
        cmd.extend(["-c:v", "copy"])
    cmd.extend([
        "-c:a", "aac", "-b:a", "128k",
        "-metadata", f"comment={comment}",
        "-movflags", "+faststart",
        # No `-shortest` — we want to keep the full audio. Video is already
        # padded via tpad when needed.
        str(output_path),
    ])

    log.info(
        "mux: video=%.2fs audio=%.2fs pad=%.2fs wm=%s",
        video_dur or -1, audio_dur or -1, pad_seconds, wm,
    )

    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise MuxError(
            f"ffmpeg failed (exit {proc.returncode}):\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  stderr: {proc.stderr.decode(errors='replace')[-2000:]}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise MuxError("ffmpeg produced no output")

    return MuxResult(
        output_path=output_path.name,
        watermark=wm,
        metadata_comment=comment,
    )
