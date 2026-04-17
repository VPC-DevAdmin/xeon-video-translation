"""Stage 6 — mux new audio onto the video and apply the watermark.

Always runs, regardless of lipsync backend:
- If lipsync ran: video input is `lipsynced.mp4`
- If lipsync was skipped: video input is the original upload, audio input is
  `translated_audio.wav` — the output is the familiar "foreign-film dub" look.

Watermarking is non-negotiable (see docs/ethics.md). It can be disabled only
via `ENABLE_WATERMARK=false` for internal pipeline testing.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings


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


def mux_and_watermark(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    watermark: bool | None = None,
) -> MuxResult:
    """Produce the final MP4 with muxed audio + optional watermark.

    `video_path` should already carry whatever lipsync produced (or be the
    original upload when lipsync is skipped). Audio is replaced wholesale.
    """
    if not video_path.exists():
        raise MuxError(f"video input missing: {video_path}")
    if not audio_path.exists():
        raise MuxError(f"audio input missing: {audio_path}")

    wm = settings.enable_watermark if watermark is None else watermark
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comment = "AI-generated: polyglot-demo open-source video translation"

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]
    if wm:
        cmd.extend(["-vf", _drawtext_filter(settings.watermark_text)])
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    else:
        # No filter → we can stream-copy the video track.
        cmd.extend(["-c:v", "copy"])
    cmd.extend([
        "-c:a", "aac", "-b:a", "128k",
        "-metadata", f"comment={comment}",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

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
