"""Optional audio-quality service client and deterministic background remix."""

import json
import urllib.request
from .windowed import run_ffmpeg, duration
from ..config import settings


def call(endpoint, body):
    request = urllib.request.Request(
        settings.audio_quality_url.rstrip("/") + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def analyze(audio, transcript, align=False, diarize=False):
    return call(
        "/analyze",
        {
            "audio_path": str(audio),
            "transcript_path": str(transcript),
            "align": align,
            "diarize": diarize,
        },
    )


def separate(source, destination):
    return call("/separate", {"audio_path": str(source), "output_dir": str(destination)})


def remix(voice, background, output, gain=0.35):
    # Pad speech and sidechain so the accompaniment survives after the last word.
    seconds = max(duration(voice), duration(background))
    run_ffmpeg(
        [
            "-i",
            voice,
            "-i",
            background,
            "-filter_complex",
            f"[0:a]apad=whole_dur={seconds},asplit=2[speech][key];[1:a]volume={gain}[bed];[bed][key]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=250[ducked];[speech][ducked]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.95[out]",
            "-map",
            "[out]",
            "-t",
            seconds,
            "-ar",
            "48000",
            output,
        ]
    )
