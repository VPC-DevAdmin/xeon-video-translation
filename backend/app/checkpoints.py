"""Content-addressed stage outputs and immutable retry inputs."""

import hashlib
import json
from pathlib import Path

ARTIFACTS = {
    "audio": ["audio.wav"],
    "stabilize": ["stabilized.mp4"],
    "transcribe": ["transcript.json"],
    "translate": ["translation.json"],
    "tts": ["translated_audio.wav"],
    "lipsync": ["lipsynced.mp4"],
    "poststabilize": ["lipsynced_stable.mp4"],
    "mux": ["final.mp4"],
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(directory, stage):
    files = {
        name: digest(directory / name) for name in ARTIFACTS[stage] if (directory / name).is_file()
    }
    if files:
        (directory / f"checkpoint-{stage}.json").write_text(json.dumps(files))


def valid(directory, stage):
    try:
        files = json.loads((directory / f"checkpoint-{stage}.json").read_text())
        return bool(files) and all(
            Path(name).name == name
            and (directory / name).is_file()
            and digest(directory / name) == value
            for name, value in files.items()
        )
    except (OSError, ValueError):
        return False
