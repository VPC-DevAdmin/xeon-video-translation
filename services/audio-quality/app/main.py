"""Optional alignment, speaker diarization and accompaniment separation.

Separate environment avoids changing the translation/TTS dependency graph.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="audio-quality")
ROOT = Path(os.getenv("JOB_ARTIFACTS_DIR", "/jobs")).resolve()
DEVICE = os.getenv("DEVICE", "cpu")
LOCK = threading.Lock()
_aligners = {}
_diarizer = None


def path(value):
    result = Path(value).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise HTTPException(400, "artifact path outside job volume")
    return result


class Analyze(BaseModel):
    audio_path: str
    transcript_path: str
    align: bool = False
    diarize: bool = False


@app.get("/health")
def health():
    import importlib.util

    return {
        "status": "ok",
        "alignment": importlib.util.find_spec("whisperx") is not None,
        "separation": importlib.util.find_spec("demucs") is not None,
        "device": DEVICE,
    }


@app.post("/analyze")
def analyze(body: Analyze):
    audio, transcript = path(body.audio_path), path(body.transcript_path)
    if not audio.is_file() or not transcript.is_file():
        raise HTTPException(404, "audio or transcript missing")
    with LOCK:
        import whisperx
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers

        global _diarizer
        result = json.loads(transcript.read_text())
        waveform = whisperx.load_audio(str(audio))
        if body.align:
            language = result["language"]
            if language not in _aligners:
                # Bound resident alignment models when switching languages.
                _aligners.clear()
                _aligners[language] = whisperx.load_align_model(
                    language_code=language, device=DEVICE
                )
            model, metadata = _aligners[language]
            aligned = whisperx.align(
                result["segments"],
                model,
                metadata,
                waveform,
                DEVICE,
                return_char_alignments=False,
            )
            result["segments"] = aligned["segments"]
        if body.diarize:
            if _diarizer is None:
                token = os.getenv("HF_TOKEN")
                token_file = os.getenv("HF_TOKEN_PATH")
                if not token and token_file and Path(token_file).is_file():
                    token = Path(token_file).read_text().strip()
                _diarizer = DiarizationPipeline(token=token, device=DEVICE)
            result = assign_word_speakers(_diarizer(waveform), result)
        result["text"] = " ".join(s["text"] for s in result["segments"])
        # Return data; the owning backend writes its checkpoint atomically.
        return result


class Separate(BaseModel):
    audio_path: str
    output_dir: str


@app.post("/separate")
def separate(body: Separate):
    source, destination = path(body.audio_path), path(body.output_dir)
    if not source.is_file():
        raise HTTPException(404, "source audio missing")
    destination.mkdir(parents=True, exist_ok=True)
    with LOCK, tempfile.TemporaryDirectory(prefix="separation-") as temporary:
        result = subprocess.run(
            [
                "python",
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "--two-stems",
                "vocals",
                "--device",
                DEVICE,
                "-o",
                temporary,
                str(source),
            ],
            capture_output=True,
            timeout=1800,
        )
        if result.returncode:
            raise HTTPException(502, result.stderr.decode(errors="replace")[-1000:])
        folder = Path(temporary) / "htdemucs" / source.stem
        for src, name in (
            ("vocals.wav", "vocals.wav"),
            ("no_vocals.wav", "background.wav"),
        ):
            shutil.copyfile(folder / src, destination / name)
        return {
            "vocals": str(destination / "vocals.wav"),
            "background": str(destination / "background.wav"),
        }
