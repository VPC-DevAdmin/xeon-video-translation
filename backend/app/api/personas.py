"""Personas: a person's face and voice captured once, reused by the video assistant.

The capture is standardized so a user can click through it: a webcam portrait
framed by a guide, an optional eight-second idle clip (future footage-based
rendering), and a ~25 s reading of a fixed script. The server checks the
portrait (face found, framed, sharp, lit) and the voice (length, level, clipping,
the words actually read), then builds the XTTS cloning conditioning and keeps
everything under JOB_ARTIFACTS_DIR/personas/<id>/ with a consent record.

Endpoints
  GET  /personas/script?language=en      the reading script and the capture rules
  POST /personas                         multipart: name, language, consent, script, portrait, voice[, idle]
  GET  /personas                         the caller's personas
  GET  /personas/{id}                    record (paths are for services sharing /jobs)
  GET  /personas/{id}/portrait           the portrait image
  POST /personas/{id}/preview            a short sentence in the cloned voice (WAV)
  DELETE /personas/{id}
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..pipeline.orchestrator import blocking_call, speech_lock
from ..security import check_owner, principal

router = APIRouter(prefix="/personas", tags=["personas"])
ROOT = (settings.job_artifacts_dir / "personas").resolve()
CONSENT_VERSION = "2026-10-04"
CONSENT_TEXT = ("I am the person being recorded, or I have their permission. I agree that this portrait "
                "and voice sample may be used to generate speech and video of this person for this demo, "
                "and that generated media will be marked as AI-generated.")
RULES = {"portrait": "Face the camera in even light, neutral expression, face inside the oval, no glasses glare.",
         "idle_seconds": 8, "voice_min_seconds": 15, "voice_target_seconds": 25, "voice_max_seconds": 45}
SCRIPTS = {
    "en": ("Hi, I'm recording a short sample so the assistant can use my voice. The quick brown fox jumps over "
           "the lazy dog, but today it is far too hot to jump anywhere. Please read at the pace you would use "
           "with a friend, and let your voice rise and fall naturally. Numbers help too: three, seventeen, "
           "forty-two, nineteen ninety-nine. That's it, thank you."),
    "es": ("Hola, estoy grabando una muestra corta para que el asistente pueda usar mi voz. El veloz murciélago "
           "hindú comía feliz cardillo y kiwi, pero hoy hace demasiado calor para volar. Lee con naturalidad, "
           "como hablarías con un amigo. Los números también ayudan: tres, diecisiete, cuarenta y dos. Gracias."),
    "fr": ("Bonjour, j'enregistre un court échantillon pour que l'assistant puisse utiliser ma voix. Portez ce "
           "vieux whisky au juge blond qui fume, mais aujourd'hui il fait bien trop chaud. Lisez naturellement, "
           "comme avec un ami. Les nombres aident aussi : trois, dix-sept, quarante-deux. Merci."),
    "de": ("Hallo, ich nehme eine kurze Probe auf, damit der Assistent meine Stimme verwenden kann. Franz jagt "
           "im komplett verwahrlosten Taxi quer durch Bayern, aber heute ist es viel zu heiß dafür. Lies "
           "natürlich, wie mit einem Freund. Zahlen helfen auch: drei, siebzehn, zweiundvierzig. Danke."),
}
PREVIEW = {"en": "Hello! This is how I will sound as your assistant. Ask me anything when you're ready.",
           "es": "¡Hola! Así sonaré como tu asistente. Pregúntame lo que quieras cuando estés listo.",
           "fr": "Bonjour ! Voilà comment je parlerai comme votre assistant. Posez-moi vos questions.",
           "de": "Hallo! So werde ich als dein Assistent klingen. Frag mich, wann immer du bereit bist."}


def script_for(language: str) -> str:
    return SCRIPTS.get(language, SCRIPTS["en"])


# ------------------------------------------------------------------ checks
def normalize_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def script_match(script: str, heard: str) -> float:
    """Fraction of the script's words recognised in order (0..1)."""
    a, b = normalize_words(script), normalize_words(heard)
    if not a:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return round(sum(block.size for block in matcher.get_matching_blocks()) / len(a), 3)


def voice_level_checks(samples, sample_rate: int) -> dict:
    """Duration, loudness and clipping of a float32 mono signal in [-1, 1]."""
    import numpy as np

    duration = len(samples) / sample_rate
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
    dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    clipped = float(np.mean(np.abs(samples) >= 0.985)) if len(samples) else 0.0
    # speech presence: share of 50 ms blocks above -45 dBFS
    block = max(1, sample_rate // 20)
    blocks = samples[: len(samples) // block * block].reshape(-1, block) if len(samples) >= block else samples[None, :]
    active = float(np.mean(np.sqrt(np.mean(blocks ** 2, axis=1)) > 10 ** (-45 / 20))) if len(samples) else 0.0
    checks = {"duration_seconds": round(duration, 2), "level_dbfs": round(float(dbfs), 1),
              "clipped_fraction": round(clipped, 4), "active_fraction": round(active, 3)}
    problems = []
    if duration < RULES["voice_min_seconds"]:
        problems.append(f"recording is {duration:.0f} s; read the whole script (at least {RULES['voice_min_seconds']} s)")
    if dbfs < -38:
        problems.append("recording is too quiet; move closer to the microphone")
    if dbfs > -6:
        problems.append("recording is too loud")
    if clipped > 0.005:
        problems.append("recording clips; lower the input level")
    if active < 0.35 and duration >= 5:
        problems.append("mostly silence; make sure the microphone picked you up")
    checks["problems"] = problems
    checks["ok"] = not problems
    return checks


def portrait_checks(image_path: Path) -> dict:
    """Face found, framed and sharp, using OpenCV's bundled frontal-face cascade."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return {"ok": False, "problems": ["could not decode the portrait"]}
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(40, w // 10), max(40, h // 10)))
    problems = []
    raw = [tuple(int(v) for v in f) for f in faces]
    if raw:
        # The cascade fires on lamps and window frames and often boxes the same face
        # twice at different scales: keep detections comparable in size to the main
        # face, and merge any that overlap it.
        main = max(raw, key=lambda f: f[2] * f[3])
        largest = main[2] * main[3]

        def overlaps(a, b):
            ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
            inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
            return inter / max(1, min(a[2] * a[3], b[2] * b[3])) > 0.3

        faces = [f for f in raw if f[2] * f[3] >= 0.33 * largest and (f == main or not overlaps(f, main))]
    result = {"width": w, "height": h, "faces": int(len(faces)), "raw_detections": raw}
    if len(faces) == 0:
        problems.append("no face found; face the camera in even light")
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = (x + fw / 2) / w, (y + fh / 2) / h
        face = gray[y: y + fh, x: x + fw]
        sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())
        brightness = float(face.mean())
        result.update({"face_box": [int(x), int(y), int(fw), int(fh)], "face_height_ratio": round(fh / h, 3),
                       "center": [round(cx, 3), round(cy, 3)], "sharpness": round(sharpness, 1),
                       "brightness": round(brightness, 1)})
        if fh / h < 0.22:
            problems.append("face is too small; move closer")
        if fh / h > 0.75:
            problems.append("face is too close; move back so the whole head fits")
        if abs(cx - 0.5) > 0.2 or abs(cy - 0.5) > 0.22:
            problems.append("face is off-centre; keep it inside the oval")
        if sharpness < 40:
            problems.append("portrait is blurry; hold still and check focus")
        if brightness < 55:
            problems.append("too dark; add light on your face")
        if brightness > 215:
            problems.append("too bright; reduce the light")
        if len(faces) > 1:
            problems.append("more than one face in frame")
        warnings = []
        border = np.concatenate([gray[: h // 8].ravel(), gray[-h // 8:].ravel(), gray[:, : w // 8].ravel(), gray[:, -w // 8:].ravel()])
        surround = float(border.mean())
        result["surround_brightness"] = round(surround, 1)
        if surround > brightness + 45:
            warnings.append("you are backlit (window or lamp behind you); face the light so your face is brighter than the background")
        if fh < 180:
            warnings.append(f"face is only {int(fh)} px tall in the capture; move closer or use a higher-resolution camera")
        result["warnings"] = warnings
    result["problems"] = problems
    result["ok"] = not problems
    return result


def render_portrait(image_path: Path, out_path: Path, face_box=None) -> dict | None:
    """Square head-and-shoulders crop centred on the face for the renderer.

    FlashHead regenerates the whole 512 px canvas, so a face that fills about half of
    it gets several times the pixels of a face in a wide webcam frame."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if face_box is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(40, w // 10), max(40, h // 10)))
        if len(faces) == 0:
            return None
        face_box = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [int(v) for v in face_box]
    side = int(min(max(fh * 2.1, fw * 2.1), min(w, h)))
    cx, cy = x + fw / 2, y + fh / 2 + fh * 0.12          # a little below the face centre: room for shoulders
    x0 = int(min(max(0, cx - side / 2), w - side))
    y0 = int(min(max(0, cy - side / 2), h - side))
    crop = image[y0: y0 + side, x0: x0 + side]
    crop = cv2.resize(crop, (768, 768), interpolation=cv2.INTER_AREA if side > 768 else cv2.INTER_CUBIC)
    cv2.imwrite(str(out_path), crop)
    return {"crop": [x0, y0, side, side], "face_height_ratio": round(fh / side, 3), "source_face_px": fh}


# ------------------------------------------------------------------ building
def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip()[-400:] or "ffmpeg failed")


def _build(directory: Path, language: str, script: str) -> dict:
    import numpy as np
    import soundfile as sf
    from ..pipeline import transcribe, tts

    _ffmpeg("-i", str(directory / "voice.upload"), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(directory / "voice.wav"))
    _ffmpeg("-i", str(directory / "voice.wav"), "-ar", "16000", "-c:a", "pcm_s16le", str(directory / "voice16.wav"))
    samples, rate = sf.read(str(directory / "voice.wav"), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    voice = voice_level_checks(samples, rate)
    heard = ""
    transcript = None
    if voice["duration_seconds"] >= 3:
        transcript = transcribe.transcribe(directory / "voice16.wav", directory / "voice16.json", language)
        heard = transcript.text.strip()
        voice["heard"] = heard
        voice["script_match"] = script_match(script, heard)
        if voice["script_match"] < 0.6:
            voice["problems"].append("the words did not match the script well enough; read the script shown")
            voice["ok"] = False
    portrait = portrait_checks(directory / "portrait.png")
    if portrait.get("face_box"):
        portrait["render_crop"] = render_portrait(directory / "portrait.png", directory / "portrait_render.png", portrait["face_box"])
    idle = None
    if (directory / "idle.upload").exists():
        encoder = os.environ.get("VIDEO_ENCODER", "libx264")
        try:
            _ffmpeg("-i", str(directory / "idle.upload"), "-an", "-vf", "fps=25", "-c:v", encoder, "-pix_fmt", "yuv420p", str(directory / "idle.mp4"))
        except RuntimeError:
            _ffmpeg("-i", str(directory / "idle.upload"), "-an", "-vf", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(directory / "idle.mp4"))
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                "stream=nb_read_frames,width,height", "-of", "json", str(directory / "idle.mp4")],
                               capture_output=True, text=True)
        try:
            stream = json.loads(probe.stdout)["streams"][0]
            idle = {"frames": int(stream.get("nb_read_frames", 0)), "width": stream.get("width"), "height": stream.get("height")}
        except (ValueError, KeyError, IndexError):
            idle = {"frames": None}
    checks = {"voice": voice, "portrait": portrait, "idle": idle}
    if voice["ok"] and portrait["ok"]:
        reference = reference_span(directory, transcript)
        voice["reference"] = reference
        build_conditioning(directory)
    return checks


def reference_span(directory: Path, transcript, max_seconds: float = 25.0, pad: float = 0.2) -> dict:
    """Cut `voice_ref.wav`: the speech-dense span of the recording the clone is built
    from. Leading and trailing silence, breaths and room noise before the first word
    make XTTS clones babble; a tight, words-only reference is what it was trained on."""
    import soundfile as sf

    audio, rate = sf.read(str(directory / "voice.wav"), dtype="float32", always_2d=False)
    total = len(audio) / rate
    words = [w for seg in (transcript.segments if transcript else []) for w in seg.words if w.start is not None and w.end is not None]
    if not words:
        start, end = 0.0, min(total, max_seconds)
    else:
        start, end = max(0.0, float(words[0].start) - pad), min(total, float(words[-1].end) + pad)
        if end - start > max_seconds:
            # the window of max_seconds with the most words
            best, best_count = start, 0
            for w in words:
                left = float(w.start) - pad
                count = sum(1 for x in words if left <= float(x.start) and float(x.end) <= left + max_seconds)
                if count > best_count:
                    best, best_count = left, count
            start, end = max(0.0, best), min(total, best + max_seconds)
    sf.write(str(directory / "voice_ref.wav"), audio[int(start * rate): int(end * rate)], rate)
    return {"start": round(start, 2), "end": round(end, 2), "seconds": round(end - start, 2), "words": len(words)}


def build_conditioning(directory: Path) -> None:
    import torch
    from ..pipeline import tts
    reference = directory / "voice_ref.wav"
    if not reference.exists():
        reference = directory / "voice.wav"
    model = tts._get_xtts().synthesizer.tts_model
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(reference)], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=30)
    torch.save({"gpt_cond_latent": gpt_cond_latent.detach().cpu(), "speaker_embedding": speaker_embedding.detach().cpu()},
               directory / "voice.pt")


@lru_cache(maxsize=8)
def _cached_conditioning(persona_id: str, mtime: float):
    import torch
    state = torch.load(ROOT / persona_id / "voice.pt", map_location="cpu", weights_only=True)
    return state


def conditioning(persona_id: str, owner_id: str | None = None) -> dict:
    """XTTS conditioning tensors on the TTS device for a persona (used by the assistant turns)."""
    from ..pipeline import tts
    record = _load(persona_id)
    if owner_id is not None and record.get("owner_id", "local") != owner_id:
        raise HTTPException(404, "persona not found")
    directory = ROOT / persona_id
    path = directory / "voice.pt"
    if not (directory / "voice_ref.wav").exists() and (directory / "voice.wav").exists():
        # Persona built before reference trimming: cut the reference and redo the latents once.
        from ..pipeline import transcribe
        transcript = transcribe.transcribe(directory / "voice16.wav", directory / "voice16.json", record.get("language", "en")) \
            if (directory / "voice16.wav").exists() else None
        record.setdefault("checks", {}).setdefault("voice", {})["reference"] = reference_span(directory, transcript)
        build_conditioning(directory)
        (directory / "persona.json").write_text(json.dumps(record, indent=1))
        _cached_conditioning.cache_clear()
    if not path.exists():
        raise HTTPException(409, "persona has no usable voice")
    state = _cached_conditioning(persona_id, path.stat().st_mtime)
    device = next(tts._get_xtts().synthesizer.tts_model.parameters()).device
    return {k: v.to(device) for k, v in state.items()}


def _load(persona_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", persona_id or ""):
        raise HTTPException(404, "persona not found")
    path = ROOT / persona_id / "persona.json"
    if not path.exists():
        raise HTTPException(404, "persona not found")
    return json.loads(path.read_text())


def _summary(record: dict) -> dict:
    checks = record.get("checks", {})
    return {"id": record["id"], "name": record["name"], "language": record["language"], "created_at": record["created_at"],
            "voice_seconds": (checks.get("voice") or {}).get("duration_seconds"),
            "script_match": (checks.get("voice") or {}).get("script_match"),
            "has_idle_clip": bool(checks.get("idle")), "status": record.get("status")}


# ---------------------------------------------------------------- endpoints
@router.get("/script")
def script(language: str = "en") -> dict:
    return {"language": language if language in SCRIPTS else "en", "text": script_for(language), "rules": RULES,
            "consent": {"version": CONSENT_VERSION, "text": CONSENT_TEXT}}


@router.post("")
async def create(name: str = Form(..., min_length=1, max_length=80), language: str = Form("en"),
                 consent: str = Form(...), script: str = Form(..., max_length=2000),
                 portrait: UploadFile = File(...), voice: UploadFile = File(...), idle: UploadFile | None = File(None)):
    if consent.lower() not in ("yes", "true", "1", "on"):
        raise HTTPException(400, "consent is required")
    from PIL import Image, ImageOps
    from io import BytesIO

    portrait_bytes = await portrait.read(12 * 1024 * 1024 + 1)
    if len(portrait_bytes) > 12 * 1024 * 1024:
        raise HTTPException(413, "portrait exceeds 12 MB")
    try:
        picture = ImageOps.exif_transpose(Image.open(BytesIO(portrait_bytes))).convert("RGB")
        picture.thumbnail((1024, 1024))
    except Exception as exc:
        raise HTTPException(422, "portrait is not a valid image") from exc
    voice_bytes = await voice.read(40 * 1024 * 1024 + 1)
    if len(voice_bytes) > 40 * 1024 * 1024:
        raise HTTPException(413, "voice recording exceeds 40 MB")
    idle_bytes = await idle.read(80 * 1024 * 1024 + 1) if idle is not None else None
    if idle_bytes is not None and len(idle_bytes) > 80 * 1024 * 1024:
        raise HTTPException(413, "idle clip exceeds 80 MB")

    persona_id = uuid.uuid4().hex
    directory = ROOT / persona_id
    directory.mkdir(parents=True)
    picture.save(directory / "portrait.png")
    (directory / "voice.upload").write_bytes(voice_bytes)
    if idle_bytes:
        (directory / "idle.upload").write_bytes(idle_bytes)
    owner_id = principal.get()
    try:
        async with speech_lock(0):
            checks = await blocking_call(lambda: _build(directory, language, script))
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(500, f"persona build failed: {type(exc).__name__}: {exc}")
    ok = checks["voice"]["ok"] and checks["portrait"]["ok"]
    record = {"id": persona_id, "name": name, "language": language, "owner_id": owner_id,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "consent": {"version": CONSENT_VERSION, "text": CONSENT_TEXT, "accepted_at": time.time()},
              "script": script, "checks": checks, "status": "ready" if ok else "rejected",
              "files": {"portrait": str(directory / "portrait.png"),
                        "portrait_render": str(directory / "portrait_render.png") if (directory / "portrait_render.png").exists() else None,
                        "voice": str(directory / "voice.wav"),
                        "idle": str(directory / "idle.mp4") if (directory / "idle.mp4").exists() else None,
                        "conditioning": str(directory / "voice.pt") if (directory / "voice.pt").exists() else None}}
    for name_ in ("voice.upload", "idle.upload"):
        (directory / name_).unlink(missing_ok=True)
    if not ok:
        rejected = ROOT / "_rejected"
        rejected.mkdir(parents=True, exist_ok=True)
        shutil.copy(directory / "portrait.png", rejected / f"{persona_id}.png")
        (rejected / f"{persona_id}.json").write_text(json.dumps(checks, indent=1))
        for old_file in sorted(rejected.glob("*.png"), key=lambda q: q.stat().st_mtime)[:-5]:
            old_file.unlink(missing_ok=True); old_file.with_suffix(".json").unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(422, {"status": "rejected", "checks": checks})
    (directory / "persona.json").write_text(json.dumps(record, indent=1))
    return {"persona": _summary(record), "checks": checks}


@router.get("")
def list_personas() -> dict:
    owner_id = principal.get()
    items = []
    if ROOT.exists():
        for path in sorted(ROOT.glob("*/persona.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                record = json.loads(path.read_text())
            except ValueError:
                continue
            if record.get("owner_id", "local") == owner_id:
                items.append(_summary(record))
    return {"personas": items}


@router.get("/{persona_id}")
def get(persona_id: str) -> dict:
    record = _load(persona_id)
    check_owner(record)
    render = ROOT / persona_id / "portrait_render.png"
    if not render.exists() and Path(record["files"]["portrait"]).exists():
        info = render_portrait(Path(record["files"]["portrait"]), render, (record.get("checks", {}).get("portrait") or {}).get("face_box"))
        if info:
            record["files"]["portrait_render"] = str(render)
            record.setdefault("checks", {}).setdefault("portrait", {})["render_crop"] = info
            (ROOT / persona_id / "persona.json").write_text(json.dumps(record, indent=1))
    return record


@router.get("/{persona_id}/portrait")
def portrait_image(persona_id: str):
    record = _load(persona_id)
    check_owner(record)
    return FileResponse(record["files"]["portrait"], media_type="image/png")


class PreviewRequest(BaseModel):
    text: str | None = Field(None, max_length=300)


@router.post("/{persona_id}/preview")
async def preview(persona_id: str, body: PreviewRequest | None = None):
    record = _load(persona_id)
    check_owner(record)
    text = (body.text if body and body.text else None) or PREVIEW.get(record["language"], PREVIEW["en"])
    out = ROOT / persona_id / "preview.wav"

    def synthesize():
        import numpy as np
        import soundfile as sf
        import torch
        from ..pipeline import tts
        model = tts._get_xtts().synthesizer.tts_model
        cond = conditioning(persona_id)
        with torch.inference_mode():
            result = model.inference(text, tts.XTTS_LANG_CODES.get(record["language"], "en"),
                                     cond["gpt_cond_latent"], cond["speaker_embedding"])
        wav = np.asarray(result["wav"], dtype=np.float32)
        sf.write(str(out), wav, 24000)

    async with speech_lock(0):
        await blocking_call(synthesize)
    return FileResponse(str(out), media_type="audio/wav")


@router.delete("/{persona_id}")
def delete(persona_id: str) -> dict:
    record = _load(persona_id)
    check_owner(record)
    shutil.rmtree(ROOT / persona_id, ignore_errors=True)
    _cached_conditioning.cache_clear()
    return {"status": "deleted"}
