"""Plan-then-speak assistant turns.

`/assistant/respond` transcribes the utterance, generates the whole reply text
first (the renderer's head start makes that affordable), then streams XTTS
audio sentence by sentence as PCM16 24 kHz in NDJSON events. No audio files
cross between services. `/assistant/speak` synthesizes a given text the same
way (used for the prepared acknowledgement)."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from .. import llm
from ..pipeline.orchestrator import blocking_call, speech_lock
from ..pipeline import transcribe, tts

router = APIRouter(prefix="/assistant", tags=["assistant"])
log = logging.getLogger(__name__)
SAMPLE_RATE = 24000
MAX_TOKENS = int(os.getenv("ASSISTANT_MAX_TOKENS", "220"))
CHUNK_SAMPLES = int(os.getenv("ASSISTANT_TTS_CHUNK_SAMPLES", "12000"))   # 0.5 s per audio event


class Turn(BaseModel):
    audio_path: str
    language: str = "en"
    voice: str | None = Field(None, max_length=100)
    persona_id: str | None = Field(None, pattern=r"^[0-9a-f]{32}$", description="cloned voice from /personas")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


class Speak(BaseModel):
    text: str = Field(..., min_length=1, max_length=800)
    language: str = "en"
    voice: str | None = Field(None, max_length=100)
    persona_id: str | None = Field(None, pattern=r"^[0-9a-f]{32}$")
    verify: bool = Field(True, description="check each sentence with the recognizer and re-synthesize a bad take")


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    """Sentence units for TTS; long run-ons are cut at clause punctuation."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        while len(part) > max_len:
            cut = max(part.rfind(c, 0, max_len) for c in ",;:，；")
            if cut < 40:
                cut = max_len
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    return out


def pcm_event(audio, sentence_id: int, final: bool, text: str) -> dict:
    import numpy as np
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16).tobytes()
    return {"type": "audio", "pcm_b64": base64.b64encode(pcm).decode(), "sample_rate": SAMPLE_RATE,
            "samples": int(len(clipped)), "sentence_id": sentence_id, "final": final, "text": text}


def _speaker(voice, persona_id=None):
    model = tts._get_xtts().synthesizer.tts_model
    if persona_id:
        from . import personas
        return model, personas.conditioning(persona_id)
    speakers = model.speaker_manager.speakers
    name = voice or os.getenv("AVATAR_SPEAKER") or next(iter(speakers))
    if name not in speakers:
        raise RuntimeError("voice is not a bundled XTTS speaker")
    return model, speakers[name]


def _synthesize(model, conditioning, text: str, language: str):
    """Yield float32 24 kHz arrays of about CHUNK_SAMPLES as XTTS streams them."""
    import numpy as np
    import torch

    stream = model.inference_stream(text, tts.XTTS_LANG_CODES[language], conditioning["gpt_cond_latent"],
                                    conditioning["speaker_embedding"], stream_chunk_size=20)
    pending, length = [], 0
    while True:
        with torch.inference_mode():
            chunk = next(stream, None)
            if chunk is not None:
                chunk = chunk.detach().float().cpu().numpy().reshape(-1)
        if chunk is not None:
            pending.append(chunk)
            length += len(chunk)
        if length >= CHUNK_SAMPLES or (chunk is None and length):
            yield np.concatenate(pending)
            pending, length = [], 0
        if chunk is None:
            return


def _verified_sentence(model, conditioning, sentence: str, language: str, attempts: int = 3):
    """Synthesize one sentence and keep only a take whose recognized words are exactly
    the sentence, trimmed to those words (the generation pipeline's rule). Cloned
    voices sometimes babble, repeat or trail off; such takes are re-synthesized.
    Falls back to the shortest fuzzy-matching take when every attempt fails."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    from .personas import script_match
    from ..pipeline.tts import _trim_tail_via_whisper, _trim_to_speech

    fallback = None
    for attempt in range(attempts):
        with torch.inference_mode():
            result = model.inference(sentence, tts.XTTS_LANG_CODES[language], conditioning["gpt_cond_latent"],
                                     conditioning["speaker_embedding"], temperature=max(0.45, 0.7 - 0.1 * attempt),
                                     repetition_penalty=10.0)
        audio = np.asarray(result["wav"], dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="speak-") as directory:
            wav = Path(directory) / "take.wav"
            sf.write(str(wav), audio, SAMPLE_RATE)
            verdict = _trim_tail_via_whisper(wav, language, sentence)
            if verdict is True:
                trimmed, _ = sf.read(str(wav), dtype="float32", always_2d=False)
                return trimmed, 1.0, sentence, attempt + 1
            heard = transcribe.transcribe(wav, wav.with_suffix(".json"), language).text.strip()
            match = script_match(sentence, heard)
            expected = 0.09 * len(sentence) + 1.5
            if verdict is None and match >= 0.6 and len(audio) / SAMPLE_RATE <= expected:
                _trim_to_speech(wav)
                trimmed, _ = sf.read(str(wav), dtype="float32", always_2d=False)
                return trimmed, match, heard, attempt + 1
            if fallback is None or (match, -len(audio)) > (fallback[1], -len(fallback[0])):
                fallback = (audio, match, heard)
        log.warning("tts take %d rejected (%s, match %.2f): wanted %r heard %r", attempt + 1, verdict, match, sentence[:60], heard[:60])
    audio, match, heard = fallback
    limit = int((0.09 * len(sentence) + 1.5) * SAMPLE_RATE)
    return audio[:limit], match, heard, attempts


def _speak_events(text: str, language: str, voice, persona_id=None, verify: bool = True):
    import numpy as np
    model, conditioning = _speaker(voice, persona_id)
    sentences = split_sentences(text)
    for sentence_id, sentence in enumerate(sentences):
        if verify:
            audio, match, heard, takes = _verified_sentence(model, conditioning, sentence, language)
            chunks = [audio[i: i + CHUNK_SAMPLES] for i in range(0, len(audio), CHUNK_SAMPLES)] or [np.zeros(0, np.float32)]
            for index, chunk in enumerate(chunks):
                event = pcm_event(chunk, sentence_id, index == len(chunks) - 1, sentence)
                if index == len(chunks) - 1:
                    event.update(verified_match=match, takes=takes, seconds=round(len(audio) / SAMPLE_RATE, 2))
                yield event
        else:
            chunks = list(_synthesize(model, conditioning, sentence, language))
            for index, audio in enumerate(chunks):
                yield pcm_event(audio, sentence_id, index == len(chunks) - 1, sentence)


def _generate(body: Turn, path: Path):
    transcript = transcribe.transcribe(path, path.with_suffix(".json"), body.language)
    if not transcript.text.strip():
        yield {"type": "done", "reply": ""}
        return
    yield {"type": "transcript", "text": transcript.text}
    log.info("assistant heard: %r", transcript.text[:200])
    history = [{"role": m["role"], "content": m.get("content", "")[:2000]}
               for m in body.history if m.get("role") in ("user", "assistant")]
    messages = [
        {"role": "system", "content": (
            f"You are a friendly video assistant. Reply in {body.language}. Speak in short natural sentences, "
            "two to four of them, like a person talking; no markdown, no lists.")},
        *history,
        {"role": "user", "content": transcript.text},
    ]
    # Plan first: the whole reply text before any audio, so TTS has full context
    # and the caller knows the reply length when it schedules playout.
    reply = "".join(llm.stream(messages, temperature=0.5, max_tokens=MAX_TOKENS, timeout=30)).strip()
    if not reply:
        yield {"type": "done", "reply": ""}
        return
    log.info("assistant reply: %r", reply[:200])
    yield {"type": "reply", "text": reply, "sentences": len(split_sentences(reply))}
    yield from _speak_events(reply, body.language, body.voice, body.persona_id)
    yield {"type": "done", "reply": reply}


async def _ndjson(generator_factory):
    async def events():
        async with speech_lock(0):
            stream = generator_factory()
            try:
                while True:
                    value = await blocking_call(lambda: next(stream, None))
                    if value is None:
                        break
                    yield json.dumps(value) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            finally:
                await blocking_call(stream.close)
    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/respond")
async def respond(body: Turn):
    path = Path(body.audio_path).resolve()
    root = (settings.job_artifacts_dir / "avatars").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(400, "audio must be a recorded assistant utterance")
    from ..security import check_owner

    session_dir = root / path.relative_to(root).parts[0]
    ownership = session_dir / "owner.json"
    if not ownership.exists():
        raise HTTPException(404, "assistant session unavailable")
    check_owner(json.loads(ownership.read_text()))
    if body.language not in tts.XTTS_LANG_CODES:
        raise HTTPException(400, "voice does not support this language")
    if body.persona_id:
        from . import personas
        check_owner(personas._load(body.persona_id))
    return await _ndjson(lambda: _generate(body, path))


@router.post("/speak")
async def speak(body: Speak):
    if body.language not in tts.XTTS_LANG_CODES:
        raise HTTPException(400, "voice does not support this language")
    if body.persona_id:
        from . import personas
        from ..security import check_owner
        check_owner(personas._load(body.persona_id))
    return await _ndjson(lambda: _speak_events(body.text, body.language, body.voice, body.persona_id, body.verify))
