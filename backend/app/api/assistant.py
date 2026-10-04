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
MATCH_THRESHOLD = float(os.getenv("ASSISTANT_TTS_MATCH", "0.9"))          # share of the sentence's words a take must carry


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
    merged: list[str] = []
    for part in out:
        if merged and (len(merged[-1]) < 25 or len(part) < 25) and len(merged[-1]) + len(part) <= max_len:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def pcm_event(audio, sentence_id: int, final: bool, text: str) -> dict:
    import numpy as np
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16).tobytes()
    return {"type": "audio", "pcm_b64": base64.b64encode(pcm).decode(), "sample_rate": SAMPLE_RATE,
            "samples": int(len(clipped)), "sentence_id": sentence_id, "final": final, "text": text}


def _stock_conditioning(model, voice=None):
    speakers = model.speaker_manager.speakers
    name = voice or os.getenv("AVATAR_SPEAKER") or next(iter(speakers))
    if name not in speakers:
        raise RuntimeError("voice is not a bundled XTTS speaker")
    return speakers[name]


def _speaker(voice, persona_id=None):
    model = tts._get_xtts().synthesizer.tts_model
    if persona_id:
        from . import personas
        return model, personas.conditioning(persona_id)
    return model, _stock_conditioning(model, voice)


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


def _aligned_span(sentence: str, words):
    """(matched ratio, first word start, last word end) of the sentence inside the
    recognized words; babble before or after the sentence falls outside the span."""
    import difflib
    from .personas import normalize_words as norm

    target = norm(sentence)
    heard = [(norm(w.text if hasattr(w, "text") else w.word), w.start, w.end) for w in words]
    heard = [(h[0][0], h[1], h[2]) for h in heard if h[0] and h[1] is not None and h[2] is not None]
    if not target or not heard:
        return 0.0, None, None
    matcher = difflib.SequenceMatcher(a=target, b=[h[0] for h in heard], autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        # Nothing of the sentence was heard: the span of whatever speech there was,
        # so a caller that has to keep the take at least drops the silence around it.
        return 0.0, float(heard[0][1]), float(heard[-1][2])
    matched = sum(b.size for b in blocks) / len(target)
    first = heard[blocks[0].b][1]
    last = heard[blocks[-1].b + blocks[-1].size - 1][2]
    return matched, float(first), float(last)


def _verified_sentence(model, conditioning, sentence: str, language: str, attempts: int = 3, fallback_conditioning=None):
    """Synthesize one sentence and keep a take whose recognized words are the sentence,
    cut to the span of those words. Cloned voices babble before or after the text,
    especially on short sentences; the cut removes it and a take that still does not
    carry the words is re-synthesized. When every take fails and a stock voice is
    given, the sentence is said once in that voice instead of playing the babble.
    Returns (audio, matched ratio, heard, takes, used_fallback_voice)."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    from ..pipeline.tts import _trim_tail_via_whisper

    fallback = None
    expected = 0.09 * len(sentence) + 1.5
    for attempt in range(attempts):
        with torch.inference_mode():
            result = model.inference(sentence, tts.XTTS_LANG_CODES[language], conditioning["gpt_cond_latent"],
                                     conditioning["speaker_embedding"], temperature=max(0.45, 0.7 - 0.1 * attempt),
                                     repetition_penalty=10.0)
        audio = np.asarray(result["wav"], dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="speak-") as directory:
            wav = Path(directory) / "take.wav"
            sf.write(str(wav), audio, SAMPLE_RATE)
            if _trim_tail_via_whisper(wav, language, sentence) is True:
                trimmed, _ = sf.read(str(wav), dtype="float32", always_2d=False)
                return trimmed, 1.0, sentence, attempt + 1, False
            transcript = transcribe.transcribe(wav, wav.with_suffix(".json"), language)
        words = [w for seg in transcript.segments for w in seg.words]
        matched, first, last = _aligned_span(sentence, words)
        heard = transcript.text.strip()
        if first is not None:
            # Whisper places the first word's start late and the last word's end early
            # (initial consonants and final releases); pad so no word is clipped.
            cut = audio[max(0, int((first - 0.3) * SAMPLE_RATE)): int((last + 0.35) * SAMPLE_RATE)]
        else:
            cut = audio
        if matched >= MATCH_THRESHOLD and len(cut) / SAMPLE_RATE <= expected:
            return cut, matched, heard, attempt + 1, False
        if fallback is None or matched > fallback[1]:
            fallback = (cut, matched, heard)
        log.warning("tts take %d rejected (match %.2f, %.1fs): wanted %r heard %r", attempt + 1, matched,
                    len(cut) / SAMPLE_RATE, sentence[:60], heard[:80])
    if fallback_conditioning is not None:
        log.warning("tts: %d cloned takes failed for %r; saying it in the stock voice", attempts, sentence[:60])
        audio, matched, heard, takes, _ = _verified_sentence(model, fallback_conditioning, sentence, language, attempts=1)
        return audio, matched, heard, attempts + takes, True
    # The best take, cut to the words that were recognized; nothing beyond them. (An
    # earlier version also clipped it to a per-character duration estimate, which
    # could cut a slow take mid-word.)
    audio, matched, heard = fallback
    return audio, matched, heard, attempts, False


def _speak_events(text: str, language: str, voice, persona_id=None, verify: bool = True):
    import numpy as np
    model, conditioning = _speaker(voice, persona_id)
    stock = None
    if persona_id:
        try:
            stock = _stock_conditioning(model)       # a cloned voice that will not carry the words falls back to this
        except RuntimeError:
            stock = None
    sentences = split_sentences(text)
    for sentence_id, sentence in enumerate(sentences):
        if verify:
            audio, match, heard, takes, fallback = _verified_sentence(model, conditioning, sentence, language,
                                                                      fallback_conditioning=stock)
            chunks = [audio[i: i + CHUNK_SAMPLES] for i in range(0, len(audio), CHUNK_SAMPLES)] or [np.zeros(0, np.float32)]
            for index, chunk in enumerate(chunks):
                event = pcm_event(chunk, sentence_id, index == len(chunks) - 1, sentence)
                if index == len(chunks) - 1:
                    event.update(verified_match=match, takes=takes, fallback=fallback, seconds=round(len(audio) / SAMPLE_RATE, 2))
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
