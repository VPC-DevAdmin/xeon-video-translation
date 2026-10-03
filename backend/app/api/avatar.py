"""Streaming assistant turns: ASR, LLM tokens (OpenAI-compatible), XTTS audio chunks."""

from __future__ import annotations
import json
import os
import re
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ..config import settings
from .. import llm
from ..pipeline.orchestrator import blocking_call, speech_lock
from ..pipeline import transcribe, tts

router = APIRouter(prefix="/avatar", tags=["avatar"])


class Turn(BaseModel):
    audio_path: str
    language: str = "en"
    voice: str | None = Field(None, max_length=100)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


def _sentences(messages):
    pending = ""
    for delta in llm.stream(messages, temperature=0.4, max_tokens=160, timeout=30):
        pending += delta
        if re.search(r"[.!?。！？]\s*$", pending) or len(pending) > 180:
            yield pending.strip()
            pending = ""
    if pending.strip():
        yield pending.strip()


def _generate(body, path):
    import numpy as np
    import soundfile as sf
    import torch

    transcript = transcribe.transcribe(path, path.with_suffix(".json"), body.language)
    if not transcript.text.strip():
        yield {"type": "done"}
        return
    yield {"type": "transcript", "text": transcript.text}
    history = [
        {"role": m["role"], "content": m.get("content", "")[:2000]}
        for m in body.history
        if m.get("role") in ("user", "assistant")
    ]
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful voice assistant. Reply in {body.language}. Use concise spoken sentences; no markdown.",
        },
        *history,
        {"role": "user", "content": transcript.text},
    ]
    model = tts._get_xtts().synthesizer.tts_model
    speakers = model.speaker_manager.speakers
    speaker = body.voice or os.getenv("AVATAR_SPEAKER") or next(iter(speakers))
    if speaker not in speakers:
        raise RuntimeError("AVATAR_SPEAKER is not a bundled XTTS speaker")
    conditioning = speakers[speaker]
    from ..streaming import prefetch

    for sentence_id, text in enumerate(prefetch(_sentences(messages))):
        yield {"type": "text", "text": text, "sentence_id": sentence_id}
        # Enter inference_mode around next(), not across yields to another thread.
        stream = model.inference_stream(
            text,
            tts.XTTS_LANG_CODES[body.language],
            conditioning["gpt_cond_latent"],
            conditioning["speaker_embedding"],
            stream_chunk_size=20,
        )
        pending = []
        length = 0
        previous_audio = None
        while True:
            with torch.inference_mode():
                chunk = next(stream, None)
                if chunk is not None:
                    chunk = chunk.detach().float().cpu().numpy().reshape(-1)
            if chunk is not None:
                pending.append(chunk)
                length += len(chunk)
            if length >= 12000 or (chunk is None and length):
                audio = np.concatenate(pending)
                if len(audio) < 2400:
                    audio = np.pad(audio, (0, 2400 - len(audio)))
                output = path.parent / f"reply-{uuid.uuid4().hex}.wav"
                sf.write(output, audio, 24000)
                # Hold one chunk so the receiver knows which chunk ends a sentence.
                if previous_audio is not None:
                    yield {
                        "type": "audio",
                        "path": str(previous_audio),
                        "sentence_id": sentence_id,
                        "text": text,
                        "final": False,
                    }
                previous_audio = output
                pending, length = [], 0
            if chunk is None:
                break
        if previous_audio is not None:
            yield {
                "type": "audio",
                "path": str(previous_audio),
                "sentence_id": sentence_id,
                "text": text,
                "final": True,
            }
    yield {"type": "done"}


@router.post("/respond")
async def respond(body: Turn):
    path = Path(body.audio_path).resolve()
    root = (settings.job_artifacts_dir / "avatars").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(400, "audio must be a recorded avatar utterance")
    from ..security import check_owner

    session_dir = root / path.relative_to(root).parts[0]
    ownership = session_dir / "owner.json"
    if not ownership.exists():
        raise HTTPException(404, "avatar session unavailable")
    check_owner(json.loads(ownership.read_text()))
    if body.language not in tts.XTTS_LANG_CODES:
        raise HTTPException(400, "avatar voice does not support this language")

    async def events():
        # The default shares speech GPU capacity safely. A separate backend
        # instance/GPU can serve this endpoint for dedicated avatar capacity.
        async with speech_lock(0):
            stream = _generate(body, path)
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
