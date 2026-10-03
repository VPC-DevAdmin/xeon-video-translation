"""Speech controls shared by the editor, live captions and avatars."""

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from ..config import settings
from ..security import check_owner
from ..pipeline.orchestrator import speech_lock, blocking_call
from ..pipeline import transcribe, tts
from ..pipeline.windowed import duration

router = APIRouter(prefix="/speech", tags=["speech"])


class PreviewTranscript(BaseModel):
    audio_path: str
    language: str | None = None


@router.post("/transcribe")
async def preview_transcript(body: PreviewTranscript):
    path = Path(body.audio_path).resolve()
    root = settings.job_artifacts_dir.resolve()
    if not path.is_file() or not any(
        path.is_relative_to(root / folder) for folder in ("avatars", "ingest")
    ):
        raise HTTPException(400, "recorded session audio required")
    relative = path.relative_to(root)
    ownership = root / relative.parts[0] / relative.parts[1] / "owner.json"
    if not ownership.exists():
        raise HTTPException(404, "session ownership unavailable")
    check_owner(json.loads(ownership.read_text()))
    # Ingest sessions enforce ownership before requesting a preview.
    if await blocking_call(lambda: duration(path)) > 16:
        raise HTTPException(413, "preview must be at most 16 seconds")
    async with speech_lock(30):
        result = await blocking_call(
            lambda: transcribe.transcribe(path, path.with_suffix(".json"), body.language).to_dict()
        )
    return {"text": result["text"], "language": result["language"], "provisional": True}


@router.get("/voices")
async def voices():
    async with speech_lock(20):

        def load():
            model = tts._get_xtts().synthesizer.tts_model
            return sorted(model.speaker_manager.speakers)

        names = await blocking_call(load)
    return {"voices": names, "languages": sorted(tts.XTTS_LANG_CODES)}


class VoicePreview(BaseModel):
    voice: str = Field(min_length=1, max_length=100)
    language: str = "en"
    text: str = Field(default="Hello. This is a preview of my voice.", min_length=1, max_length=250)


@router.post("/preview")
async def preview_voice(body: VoicePreview):
    if body.language not in tts.XTTS_LANG_CODES:
        raise HTTPException(400, "unsupported voice language")
    root = settings.job_artifacts_dir / "previews"
    root.mkdir(exist_ok=True)
    path = root / (uuid.uuid4().hex + ".wav")
    async with speech_lock(10):

        def generate():
            model = tts._get_xtts()
            if body.voice not in model.synthesizer.tts_model.speaker_manager.speakers:
                raise HTTPException(400, "unknown voice")
            model.tts_to_file(
                text=body.text,
                speaker=body.voice,
                language=tts.XTTS_LANG_CODES[body.language],
                file_path=str(path),
            )

        try:
            await blocking_call(generate)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return FileResponse(
        path, media_type="audio/wav", background=BackgroundTask(path.unlink, missing_ok=True)
    )
