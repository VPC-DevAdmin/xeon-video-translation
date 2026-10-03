"""Immutable job revisions, targeted retries, subtitles and download bundles."""

import copy
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from .. import storage, state_store, checkpoints, operations
from ..options import JobOptions
from ..security import principal, check_owner
from ..pipeline.orchestrator import (
    get_job,
    register_job,
    STAGE_NAMES,
    StageStatus,
    StageResult,
    blocking_call,
)
from .jobs import _kickoff

router = APIRouter(prefix="/jobs", tags=["studio"])


def owned(identifier):
    try:
        state = get_job(identifier)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if state is None:
        raise HTTPException(404, "job not found")
    check_owner(state.to_dict())
    return state


class Revision(BaseModel):
    from_stage: str | None = None
    translation: list[str] | None = Field(None, max_length=10000)
    transcript: list[str] | None = Field(None, max_length=10000)
    speakers: list[str] | None = Field(None, max_length=10000)
    options: JobOptions | None = None


@router.post("/{identifier}/revise", status_code=201)
async def revise(identifier: str, body: Revision, background: BackgroundTasks):
    parent = owned(identifier)
    if parent.status not in state_store.TERMINAL:
        raise HTTPException(409, "wait for or cancel the active job")
    await blocking_call(lambda: operations.check_capacity(principal.get()))
    if body.from_stage and body.from_stage not in STAGE_NAMES:
        raise HTTPException(400, "unknown stage")
    if body.from_stage:
        stage = STAGE_NAMES.index(body.from_stage)
        if (
            (body.transcript is not None or body.speakers is not None)
            and stage <= STAGE_NAMES.index("transcribe")
        ) or (body.translation is not None and stage <= STAGE_NAMES.index("translate")):
            raise HTTPException(400, "retry stage would overwrite the supplied edits")
    parent = copy.deepcopy(parent)
    by_name = {stage.name: stage for stage in parent.stages}
    parent.stages = [by_name.get(name, StageResult(name=name)) for name in STAGE_NAMES]
    source = storage.job_dir(identifier)
    edited_index = (
        STAGE_NAMES.index("translate")
        if body.translation is not None
        else STAGE_NAMES.index("transcribe")
        if body.transcript is not None or body.speakers is not None
        else None
    )
    if edited_index is not None:
        for stage in parent.stages[:edited_index]:
            if stage.status == StageStatus.SKIPPED:
                continue
            if not await blocking_call(lambda: checkpoints.valid(source, stage.name)):
                raise HTTPException(
                    409, f"retry {stage.name} before editing: upstream checkpoint unavailable"
                )
            stage.status = StageStatus.DONE
    inputs = list(source.glob("input.*"))
    if len(inputs) != 1:
        raise HTTPException(409, "source media unavailable")
    new_id = storage.new_job_id()
    destination = storage.job_dir(new_id)
    destination.mkdir()
    try:
        await blocking_call(lambda: shutil.copy2(inputs[0], destination / inputs[0].name))
        earliest = (
            STAGE_NAMES.index(body.from_stage)
            if body.from_stage
            else next(
                (
                    i
                    for i, s in enumerate(parent.stages)
                    if s.status not in (StageStatus.DONE, StageStatus.SKIPPED)
                ),
                len(STAGE_NAMES),
            )
        )
        # User-supplied documents complete the failed stage themselves.
        if body.transcript is not None or body.speakers is not None:
            earliest = max(earliest, STAGE_NAMES.index("translate"))
        if body.translation is not None:
            earliest = max(earliest, STAGE_NAMES.index("tts"))
        if body.options is not None:
            updated = body.options.model_dump()
            old = JobOptions.model_validate(parent.options).model_dump()
            if updated.get("glossary") != old.get("glossary", {}):
                earliest = min(earliest, STAGE_NAMES.index("translate"))
            if any(
                updated.get(k) != old.get(k)
                for k in ("voice", "speaker_voices", "rewrite_overruns")
            ):
                earliest = min(earliest, STAGE_NAMES.index("tts"))
            if any(updated.get(k) != old.get(k) for k in ("diarization", "alignment")):
                earliest = min(earliest, STAGE_NAMES.index("transcribe"))
            if updated.get("background_audio") != old.get("background_audio", False):
                earliest = min(earliest, STAGE_NAMES.index("audio"))
            if updated.get("background_gain") != old.get("background_gain", 0.35):
                earliest = min(earliest, STAGE_NAMES.index("mux"))
            if updated.get("windowed_lipsync") != old.get("windowed_lipsync", False):
                earliest = min(earliest, STAGE_NAMES.index("lipsync"))
        if body.transcript is not None or body.speakers is not None:
            earliest = min(earliest, STAGE_NAMES.index("translate"))
        if body.translation is not None:
            earliest = min(earliest, STAGE_NAMES.index("tts"))
        if body.translation is not None and earliest <= STAGE_NAMES.index("translate"):
            raise HTTPException(400, "apply upstream options before editing the translation")
        if (
            body.transcript is not None or body.speakers is not None
        ) and earliest <= STAGE_NAMES.index("transcribe"):
            raise HTTPException(400, "apply audio options before editing the transcript")
        for name in ("segments", "render-windows"):
            if (source / name).exists():
                await blocking_call(lambda: shutil.copytree(source / name, destination / name))
        for file in source.iterdir():
            if file.is_file() and file.name not in ("meta.json", inputs[0].name):
                await blocking_call(lambda: shutil.copy2(file, destination / file.name))
        state = copy.deepcopy(parent)
        state.job_id = new_id
        state.parent_job_id = identifier
        state.status = "queued"
        state.created_at = storage.now_iso()
        state.started_at = None
        state.completed_at = None
        state.error = None
        state.current_stage = None
        if body.options:
            state.options = body.options.model_dump()
        for kind, texts in (("transcript", body.transcript), ("translation", body.translation)):
            if texts is None and not (kind == "transcript" and body.speakers is not None):
                continue
            path = destination / f"{kind}.json"
            if not path.exists():
                raise HTTPException(409, f"{kind} is not available")
            document = json.loads(path.read_text())
            segments = document.get("segments", [])
            if texts is not None:
                if len(texts) != len(segments) or any(
                    not t.strip() or len(t) > 10000 for t in texts
                ):
                    raise HTTPException(400, "provide one nonempty text per segment")
                for segment, text in zip(segments, texts):
                    segment["text"] = text.strip()
                    segment.pop("words", None)
                document["text"] = " ".join(texts)
            if kind == "transcript" and body.speakers is not None:
                if len(body.speakers) != len(segments) or any(
                    not s or len(s) > 100 for s in body.speakers
                ):
                    raise HTTPException(400, "invalid speaker assignments")
                for segment, speaker in zip(segments, body.speakers):
                    segment["speaker"] = speaker
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2))
            edited_stage = "transcribe" if kind == "transcript" else "translate"
            checkpoints.record(destination, edited_stage)
            state.stages[STAGE_NAMES.index(edited_stage)] = StageResult(
                name=edited_stage, status=StageStatus.DONE
            )
        for i, name in enumerate(STAGE_NAMES):
            if i >= earliest:
                for artifact in checkpoints.ARTIFACTS[name]:
                    (destination / artifact).unlink(missing_ok=True)
                (destination / f"checkpoint-{name}.json").unlink(missing_ok=True)
                if i < len(state.stages):
                    state.stages[i] = StageResult(name=name)
        if earliest <= STAGE_NAMES.index("audio"):
            for name in ("vocals.wav", "background.wav", "remixed_audio.wav"):
                (destination / name).unlink(missing_ok=True)
        # Edited source documents are upstream of the invalidated stages.
        register_job(state)
        await blocking_call(lambda: operations.refresh_usage(state.job_id))
        storage.write_meta(new_id, state.to_dict())
        background.add_task(_kickoff, state, destination / inputs[0].name)
        return {"job_id": new_id, "parent_job_id": identifier, "status": "queued"}
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


@router.delete("/{identifier}")
async def delete(identifier: str):
    state = owned(identifier)
    if state.status not in state_store.TERMINAL:
        raise HTTPException(409, "cancel active work before deletion")
    state_store.remove(identifier)
    shutil.rmtree(storage.job_dir(identifier), ignore_errors=True)
    return {"status": "deleted"}


def timestamp(seconds, separator):
    value = max(0, round(float(seconds) * 1000))
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    sec, ms = divmod(value, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02}{separator}{ms:03}"


def subtitles(segments, kind):
    sep = "," if kind == "srt" else "."
    lines = ["WEBVTT\n"] if kind == "vtt" else []
    for i, segment in enumerate(segments, 1):
        text = segment["text"].replace("-->", "→").replace("\r", "").replace("\n", " ")
        lines.append(
            f"{i}\n{timestamp(segment['start'], sep)} --> {timestamp(segment['end'], sep)}\n{text}\n"
        )
    return "\n".join(lines)


@router.get("/{identifier}/subtitles/{kind}")
async def export_subtitles(identifier: str, kind: str):
    owned(identifier)
    if kind not in ("srt", "vtt"):
        raise HTTPException(400, "choose srt or vtt")
    path = storage.job_artifact_path(identifier, "translation.json")
    if not path.exists():
        raise HTTPException(404, "translation unavailable")
    return Response(
        subtitles(json.loads(path.read_text())["segments"], kind),
        media_type="text/vtt" if kind == "vtt" else "application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="translation.{kind}"'},
    )


@router.get("/{identifier}/bundle")
async def bundle(identifier: str):
    state = owned(identifier)
    if state.status not in state_store.TERMINAL:
        raise HTTPException(409, "wait for processing to finish")
    directory = Path(tempfile.mkdtemp(prefix="job-export-"))
    output = directory / "translation.zip"
    try:

        def compress():
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in (
                    "final.mp4",
                    "translated_audio.wav",
                    "translated_audio.timing.json",
                    "transcript.json",
                    "translation.json",
                    "meta.json",
                ):
                    path = storage.job_artifact_path(identifier, name)
                    if path.exists():
                        archive.write(path, name)
                path = storage.job_artifact_path(identifier, "translation.json")
                if path.exists():
                    for kind in ("srt", "vtt"):
                        archive.writestr(
                            f"translation.{kind}",
                            subtitles(json.loads(path.read_text())["segments"], kind),
                        )

        await blocking_call(compress)
        return FileResponse(
            output,
            filename=f"{identifier}.zip",
            background=BackgroundTask(shutil.rmtree, directory, True),
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
