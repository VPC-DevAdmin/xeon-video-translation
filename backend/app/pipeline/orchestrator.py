"""Pipeline orchestrator.

Runs stages 1–4 (M1+M2+M3). Emits events to a per-job asyncio.Queue that the
SSE endpoint consumes. Stages 5–6 are stubbed for future milestones.

Stages run synchronously inside a worker thread; the orchestrator exposes an
async wrapper so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .. import storage
from . import audio, transcribe, translate, tts

log = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    name: str
    status: StageStatus = StageStatus.PENDING
    duration_ms: int | None = None
    output: Any | None = None
    error: str | None = None


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    current_stage: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    target_language: str = "es"
    source_language: str | None = None
    input_filename: str = ""
    created_at: str = field(default_factory=storage.now_iso)
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "target_language": self.target_language,
            "source_language": self.source_language,
            "input_filename": self.input_filename,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration_ms": s.duration_ms,
                    "output": s.output,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


# Defines the full pipeline. Stages 4–6 are placeholders that the orchestrator
# marks SKIPPED until those milestones land.
STAGE_NAMES = ["audio", "transcribe", "translate", "tts", "lipsync", "mux"]


# --------------------------------------------------------------------------- #
# In-memory registry of running jobs and their event queues.
# Single-process, single-worker is fine for v1 (MAX_CONCURRENT_JOBS=1).
# --------------------------------------------------------------------------- #

_jobs: dict[str, JobState] = {}
_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
_run_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-create so the semaphore binds to the running loop."""
    global _run_semaphore
    if _run_semaphore is None:
        from ..config import settings
        _run_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _run_semaphore


def get_job(job_id: str) -> JobState | None:
    if job_id in _jobs:
        return _jobs[job_id]
    meta = storage.read_meta(job_id)
    if meta is None:
        return None
    # Reconstruct minimal state from meta for completed jobs.
    state = JobState(
        job_id=job_id,
        status=meta.get("status", "completed"),
        current_stage=meta.get("current_stage"),
        target_language=meta.get("target_language", "es"),
        source_language=meta.get("source_language"),
        input_filename=meta.get("input_filename", ""),
        created_at=meta.get("created_at", storage.now_iso()),
        completed_at=meta.get("completed_at"),
        error=meta.get("error"),
        stages=[
            StageResult(
                name=s["name"],
                status=StageStatus(s["status"]),
                duration_ms=s.get("duration_ms"),
                output=s.get("output"),
                error=s.get("error"),
            )
            for s in meta.get("stages", [])
        ],
    )
    return state


def get_queue(job_id: str) -> asyncio.Queue[dict[str, Any]] | None:
    return _queues.get(job_id)


def register_job(state: JobState) -> asyncio.Queue[dict[str, Any]]:
    _jobs[state.job_id] = state
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _queues[state.job_id] = q
    return q


def _persist(state: JobState) -> None:
    storage.write_meta(state.job_id, state.to_dict())


async def _emit(queue: asyncio.Queue[dict[str, Any]], event: str, data: dict[str, Any]) -> None:
    await queue.put({"event": event, "data": data})


async def run_pipeline(
    state: JobState,
    input_path: Path,
) -> None:
    """Run the full pipeline for `state`, writing artifacts under jobs/<id>/."""
    queue = _queues[state.job_id]
    sem = _get_semaphore()

    async with sem:
        state.status = "running"
        _persist(state)
        await _emit(queue, "job_started", {"job_id": state.job_id})

        # Pre-create stage records so the UI can render the full strip immediately.
        state.stages = [StageResult(name=n) for n in STAGE_NAMES]

        try:
            await _run_stage_audio(state, queue, input_path)
            await _run_stage_transcribe(state, queue)
            await _run_stage_translate(state, queue)
            await _run_stage_tts(state, queue)

            # Stages 5–6 not implemented yet — mark and move on.
            for name in ("lipsync", "mux"):
                _stage(state, name).status = StageStatus.SKIPPED
                await _emit(queue, "stage_skipped", {"stage": name})

            state.status = "completed"
            state.completed_at = storage.now_iso()
            _persist(state)
            await _emit(queue, "job_completed", {"job_id": state.job_id})
        except Exception as e:
            state.status = "failed"
            state.error = f"{type(e).__name__}: {e}"
            state.completed_at = storage.now_iso()
            log.exception("pipeline failed for job %s", state.job_id)
            _persist(state)
            await _emit(queue, "error", {
                "job_id": state.job_id,
                "error": state.error,
                "trace": traceback.format_exc(limit=3),
            })
        finally:
            await _emit(queue, "stream_end", {})


# --------------------------------------------------------------------------- #
# Stage runners. Each one updates state, emits events, persists meta.
# --------------------------------------------------------------------------- #


def _stage(state: JobState, name: str) -> StageResult:
    for s in state.stages:
        if s.name == name:
            return s
    raise KeyError(name)


async def _run_stage_audio(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
    input_path: Path,
) -> None:
    name = "audio"
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    _persist(state)
    await _emit(queue, "stage_started", {"stage": name})

    out_path = storage.job_artifact_path(state.job_id, "audio.wav")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        duration = audio.probe_duration_seconds(input_path)
        from ..config import settings
        if duration > settings.max_video_duration_seconds:
            raise audio.AudioExtractionError(
                f"clip is {duration:.1f}s, max is {settings.max_video_duration_seconds}s"
            )
        audio.extract_audio(input_path, out_path)
        return {"path": out_path.name, "duration_seconds": duration}

    try:
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = result
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(queue, "stage_completed", {"stage": name, "output": result,
                                                "duration_ms": stage.duration_ms})
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_transcribe(
    state: JobState, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    name = "transcribe"
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    _persist(state)
    await _emit(queue, "stage_started", {"stage": name})

    audio_path = storage.job_artifact_path(state.job_id, "audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "transcript.json")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        return transcribe.transcribe(audio_path, out_path, language=state.source_language).to_dict()

    try:
        result = await asyncio.to_thread(_do)
        # Auto-detected source language flows into the translate stage.
        if not state.source_language:
            state.source_language = result.get("language")
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "language": result["language"],
            "language_probability": result["language_probability"],
            "text": result["text"],
            "segment_count": len(result["segments"]),
            "path": out_path.name,
        }
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(queue, "stage_completed", {"stage": name, "output": stage.output,
                                                "duration_ms": stage.duration_ms,
                                                "transcript": result})
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_tts(
    state: JobState, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    name = "tts"
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    _persist(state)
    await _emit(queue, "stage_started", {"stage": name})

    translation_path = storage.job_artifact_path(state.job_id, "translation.json")
    reference_audio = storage.job_artifact_path(state.job_id, "audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "translated_audio.wav")
    started = time.perf_counter()

    import json
    translation = json.loads(translation_path.read_text())

    def _do() -> dict[str, Any]:
        return tts.synthesize(
            translation=translation,
            reference_audio=reference_audio,
            output_path=out_path,
        ).to_dict()

    try:
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "backend": result["backend"],
            "language": result["language"],
            "path": out_path.name,
        }
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(queue, "stage_completed", {
            "stage": name, "output": stage.output,
            "duration_ms": stage.duration_ms,
        })
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_translate(
    state: JobState, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    name = "translate"
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    _persist(state)
    await _emit(queue, "stage_started", {"stage": name})

    transcript_path = storage.job_artifact_path(state.job_id, "transcript.json")
    out_path = storage.job_artifact_path(state.job_id, "translation.json")
    started = time.perf_counter()

    import json
    transcript = json.loads(transcript_path.read_text())

    def _do() -> dict[str, Any]:
        return translate.translate(
            transcript=transcript,
            output_path=out_path,
            target_language=state.target_language,
            source_language=state.source_language,
        ).to_dict()

    try:
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "source_language": result["source_language"],
            "target_language": result["target_language"],
            "backend": result["backend"],
            "text": result["text"],
            "segment_count": len(result["segments"]),
            "path": out_path.name,
        }
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(queue, "stage_completed", {"stage": name, "output": stage.output,
                                                "duration_ms": stage.duration_ms,
                                                "translation": result})
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise
