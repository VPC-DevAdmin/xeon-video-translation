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
from . import audio, lipsync, transcribe, translate, tts, watermark

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
    # 0.0–1.0 progress for stages that emit it (only wav2lip today).
    progress: float | None = None
    # Rough ETA in seconds at the moment the stage started.
    eta_seconds: float | None = None


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    current_stage: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    target_language: str = "es"
    source_language: str | None = None
    # Per-job lipsync backend override; None means "use settings.lipsync_backend".
    lipsync_backend: str | None = None
    # Per-request tuning forwarded to the lipsync-musetalk service. All
    # None means "use the service's env-derived defaults" (pre-PR behavior).
    lipsync_quality: dict[str, Any] | None = None
    # Per-job TTS backend override ("xtts" | "f5tts"). None means "use
    # settings.tts_backend" (env default, currently xtts).
    tts_backend: str | None = None
    # Source clip duration in seconds, used for ETA calculations downstream.
    source_duration_seconds: float | None = None
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
            "lipsync_backend": self.lipsync_backend,
            "lipsync_quality": self.lipsync_quality,
            "tts_backend": self.tts_backend,
            "source_duration_seconds": self.source_duration_seconds,
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
                    "progress": s.progress,
                    "eta_seconds": s.eta_seconds,
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
        lipsync_backend=meta.get("lipsync_backend"),
        lipsync_quality=meta.get("lipsync_quality"),
        tts_backend=meta.get("tts_backend"),
        source_duration_seconds=meta.get("source_duration_seconds"),
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
                progress=s.get("progress"),
                eta_seconds=s.get("eta_seconds"),
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
            await _run_stage_lipsync(state, queue, input_path)
            await _run_stage_mux(state, queue, input_path)

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
# ETA model. Deliberately conservative; gets revised once stages run.
# --------------------------------------------------------------------------- #


def estimate_eta_seconds(state: JobState, stage_name: str) -> float | None:
    """Rough ETA for `stage_name`, in seconds, based on source duration."""
    src = state.source_duration_seconds
    backend = lipsync.backend_in_use(state.lipsync_backend)
    if src is None and stage_name not in ("audio",):
        return None
    if stage_name == "audio":
        return 1.0
    if stage_name == "transcribe":
        # faster-whisper base int8 ~ 6× realtime on Xeon.
        return max(2.0, src / 6.0) if src else None
    if stage_name == "translate":
        # NLLB-600M ~ 3s per segment; rough proxy is src/2.
        return max(3.0, src / 2.0) if src else None
    if stage_name == "tts":
        # XTTS-v2 ~ 0.5× realtime on CPU; src is the upper bound of output duration.
        return max(5.0, src * 2.0) if src else None
    if stage_name == "lipsync":
        factor = lipsync.realtime_factor(backend)
        if factor == 0.0 or src is None:
            return 0.0
        return src * factor
    if stage_name == "mux":
        return 2.0
    return None


# --------------------------------------------------------------------------- #
# Cross-thread progress emission. Worker threads call the returned function
# with a float in [0.0, 1.0]; it bridges back to the loop safely.
# --------------------------------------------------------------------------- #


def _progress_emitter(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
    state: JobState,
    stage_name: str,
):
    stage = _stage(state, stage_name)
    last_emit = [0.0]  # mutable cell

    def cb(p: float) -> None:
        p = max(0.0, min(1.0, float(p)))
        stage.progress = p
        # Throttle: only emit every 2% to avoid flooding the SSE queue.
        if p - last_emit[0] < 0.02 and p < 1.0:
            return
        last_emit[0] = p
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"event": "stage_progress", "data": {"stage": stage_name, "percent": p}},
        )

    return cb


# --------------------------------------------------------------------------- #
# Stage runners. Each one updates state, emits events, persists meta.
# --------------------------------------------------------------------------- #


def _stage(state: JobState, name: str) -> StageResult:
    for s in state.stages:
        if s.name == name:
            return s
    raise KeyError(name)


async def _start_stage(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
    name: str,
) -> StageResult:
    """Common prologue for every stage: mark running, publish ETA, emit event."""
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    stage.eta_seconds = estimate_eta_seconds(state, name)
    _persist(state)
    await _emit(queue, "stage_started", {
        "stage": name,
        "eta_seconds": stage.eta_seconds,
    })
    return stage


async def _run_stage_audio(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
    input_path: Path,
) -> None:
    name = "audio"
    stage = await _start_stage(state, queue, name)

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
        # Now that we know source duration, broadcast ETAs for the rest of the pipeline.
        state.source_duration_seconds = float(result["duration_seconds"])
        remaining_etas = {
            n: estimate_eta_seconds(state, n)
            for n in ("transcribe", "translate", "tts", "lipsync", "mux")
        }
        _persist(state)
        await _emit(queue, "stage_completed", {"stage": name, "output": result,
                                                "duration_ms": stage.duration_ms})
        await _emit(queue, "pipeline_etas", {
            "source_duration_seconds": state.source_duration_seconds,
            "lipsync_backend": lipsync.backend_in_use(state.lipsync_backend),
            "etas": remaining_etas,
        })
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
    stage = await _start_stage(state, queue, name)

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
    stage = await _start_stage(state, queue, name)

    translation_path = storage.job_artifact_path(state.job_id, "translation.json")
    transcript_path = storage.job_artifact_path(state.job_id, "transcript.json")
    reference_audio = storage.job_artifact_path(state.job_id, "audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "translated_audio.wav")
    started = time.perf_counter()

    import json
    translation = json.loads(translation_path.read_text())
    transcript = json.loads(transcript_path.read_text()) if transcript_path.exists() else {}
    first_speech = transcript.get("first_speech_seconds")
    # Per-segment TTS + smart reference selection both need transcript
    # segments with word timestamps. Missing/empty → synthesize falls back
    # to the single-shot path.
    transcript_segments = transcript.get("segments") or None

    def _do() -> dict[str, Any]:
        return tts.synthesize(
            translation=translation,
            reference_audio=reference_audio,
            output_path=out_path,
            first_speech_seconds=first_speech,
            source_duration_seconds=state.source_duration_seconds,
            transcript_segments=transcript_segments,
            backend=state.tts_backend,
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
    stage = await _start_stage(state, queue, name)

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


async def _run_stage_lipsync(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
    input_path: Path,
) -> None:
    name = "lipsync"
    stage = await _start_stage(state, queue, name)

    audio_path = storage.job_artifact_path(state.job_id, "translated_audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "lipsynced.mp4")
    started = time.perf_counter()

    backend = lipsync.backend_in_use(state.lipsync_backend)
    loop = asyncio.get_running_loop()
    progress_cb = _progress_emitter(loop, queue, state, name)

    def _do() -> dict[str, Any]:
        result = lipsync.run(
            backend=backend,
            video_in=input_path,
            audio_in=audio_path,
            output_path=out_path,
            progress=progress_cb,
            quality_overrides=state.lipsync_quality,
        )
        return result.to_dict()

    try:
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "backend": result["backend"],
            "passthrough": result["passthrough"],
            "path": out_path.name,
        }
        stage.progress = 1.0
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


async def _run_stage_mux(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
    input_path: Path,
) -> None:
    name = "mux"
    stage = await _start_stage(state, queue, name)

    # Video input: if lipsync produced something, use that. Otherwise the original upload.
    lipsynced = storage.job_artifact_path(state.job_id, "lipsynced.mp4")
    video_in = lipsynced if lipsynced.exists() else input_path
    audio_in = storage.job_artifact_path(state.job_id, "translated_audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "final.mp4")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        return watermark.mux_and_watermark(
            video_path=video_in,
            audio_path=audio_in,
            output_path=out_path,
        ).to_dict()

    try:
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "path": out_path.name,
            "watermark": result["watermark"],
            "source_video": video_in.name,
        }
        stage.progress = 1.0
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
