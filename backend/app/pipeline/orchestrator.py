"""Pipeline orchestrator.

Runs the translation stages in separate fast and batch lanes. Each job has
an event log with independent SSE subscribers and a durable metadata snapshot.

Stages run synchronously inside a worker thread; the orchestrator exposes an
async wrapper so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .. import storage, state_store, checkpoints, operations
from ..config import settings
from ..events import EventLog
from . import audio, lipsync, stabilize, transcribe, translate, tts, watermark

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
    owner_id: str = "local"
    parent_job_id: str | None = None
    options: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    mode: str | None = None
    status: str = "queued"  # queued | running | completed | failed | cancelled
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
    # Per-job pre-stabilization toggle. True runs vidstab (or deshake
    # fallback) before transcribe/lipsync see the video. None means
    # "use settings.enable_video_stabilization" (env default, False).
    enable_stabilization: bool | None = None
    # Per-job post-stabilization toggle. True runs vidstab on the
    # lipsynced video before the final mux. Catches jitter from any
    # source (source motion, pipeline artifacts, sub-pixel VAE drift).
    # None means "use settings.enable_output_stabilization" (env
    # default, False). Can stack with pre-stabilization or be used
    # independently.
    enable_output_stabilization: bool | None = None
    # Source clip duration in seconds, used for ETA calculations downstream.
    source_duration_seconds: float | None = None
    input_filename: str = ""
    created_at: str = field(default_factory=storage.now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "mode": self.mode,
            "owner_id": self.owner_id,
            "parent_job_id": self.parent_job_id,
            "options": self.options,
            "provenance": self.provenance,
            "current_stage": self.current_stage,
            "target_language": self.target_language,
            "source_language": self.source_language,
            "lipsync_backend": self.lipsync_backend,
            "lipsync_quality": self.lipsync_quality,
            "tts_backend": self.tts_backend,
            "enable_stabilization": self.enable_stabilization,
            "enable_output_stabilization": self.enable_output_stabilization,
            "source_duration_seconds": self.source_duration_seconds,
            "input_filename": self.input_filename,
            "created_at": self.created_at,
            "started_at": self.started_at,
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


# Full translation pipeline; stabilization stages are optional.
STAGE_NAMES = [
    "audio",
    "stabilize",  # optional pre-stabilize; SKIPPED when not requested
    "transcribe",
    "translate",
    "tts",
    "lipsync",
    "poststabilize",  # optional post-stabilize; SKIPPED when not requested
    "mux",
]


# --------------------------------------------------------------------------- #
# In-memory registry of running jobs and their event queues.
# Single-process, single-worker is fine for v1 (MAX_CONCURRENT_JOBS=1).
# --------------------------------------------------------------------------- #

_jobs: dict[str, JobState] = {}
_queues: dict[str, EventLog] = {}
_tasks: dict[str, asyncio.Task] = {}
import threading

_cancel_signals: dict[str, threading.Event] = {}
_shutting_down = False


def get_job(job_id: str) -> JobState | None:
    if job_id in _jobs:
        return _jobs[job_id]
    meta = storage.read_meta(job_id)
    if meta is None:
        return None
    # Reconstruct minimal state from meta for completed jobs.
    state = JobState(
        job_id=job_id,
        mode=meta.get("mode"),
        owner_id=meta.get("owner_id", "local"),
        parent_job_id=meta.get("parent_job_id"),
        options=meta.get("options", {}),
        provenance=meta.get("provenance", {}),
        enable_stabilization=meta.get("enable_stabilization"),
        enable_output_stabilization=meta.get("enable_output_stabilization"),
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
        started_at=meta.get("started_at"),
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


def get_queue(job_id: str) -> EventLog | None:
    return _queues.get(job_id)


def register_job(state: JobState) -> EventLog:
    storage.write_meta(state.job_id, state.to_dict())
    _jobs[state.job_id] = state
    q = _queues.setdefault(state.job_id, EventLog())
    return q


def pending_count() -> int:
    return sum(s.status in ("queued", "running", "cancelling") for s in _jobs.values())


def cancel_job(job_id: str) -> tuple[bool, str]:
    state = _jobs.get(job_id)
    if state is None:
        return False, "unknown"
    if state.status in ("completed", "failed", "cancelled"):
        return False, "already-terminal"
    if state.status == "queued" and job_id not in _tasks:
        state.status = "cancelled"
        state.error = "cancelled by user before dispatch"
        state.completed_at = storage.now_iso()
        _persist(state)
        queue = _queues.pop(job_id, None)
        if queue:
            queue.put_nowait({"event": "error", "data": {"error": state.error}})
            queue.put_nowait({"event": "stream_end", "data": {}})
        _jobs.pop(job_id, None)
        return True, "cancelled"
    if state.status == "queued" and (task := _tasks.get(job_id)) is not None:
        task.cancel()
    if job_id in _cancel_signals:
        _cancel_signals[job_id].set()
    state.status = "cancelling"
    state.error = "cancellation requested; finishing the active inference safely"
    _persist(state)
    return True, "cancelling"


_lanes: dict[str, asyncio.Semaphore] = {}
_speech_lock: asyncio.Lock | None = None


def speech_lock(priority=10):
    global _speech_lock
    if _speech_lock is None:
        from ..scheduler import SpeechScheduler

        _speech_lock = SpeechScheduler()
    return _speech_lock.lease(priority)


async def blocking_call(fn):
    # Cancelling an asyncio wrapper cannot stop native inference. Keep the
    # resource lease until its thread exits, including during shutdown.
    task = asyncio.create_task(asyncio.to_thread(fn))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
        except Exception:
            log.exception("inference failed while draining cancellation")
        raise


async def recover_jobs() -> None:
    global _shutting_down
    _shutting_down = False
    state_store.recover_interrupted()
    # One API worker owns the durable on-disk queue. Queued jobs resume;
    # interrupted inference is marked failed instead of duplicating GPU work.
    for directory in settings.job_artifacts_dir.iterdir():
        if not directory.is_dir():
            continue
        try:
            state = get_job(directory.name)
        except (ValueError, OSError):
            continue
        if state is None or state.status not in ("queued", "running", "cancelling"):
            continue
        inputs = list(directory.glob("input.*"))
        if state.status == "queued" and len(inputs) == 1:
            register_job(state)
            asyncio.create_task(run_pipeline(state, inputs[0]))
        else:
            state.status = "failed"
            state.error = "service restarted during inference; submit again to retry"
            state.completed_at = storage.now_iso()
            _persist(state)


def _persist(state: JobState) -> None:
    storage.write_meta(state.job_id, state.to_dict())


async def _emit(queue: EventLog, event: str, data: dict[str, Any]) -> None:
    await queue.put({"event": event, "data": data})


async def run_pipeline(state: JobState, input_path: Path) -> None:
    import uuid
    from datetime import datetime

    if _shutting_down:
        return  # persisted queued work is recovered by the next dispatcher
    existing = _tasks.get(state.job_id)
    if existing is not None and existing is not asyncio.current_task() and not existing.done():
        return  # duplicate dispatch must not close the winning worker's stream
    token = uuid.uuid4().hex
    if not state_store.claim(state.job_id, token):
        queue = _queues.pop(state.job_id, None)
        _jobs.pop(state.job_id, None)
        if queue:
            durable = state_store.read(state.job_id) or {}
            await _emit(queue, "snapshot", durable)
            await _emit(queue, "stream_end", {})
        return
    queue = _queues[state.job_id]

    async def renew():
        while True:
            await asyncio.sleep(15)
            if not state_store.heartbeat(state.job_id, token):
                cancel_job(state.job_id)
                return

    renewal = asyncio.create_task(renew())
    backend = lipsync.backend_in_use(state.lipsync_backend)
    lane = "batch" if backend == "latentsync" else "fast"
    sem = _lanes.setdefault(lane, asyncio.Semaphore(1))
    _tasks[state.job_id] = asyncio.current_task()
    _cancel_signals[state.job_id] = threading.Event()
    try:
        async with sem:
            if state.status == "cancelling":
                raise asyncio.CancelledError()
            state.started_at = storage.now_iso()
            state.status = "running"
            if not state.stages:
                state.stages = [StageResult(name=n) for n in STAGE_NAMES]
            current_provenance = operations.provenance()
            configuration_changed = bool(
                state.provenance and state.provenance != current_provenance
            )
            state.provenance = current_provenance
            state_store.metric(
                state.job_id,
                "queue_seconds",
                (
                    datetime.fromisoformat(state.started_at)
                    - datetime.fromisoformat(state.created_at)
                ).total_seconds(),
            )
            _persist(state)
            await _emit(queue, "job_started", {"job_id": state.job_id})
            invalidated = configuration_changed

            async def execute(name, runner, *args):
                nonlocal invalidated
                if state.status == "cancelling":
                    raise asyncio.CancelledError()
                stage = _stage(state, name)
                directory = storage.job_dir(state.job_id)
                if (
                    not invalidated
                    and stage.status == StageStatus.DONE
                    and checkpoints.valid(directory, name)
                ):
                    await _emit(queue, "stage_reused", {"stage": name})
                    return
                if stage.status != StageStatus.SKIPPED:
                    invalidated = True
                started = time.perf_counter()
                await runner(state, queue, *args)
                if stage.status == StageStatus.DONE:
                    checkpoints.record(directory, name)
                await blocking_call(lambda: operations.refresh_usage(state.job_id))
                state_store.metric(state.job_id, name + "_seconds", time.perf_counter() - started)

            await execute("audio", _run_stage_audio, input_path)
            if state.options.get("background_audio", settings.enable_background_audio):
                from . import enhancements

                background = storage.job_artifact_path(state.job_id, "background.wav")
                if not background.exists():
                    await blocking_call(
                        lambda: enhancements.separate(input_path, background.parent)
                    )
            await execute("stabilize", _run_stage_stabilize, input_path)
            stabilized = storage.job_artifact_path(state.job_id, "stabilized.mp4")
            if _stage(state, "stabilize").status == StageStatus.DONE and stabilized.exists():
                input_path = stabilized
            for name, runner in (
                ("transcribe", _run_stage_transcribe),
                ("translate", _run_stage_translate),
                ("tts", _run_stage_tts),
            ):
                async with speech_lock(20 if lane == "batch" else 10):
                    await execute(name, runner)
            await execute("lipsync", _run_stage_lipsync, input_path)
            await execute("poststabilize", _run_stage_poststabilize)
            await execute("mux", _run_stage_mux, input_path)
            if state.status == "cancelling":
                raise asyncio.CancelledError()
            state.status = "completed"
            state.completed_at = storage.now_iso()
            _persist(state)
            await _emit(queue, "job_completed", {"job_id": state.job_id})
    except asyncio.CancelledError:
        interrupted = _shutting_down and state.status != "cancelling"
        state.status = "queued" if interrupted else "cancelled"
        state.error = (
            "service shutdown; will resume from completed stages"
            if interrupted
            else "cancelled by user"
        )
        state.completed_at = None if interrupted else storage.now_iso()
        _persist(state)
        await _emit(queue, "error", {"job_id": state.job_id, "error": state.error})
    except Exception as exc:
        state.status = "cancelled" if state.status == "cancelling" else "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        state.completed_at = storage.now_iso()
        log.exception("pipeline failed for job %s", state.job_id)
        _persist(state)
        await _emit(queue, "error", {"job_id": state.job_id, "error": state.error})
    finally:
        renewal.cancel()
        await asyncio.gather(renewal, return_exceptions=True)
        await blocking_call(lambda: operations.refresh_usage(state.job_id))
        state_store.release(state.job_id, token)
        await _emit(queue, "stream_end", {})
        _cancel_signals.pop(state.job_id, None)
        _tasks.pop(state.job_id, None)
        _jobs.pop(state.job_id, None)
        _queues.pop(state.job_id, None)


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
    if settings.resolved_device == "cuda":
        # Measured on the XE7740 (RTX PRO 6000, 52 s clip, warm models,
        # 2026-10-03): transcribe 2.0 s (large-v3 fp16), translate 2.5 s
        # (NLLB 3.3B fp16, 9 segments), tts 14-21 s (XTTS / F5 / IndicF5),
        # mux ~4 s (NVENC; decode + drawtext dominate), vidstab 170 s.
        gpu = {
            "stabilize": max(3.0, src * 3.5) if src else None,
            "poststabilize": max(3.0, src * 3.5) if src else None,
            "transcribe": max(1.0, src / 25.0) if src else None,
            "translate": max(1.0, src / 20.0) if src else None,
            "tts": max(3.0, src * 0.35) if src else None,
            "mux": 4.0,
        }
        if stage_name in gpu:
            return gpu[stage_name]
    if stage_name == "stabilize":
        # vidstab two-pass on CPU: detect + transform at ~1-2× realtime
        # depending on resolution. Cap at 3 so the ETA doesn't feel
        # absurd for short clips.
        return max(3.0, src * 2.0) if src else None
    if stage_name == "poststabilize":
        # Same ffmpeg two-pass on the lipsynced output; same cost.
        return max(3.0, src * 2.0) if src else None
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
    queue: EventLog,
    state: JobState,
    stage_name: str,
):
    """Return a progress callback safe to call from any thread.

    The callback is *monotonic*: `cb(0.3)` followed by `cb(0.1)` leaves
    progress at 0.3. This matters because we now have two sources
    writing into the same channel — real progress from backends that
    stream it (wav2lip), and a time-based ticker from the orchestrator
    for backends that don't (musetalk, latentsync). The higher value
    wins; neither can drag the bar backwards.
    """
    stage = _stage(state, stage_name)
    last_emit = [0.0]  # last value we actually emitted to SSE
    max_seen = [0.0]  # max p observed across all calls

    def cb(p: float) -> None:
        p = max(0.0, min(1.0, float(p)))
        # Monotonic: never regress. The 1.0 "done" signal still
        # overwrites as a special case (it always passes the ≥ check).
        if p < max_seen[0]:
            return
        max_seen[0] = p
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
    queue: EventLog,
    name: str,
) -> StageResult:
    """Common prologue for every stage: mark running, publish ETA, emit event."""
    if state.status == "cancelling":
        raise asyncio.CancelledError()
    stage = _stage(state, name)
    state.current_stage = name
    stage.status = StageStatus.RUNNING
    stage.eta_seconds = estimate_eta_seconds(state, name)
    _persist(state)
    await _emit(
        queue,
        "stage_started",
        {
            "stage": name,
            "eta_seconds": stage.eta_seconds,
        },
    )
    return stage


async def _run_stage_audio(
    state: JobState,
    queue: EventLog,
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
        result = await blocking_call(_do)
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
        await _emit(
            queue,
            "stage_completed",
            {"stage": name, "output": result, "duration_ms": stage.duration_ms},
        )
        await _emit(
            queue,
            "pipeline_etas",
            {
                "source_duration_seconds": state.source_duration_seconds,
                "lipsync_backend": lipsync.backend_in_use(state.lipsync_backend),
                "etas": remaining_etas,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_stabilize(
    state: JobState,
    queue: EventLog,
    input_path: Path,
) -> Path:
    """Pre-stabilization pass. Returns the path subsequent stages should use.

    When disabled (the default), marks the stage SKIPPED and returns
    the input path unchanged. When enabled via per-request or env
    default, produces stabilized.mp4 and returns that path so
    downstream stages (transcribe / lipsync / mux) operate on the
    stable video.

    Stabilization failure (e.g. ffmpeg without vidstab *and* deshake
    paths both error) is logged and the stage is marked FAILED, but
    we still return the original input_path — the pipeline continues
    with the un-stabilized source rather than aborting the whole job.
    A shaky final video is usually better than no video.
    """
    from ..config import settings

    name = "stabilize"
    stage = _stage(state, name)

    # Resolve the toggle: per-request value takes priority; env default
    # is the fallback. Don't run if neither opts in.
    enable = state.enable_stabilization
    if enable is None:
        enable = settings.enable_video_stabilization

    if not enable:
        stage.status = StageStatus.SKIPPED
        stage.duration_ms = 0
        _persist(state)
        await _emit(queue, "stage_skipped", {"stage": name})
        return input_path

    stage = await _start_stage(state, queue, name)
    out_path = storage.job_artifact_path(state.job_id, "stabilized.mp4")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        return stabilize.stabilize_video(
            input_path=input_path,
            output_path=out_path,
            smoothing=settings.stabilize_smoothing,
            shakiness=settings.stabilize_shakiness,
        ).to_dict()

    try:
        result = await blocking_call(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = result
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": result,
                "duration_ms": stage.duration_ms,
            },
        )
        return out_path
    except Exception as e:
        # Don't abort the whole pipeline on stabilization failure —
        # shaky video is better than no video. Mark FAILED, log, and
        # fall through to the original input.
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        await _emit(
            queue,
            "stage_failed_soft",
            {
                "stage": name,
                "error": stage.error,
                "continuing_with": "original input",
            },
        )
        log.warning(
            "stabilize failed (%s); continuing with original input",
            stage.error,
        )
        return input_path


async def _run_stage_transcribe(state: JobState, queue: EventLog) -> None:
    name = "transcribe"
    stage = await _start_stage(state, queue, name)

    audio_path = storage.job_artifact_path(state.job_id, "audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "transcript.json")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        speech = storage.job_artifact_path(state.job_id, "vocals.wav")
        result = transcribe.transcribe(
            speech if speech.exists() else audio_path, out_path, language=state.source_language
        ).to_dict()
        if state.options.get("alignment", settings.enable_alignment) or state.options.get(
            "diarization", settings.enable_diarization
        ):
            from . import enhancements

            result = enhancements.analyze(
                audio_path,
                out_path,
                align=state.options.get("alignment", settings.enable_alignment),
                diarize=state.options.get("diarization", settings.enable_diarization),
            )
            import json

            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    try:
        result = await blocking_call(_do)
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
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": stage.output,
                "duration_ms": stage.duration_ms,
                "transcript": result,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_tts(state: JobState, queue: EventLog) -> None:
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
            options=state.options,
        ).to_dict()

    try:
        result = await blocking_call(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        translation["text"] = " ".join(
            s["text"] for s in translation.get("segments", [])
        ) or translation.get("text", "")
        translation_path.write_text(json.dumps(translation, ensure_ascii=False, indent=2))
        checkpoints.record(translation_path.parent, "translate")
        stage.output = {
            "backend": result["backend"],
            "language": result["language"],
            "path": out_path.name,
        }
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": stage.output,
                "duration_ms": stage.duration_ms,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_translate(state: JobState, queue: EventLog) -> None:
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
            backend_override=(
                settings.quality_translate_backend if state.mode == "quality" else None
            ),
            glossary=state.options.get("glossary"),
        ).to_dict()

    try:
        result = await blocking_call(_do)
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
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": stage.output,
                "duration_ms": stage.duration_ms,
                "translation": result,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise


async def _run_stage_lipsync(
    state: JobState,
    queue: EventLog,
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
        from . import windowed

        force = state.options.get("windowed_lipsync", settings.windowed_lipsync)
        plan = None
        if backend in ("musetalk", "latentsync"):
            budget = (
                settings.musetalk_frame_budget_mb
                if backend == "musetalk"
                else settings.latentsync_frame_budget_mb
            )
            plan = windowed.bounded_plan(
                input_path,
                audio_path,
                budget,
                settings.window_seconds,
                settings.window_overlap_seconds,
                force,
                fps=25 if backend == "latentsync" else None,
            )
        elif force and backend != "none":
            plan = (settings.window_seconds, settings.window_overlap_seconds)
        if plan:
            windowed.render(
                input_path,
                audio_path,
                out_path,
                lambda video, audio, out: lipsync.run(
                    backend, video, audio, out, quality_overrides=state.lipsync_quality
                ),
                size=plan[0],
                overlap=plan[1],
                progress=progress_cb,
                cancel=_cancel_signals.get(state.job_id),
                configuration={
                    "backend": backend,
                    "quality": state.lipsync_quality,
                    "provenance": state.provenance,
                },
            )
            return {"backend": backend, "passthrough": False}
        result = lipsync.run(
            backend=backend,
            video_in=input_path,
            audio_in=audio_path,
            output_path=out_path,
            progress=progress_cb,
            quality_overrides=state.lipsync_quality,
        )
        return result.to_dict()

    # Fallback progress ticker. musetalk and latentsync don't stream
    # real per-step progress from their services — without this, the
    # bar would sit at 0.05 (the initial "request started" tick) for
    # the entire 30+ minute LatentSync run. The ticker estimates from
    # elapsed vs eta and calls the monotonic progress_cb; real progress
    # from wav2lip still dominates when it flows (higher values win).
    # Path the lipsync service writes real per-phase progress to (when
    # it supports it — currently latentsync does, musetalk and wav2lip
    # don't). Ticker prefers real progress over time-based estimate.
    progress_file = storage.job_artifact_path(state.job_id, "latentsync_progress.json")
    # Clear any stale progress file from a previous run.
    try:
        progress_file.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass

    ticker = asyncio.create_task(
        _tick_lipsync_progress(
            progress_cb,
            stage.eta_seconds,
            started,
            progress_file=progress_file,
        )
    )

    try:
        result = await blocking_call(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "backend": result["backend"],
            "passthrough": result["passthrough"],
            "path": out_path.name,
        }
        stage.progress = 1.0
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": stage.output,
                "duration_ms": stage.duration_ms,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise
    finally:
        # Always tear the ticker down — on success, failure, or cancel.
        # It's an asyncio task on this loop, so awaiting its cancel is
        # cheap and guarantees we don't leak a background coroutine.
        ticker.cancel()
        try:
            await ticker
        except (asyncio.CancelledError, Exception):
            pass


async def _tick_lipsync_progress(
    progress_cb,
    eta_seconds: float | None,
    started_at: float,
    *,
    progress_file: Path | None = None,
    interval_s: float = 5.0,
    cap: float = 0.99,
) -> None:
    """Emit progress updates at regular intervals during lipsync.

    Priority order for the value reported:

      1. Real phase-level progress written by the service to
         `progress_file` (currently only latentsync does this). When
         present, the ticker forwards that value directly — the bar
         reflects actual work done, not "elapsed vs. guessed ETA".

      2. Time-based estimate: ``min(cap, elapsed / eta_seconds)``.
         Used when the service doesn't write progress (wav2lip,
         musetalk today) or when its first write hasn't happened yet.

    The ticker shortened its polling interval from 15 s to 5 s because
    real progress is much cheaper to read than to compute — just a
    JSON file read. And the `cap` raised from 0.98 to 0.99 to let real
    progress climb closer to the "done" transition before the ticker
    starts holding it back.

    Calls go through the same monotonic ``progress_cb`` that real
    backend progress uses, so wav2lip's in-process per-batch callbacks
    still dominate when they flow (higher values win; the callback
    enforces monotonicity).
    """
    import json as _json

    # No ETA and no progress file means we can't report anything
    # meaningful. Stay silent.
    if (not eta_seconds or eta_seconds <= 0) and progress_file is None:
        return

    try:
        while True:
            await asyncio.sleep(interval_s)

            # 1) Prefer real progress from the service.
            real_pct: float | None = None
            if progress_file is not None and progress_file.exists():
                try:
                    data = _json.loads(progress_file.read_text())
                    real_pct = float(data.get("percent"))
                except Exception:
                    real_pct = None

            if real_pct is not None:
                # Cap at `cap` so the true "done" transition (1.0)
                # only fires from the completion handler, not mid-run.
                progress_cb(min(cap, real_pct))
                continue

            # 2) Fall back to time-based estimate.
            if eta_seconds and eta_seconds > 0:
                elapsed = time.perf_counter() - started_at
                p = min(cap, elapsed / eta_seconds)
                progress_cb(p)
    except asyncio.CancelledError:
        # Normal termination when the stage completes.
        pass


async def _run_stage_poststabilize(
    state: JobState,
    queue: EventLog,
) -> None:
    """Optional post-stabilization of the lipsynced video.

    Runs vidstab (or deshake fallback) on lipsynced.mp4 and writes
    lipsynced_stable.mp4. The mux stage prefers `lipsynced_stable.mp4`
    when it exists, falling back to `lipsynced.mp4` otherwise —
    meaning this stage being SKIPPED or FAILED is safe and transparent.

    Use case: the lipsync pipeline introduces its own per-frame jitter
    (affine smoothing helps but can't eliminate it entirely; VAE
    precision can shift pixel content; etc). Stabilizing the output
    directly is the catch-all that doesn't care what caused the
    jitter. Tradeoff is the usual vidstab artifact risk — floaty feel
    on intentional motion, brief warping on fast motion. For
    stationary-subject clips (most of this workflow's use case),
    downside is invisible.
    """
    from ..config import settings

    name = "poststabilize"
    stage = _stage(state, name)

    # Per-request flag wins; env default is the fallback.
    enable = state.enable_output_stabilization
    if enable is None:
        enable = settings.enable_output_stabilization

    # Skip if disabled, or if lipsync didn't produce a file (passthrough
    # path, or lipsync=none — stabilizing the untouched source via this
    # stage provides no value over the pre-stab stage).
    lipsynced = storage.job_artifact_path(state.job_id, "lipsynced.mp4")
    if not enable or not lipsynced.exists():
        stage.status = StageStatus.SKIPPED
        stage.duration_ms = 0
        _persist(state)
        await _emit(queue, "stage_skipped", {"stage": name})
        return

    stage = await _start_stage(state, queue, name)
    out_path = storage.job_artifact_path(state.job_id, "lipsynced_stable.mp4")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        return stabilize.stabilize_video(
            input_path=lipsynced,
            output_path=out_path,
            smoothing=settings.stabilize_smoothing,
            shakiness=settings.stabilize_shakiness,
        ).to_dict()

    try:
        result = await blocking_call(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = result
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": result,
                "duration_ms": stage.duration_ms,
            },
        )
    except Exception as e:
        # Soft-fail: same philosophy as the pre-stabilize stage. If
        # post-stab fails we ship the un-post-stabilized lipsync
        # output rather than killing the job.
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        await _emit(
            queue,
            "stage_failed_soft",
            {
                "stage": name,
                "error": stage.error,
                "continuing_with": "un-stabilized lipsync output",
            },
        )
        log.warning(
            "poststabilize failed (%s); mux will use lipsynced.mp4",
            stage.error,
        )


async def _run_stage_mux(
    state: JobState,
    queue: EventLog,
    input_path: Path,
) -> None:
    name = "mux"
    stage = await _start_stage(state, queue, name)

    # Video input preference: post-stabilized if present, then plain
    # lipsynced if present, else the original upload. This chain
    # ensures a failed/skipped stabilize stage is transparent —
    # mux just falls back to the next-best artifact.
    lipsynced_stable = storage.job_artifact_path(state.job_id, "lipsynced_stable.mp4")
    lipsynced = storage.job_artifact_path(state.job_id, "lipsynced.mp4")
    if lipsynced_stable.exists():
        video_in = lipsynced_stable
    elif lipsynced.exists():
        video_in = lipsynced
    else:
        video_in = input_path
    audio_in = storage.job_artifact_path(state.job_id, "translated_audio.wav")
    out_path = storage.job_artifact_path(state.job_id, "final.mp4")
    started = time.perf_counter()

    def _do() -> dict[str, Any]:
        final_audio = audio_in
        background = storage.job_artifact_path(state.job_id, "background.wav")
        if (
            state.options.get("background_audio", settings.enable_background_audio)
        ) and background.exists():
            from . import enhancements

            final_audio = storage.job_artifact_path(state.job_id, "mixed_audio.wav")
            enhancements.remix(
                audio_in,
                background,
                final_audio,
                state.options.get("background_gain", settings.background_gain),
            )
        return watermark.mux_and_watermark(
            video_path=video_in,
            audio_path=final_audio,
            output_path=out_path,
        ).to_dict()

    try:
        result = await blocking_call(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = {
            "path": out_path.name,
            "watermark": result["watermark"],
            "source_video": video_in.name,
        }
        stage.progress = 1.0
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(
            queue,
            "stage_completed",
            {
                "stage": name,
                "output": stage.output,
                "duration_ms": stage.duration_ms,
            },
        )
    except Exception as e:
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        raise
