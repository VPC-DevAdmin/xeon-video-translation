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
    # Per-job pre-stabilization toggle. True runs vidstab (or deshake
    # fallback) before transcribe/lipsync see the video. None means
    # "use settings.enable_video_stabilization" (env default, False).
    enable_stabilization: bool | None = None
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
            "enable_stabilization": self.enable_stabilization,
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
STAGE_NAMES = [
    "audio",
    "stabilize",   # optional; SKIPPED when not requested
    "transcribe",
    "translate",
    "tts",
    "lipsync",
    "mux",
]


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
            # Optional pre-stabilization. When enabled, produces
            # stabilized.mp4 and subsequent stages use it as the video
            # input. When disabled, the stage is marked SKIPPED and
            # input_path is untouched.
            input_path = await _run_stage_stabilize(state, queue, input_path)
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
    if stage_name == "stabilize":
        # vidstab two-pass on CPU: detect + transform at ~1-2× realtime
        # depending on resolution. Cap at 3 so the ETA doesn't feel
        # absurd for short clips.
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
    queue: asyncio.Queue[dict[str, Any]],
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
    max_seen = [0.0]   # max p observed across all calls

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


async def _run_stage_stabilize(
    state: JobState,
    queue: asyncio.Queue[dict[str, Any]],
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
        result = await asyncio.to_thread(_do)
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        stage.output = result
        stage.status = StageStatus.DONE
        _persist(state)
        await _emit(queue, "stage_completed", {
            "stage": name, "output": result,
            "duration_ms": stage.duration_ms,
        })
        return out_path
    except Exception as e:
        # Don't abort the whole pipeline on stabilization failure —
        # shaky video is better than no video. Mark FAILED, log, and
        # fall through to the original input.
        stage.status = StageStatus.FAILED
        stage.error = f"{type(e).__name__}: {e}"
        stage.duration_ms = int((time.perf_counter() - started) * 1000)
        _persist(state)
        await _emit(queue, "stage_failed_soft", {
            "stage": name, "error": stage.error,
            "continuing_with": "original input",
        })
        log.warning(
            "stabilize failed (%s); continuing with original input",
            stage.error,
        )
        return input_path


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

    # Fallback progress ticker. musetalk and latentsync don't stream
    # real per-step progress from their services — without this, the
    # bar would sit at 0.05 (the initial "request started" tick) for
    # the entire 30+ minute LatentSync run. The ticker estimates from
    # elapsed vs eta and calls the monotonic progress_cb; real progress
    # from wav2lip still dominates when it flows (higher values win).
    ticker = asyncio.create_task(
        _tick_lipsync_progress(progress_cb, stage.eta_seconds, started)
    )

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
    interval_s: float = 15.0,
    cap: float = 0.98,
) -> None:
    """Emit time-based progress estimates every `interval_s` seconds.

    This is a fallback for lipsync backends that don't stream real
    per-step progress (musetalk, latentsync). Without it, the bar sits
    at 0.05 for the full run and users think the job is stuck.

    The estimate is naive: ``min(cap, elapsed / eta_seconds)``. It caps
    at `cap` (default 0.98) so the final 2% is reserved for the actual
    completion signal — users see a clear "done" transition at the end
    rather than the bar silently sitting at 100% for a beat.

    Calls go through the same monotonic ``progress_cb`` that real
    backend progress uses, so if a backend *does* stream (wav2lip) the
    higher real value wins and this ticker becomes a no-op for that
    run. No flag or backend-specific branching needed.

    No-op when we don't have an ETA. Better to show nothing than to
    report fake progress against an unknown denominator.
    """
    if not eta_seconds or eta_seconds <= 0:
        return
    try:
        while True:
            await asyncio.sleep(interval_s)
            elapsed = time.perf_counter() - started_at
            p = min(cap, elapsed / eta_seconds)
            progress_cb(p)
    except asyncio.CancelledError:
        # Normal termination when the stage completes. Swallow so the
        # finally-block in the caller doesn't have to special-case it.
        pass


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
