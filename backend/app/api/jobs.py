"""HTTP endpoints for job submission, status, and artifact download."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    Depends,
    Request,
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from .. import storage, state_store, operations
from ..security import principal, check_owner
from ..config import settings
from ..modes import MODES
from ..pipeline.orchestrator import (
    JobState,
    cancel_job,
    get_job,
    register_job,
    run_pipeline,
    blocking_call,
)


def valid_job_path(request: Request):
    value = request.path_params.get("job_id")
    if value:
        try:
            storage.job_dir(value)
            meta = storage.read_meta(value)
            if meta is None:
                raise HTTPException(404, "job not found")
            check_owner(meta)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(valid_job_path)])


_uploading: set[str] = set()

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


_ALLOWED_LIPSYNC = {"none", "wav2lip", "musetalk", "latentsync"}


_ALLOWED_BLEND_MODES = {"raw", "jaw", "mouth", "neck"}
_ALLOWED_FACE_RESTORE = {"codeformer", "none"}
_ALLOWED_TTS_BACKENDS = {"xtts", "f5tts", "indicf5", "auto"}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_job(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None),
    lipsync_backend: str | None = Form(None),
    mode: str | None = Form(None),
    request_id: str | None = Form(None),
    options_json: str = Form("{}"),
    # Per-request TTS backend (xtts | f5tts). Omit → env default (xtts).
    tts_backend: str | None = Form(None),
    # Per-request pre-stabilization toggle. Omit → env default
    # (ENABLE_VIDEO_STABILIZATION, currently False). Accepts truthy
    # strings ("1", "true", "yes") or falsy strings.
    enable_stabilization: str | None = Form(None),
    # Per-request post-stabilization toggle. Stabilizes the lipsynced
    # output before mux. Independent of enable_stabilization; they can
    # stack. Omit → env default (ENABLE_OUTPUT_STABILIZATION).
    enable_output_stabilization: str | None = Form(None),
    # Per-request musetalk knobs (forwarded to the lipsync service).
    # Each is optional; missing fields fall through to service env defaults.
    musetalk_blend_mode: str | None = Form(None),
    musetalk_blend_feather: float | None = Form(None),
    musetalk_face_restore: str | None = Form(None),
    musetalk_face_restore_fidelity: float | None = Form(None),
    musetalk_face_restore_blend: float | None = Form(None),
) -> dict:
    """Accept a video upload, persist it, and kick off the pipeline."""
    from ..options import from_request

    try:
        options = from_request(options_json)
    except ValueError as exc:
        raise HTTPException(400, "invalid job options") from exc
    if not settings.recover_jobs:
        raise HTTPException(503, "this service is reserved for avatar speech")
    if mode is not None and mode not in MODES:
        raise HTTPException(400, "unsupported mode")
    preset = MODES.get(mode, {})
    lipsync_backend = lipsync_backend or preset.get("lipsync_backend")
    tts_backend = tts_backend or preset.get("tts_backend")
    musetalk_face_restore = musetalk_face_restore or preset.get("musetalk_face_restore")
    musetalk_blend_mode = musetalk_blend_mode or preset.get("musetalk_blend_mode")
    if enable_output_stabilization is None:
        enable_output_stabilization = preset.get("enable_output_stabilization")
    from ..pipeline.translate import NLLB_LANG_CODES

    if target_language.lower() not in NLLB_LANG_CODES:
        raise HTTPException(400, "unsupported target language")
    if request_id:
        try:
            existing = storage.read_meta(request_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if existing:
            check_owner(existing)
            return {k: existing[k] for k in ("job_id", "status", "created_at")}
    await blocking_call(lambda: operations.check_capacity(principal.get()))
    from ..pipeline.orchestrator import pending_count

    if pending_count() + len(_uploading) >= settings.max_pending_jobs:
        raise HTTPException(429, "job queue is full; retry later")
    filename = video.filename or "input"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"unsupported file extension: {ext!r}")

    lipsync_backend_norm: str | None = None
    if lipsync_backend:
        lipsync_backend_norm = lipsync_backend.lower().strip()
        if lipsync_backend_norm not in _ALLOWED_LIPSYNC:
            raise HTTPException(
                400,
                f"unsupported lipsync_backend: {lipsync_backend!r}. "
                f"Allowed: {sorted(_ALLOWED_LIPSYNC)}",
            )

    # Normalize + validate the per-request musetalk knobs. Invalid input
    # raises 400; None stays None and the service falls back to env.
    def _norm_enum(val: str | None, allowed: set, label: str) -> str | None:
        if val is None:
            return None
        v = val.lower().strip()
        if v not in allowed:
            raise HTTPException(
                400,
                f"unsupported {label}: {val!r}. Allowed: {sorted(allowed)}",
            )
        return v

    def _norm_ratio(val: float | None, lo: float, hi: float, label: str) -> float | None:
        if val is None:
            return None
        if not lo <= val <= hi:
            raise HTTPException(
                400,
                f"{label}={val} out of range [{lo}, {hi}]",
            )
        return val

    lipsync_quality: dict | None = None
    q = {
        "blend_mode": _norm_enum(musetalk_blend_mode, _ALLOWED_BLEND_MODES, "musetalk_blend_mode"),
        "blend_feather": _norm_ratio(musetalk_blend_feather, 0.02, 0.30, "musetalk_blend_feather"),
        "face_restore": _norm_enum(
            musetalk_face_restore, _ALLOWED_FACE_RESTORE, "musetalk_face_restore"
        ),
        "face_restore_fidelity": _norm_ratio(
            musetalk_face_restore_fidelity, 0.0, 1.0, "musetalk_face_restore_fidelity"
        ),
        "face_restore_blend": _norm_ratio(
            musetalk_face_restore_blend, 0.0, 1.0, "musetalk_face_restore_blend"
        ),
    }
    if any(v is not None for v in q.values()):
        lipsync_quality = {k: v for k, v in q.items() if v is not None}

    tts_backend_norm = _norm_enum(tts_backend, _ALLOWED_TTS_BACKENDS, "tts_backend")

    # Normalize boolean-typed form fields. None → None (falls through
    # to settings.<field> defaults in the orchestrator). Explicit
    # "true"/"false"/"1"/"0" maps as expected.
    def _norm_bool(val: str | None, field: str) -> bool | None:
        if val is None:
            return None
        v = val.lower().strip()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off", ""):
            return False
        raise HTTPException(400, f"invalid boolean for {field}: {val!r}")

    enable_stabilization_norm = _norm_bool(enable_stabilization, "enable_stabilization")
    enable_output_stabilization_norm = _norm_bool(
        enable_output_stabilization,
        "enable_output_stabilization",
    )

    job_id = request_id or storage.new_job_id()
    job_directory = storage.job_dir(job_id)
    try:
        job_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise HTTPException(409, "submission in progress; retry with the same request_id")
    _uploading.add(job_id)
    input_path = job_directory / f"input{ext}"

    # Stream the upload to disk and check size as we go.
    max_bytes = settings.max_video_size_mb * 1024 * 1024
    written = 0
    try:
        with input_path.open("wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    input_path.unlink(missing_ok=True)
                    shutil.rmtree(job_directory, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"upload exceeds {settings.max_video_size_mb} MB",
                    )
                f.write(chunk)
        if not written:
            raise HTTPException(422, "empty video upload")
    except BaseException:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    finally:
        _uploading.discard(job_id)

    state = JobState(
        job_id=job_id,
        owner_id=principal.get(),
        options=options,
        target_language=target_language.lower(),
        source_language=source_language.lower() if source_language else None,
        lipsync_backend=lipsync_backend_norm,
        lipsync_quality=lipsync_quality,
        tts_backend=tts_backend_norm,
        enable_stabilization=enable_stabilization_norm,
        enable_output_stabilization=enable_output_stabilization_norm,
        input_filename=filename,
        mode=mode,
    )
    register_job(state)
    await blocking_call(lambda: operations.refresh_usage(state.job_id))
    storage.write_meta(job_id, state.to_dict())

    # Run the pipeline as a background task on the same event loop.
    background.add_task(_kickoff, state, input_path)

    return {
        "job_id": job_id,
        "status": state.status,
        "created_at": state.created_at,
    }


async def _kickoff(state: JobState, input_path: Path) -> None:
    # Wrap so any unexpected exception still gets logged.
    try:
        await run_pipeline(state, input_path)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("pipeline kickoff failed")


@router.get("")
async def list_jobs(limit: int = 20) -> dict:
    """List recent jobs, newest first.

    Reads meta.json from each directory under JOB_ARTIFACTS_DIR. Good enough
    for a single-machine demo; we're not pretending this scales.
    """
    limit = max(1, min(200, limit))
    jobs = state_store.list_jobs(principal.get(), limit)
    queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["created_at"])
    positions = {j["job_id"]: i + 1 for i, j in enumerate(queued)}
    for job in jobs:
        job["queue_position"] = positions.get(job["job_id"])
    return {"jobs": jobs}


@router.get("/{job_id}")
async def get_job_status(job_id: str) -> dict:
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, "job not found")
    payload = state.to_dict()
    if state.status == "completed":
        for name in ("final.mp4", "translated_audio.wav", "translation.json"):
            p = storage.job_artifact_path(job_id, name)
            if p.exists():
                payload["result_url"] = f"/jobs/{job_id}/artifacts/{name}"
                break
    return payload


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel(job_id: str) -> dict:
    """Request cancellation of an in-flight pipeline.

    Queued work cancels immediately. Running work enters cancelling and
    retains its resource lease until the current blocking inference exits.

    Returns 404 if the job_id is unknown or the job was never running
    in this process. 409 if the job is already in a terminal state.
    """
    # 404 first — get_job returns a reconstructed state for completed
    # jobs, so we check that before cancel_job which only knows about
    # in-process state.
    if get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    ok, reason = cancel_job(job_id)
    if not ok:
        if reason == "already-terminal":
            raise HTTPException(409, "job is already in a terminal state")
        # "unknown" means the job isn't in the live task registry —
        # probably completed/failed before cancel reached us, or was
        # reconstructed from disk for an old job_id.
        raise HTTPException(404, "job is not currently running")
    return {"job_id": job_id, "status": reason}


@router.get("/{job_id}/artifacts")
async def list_artifacts(job_id: str) -> dict:
    """List all files written to this job's directory."""
    d = storage.job_dir(job_id)
    if not d.exists():
        raise HTTPException(404, "job not found")
    artifacts = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        artifacts.append(
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "url": f"/jobs/{job_id}/artifacts/{p.name}",
            }
        )
    return {"job_id": job_id, "artifacts": artifacts}


@router.get("/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str):
    try:
        path = storage.job_artifact_path(job_id, name)
    except ValueError:
        raise HTTPException(400, "invalid artifact name")
    if not path.exists():
        raise HTTPException(404, "artifact not found")
    return FileResponse(path, filename=name)
