"""HTTP endpoints for job submission, status, and artifact download."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .. import storage
from ..config import settings
from ..pipeline.orchestrator import JobState, get_job, register_job, run_pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])


_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


_ALLOWED_LIPSYNC = {"none", "wav2lip", "musetalk", "latentsync"}


_ALLOWED_BLEND_MODES = {"raw", "jaw", "mouth", "neck"}
_ALLOWED_FACE_RESTORE = {"codeformer", "none"}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_job(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None),
    lipsync_backend: str | None = Form(None),
    # Per-request musetalk knobs (forwarded to the lipsync service).
    # Each is optional; missing fields fall through to service env defaults.
    musetalk_blend_mode: str | None = Form(None),
    musetalk_blend_feather: float | None = Form(None),
    musetalk_face_restore: str | None = Form(None),
    musetalk_face_restore_fidelity: float | None = Form(None),
    musetalk_face_restore_blend: float | None = Form(None),
) -> dict:
    """Accept a video upload, persist it, and kick off the pipeline."""
    filename = video.filename or "input"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"unsupported file extension: {ext!r}")

    lipsync_backend_norm: str | None = None
    if lipsync_backend:
        lipsync_backend_norm = lipsync_backend.lower().strip()
        if lipsync_backend_norm not in _ALLOWED_LIPSYNC:
            raise HTTPException(
                400, f"unsupported lipsync_backend: {lipsync_backend!r}. "
                     f"Allowed: {sorted(_ALLOWED_LIPSYNC)}"
            )

    # Normalize + validate the per-request musetalk knobs. Invalid input
    # raises 400; None stays None and the service falls back to env.
    def _norm_enum(val: str | None, allowed: set, label: str) -> str | None:
        if val is None:
            return None
        v = val.lower().strip()
        if v not in allowed:
            raise HTTPException(
                400, f"unsupported {label}: {val!r}. Allowed: {sorted(allowed)}",
            )
        return v

    def _norm_ratio(val: float | None, lo: float, hi: float, label: str) -> float | None:
        if val is None:
            return None
        if not lo <= val <= hi:
            raise HTTPException(
                400, f"{label}={val} out of range [{lo}, {hi}]",
            )
        return val

    lipsync_quality: dict | None = None
    q = {
        "blend_mode": _norm_enum(musetalk_blend_mode, _ALLOWED_BLEND_MODES, "musetalk_blend_mode"),
        "blend_feather": _norm_ratio(musetalk_blend_feather, 0.02, 0.30, "musetalk_blend_feather"),
        "face_restore": _norm_enum(musetalk_face_restore, _ALLOWED_FACE_RESTORE, "musetalk_face_restore"),
        "face_restore_fidelity": _norm_ratio(musetalk_face_restore_fidelity, 0.0, 1.0, "musetalk_face_restore_fidelity"),
        "face_restore_blend": _norm_ratio(musetalk_face_restore_blend, 0.0, 1.0, "musetalk_face_restore_blend"),
    }
    if any(v is not None for v in q.values()):
        lipsync_quality = {k: v for k, v in q.items() if v is not None}

    job_id = storage.new_job_id()
    job_directory = storage.job_dir(job_id)
    input_path = job_directory / f"input{ext}"

    # Stream the upload to disk and check size as we go.
    max_bytes = settings.max_video_size_mb * 1024 * 1024
    written = 0
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

    state = JobState(
        job_id=job_id,
        target_language=target_language.lower(),
        source_language=source_language.lower() if source_language else None,
        lipsync_backend=lipsync_backend_norm,
        lipsync_quality=lipsync_quality,
        input_filename=filename,
    )
    register_job(state)
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
    limit = max(1, min(200, int(limit)))
    jobs: list[dict] = []
    d = settings.job_artifacts_dir
    if d.exists():
        dirs = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for job_dir in dirs:
            if not job_dir.is_dir():
                continue
            meta = storage.read_meta(job_dir.name)
            if not meta:
                continue
            jobs.append({
                "job_id": job_dir.name,
                "status": meta.get("status"),
                "current_stage": meta.get("current_stage"),
                "created_at": meta.get("created_at"),
                "completed_at": meta.get("completed_at"),
                "input_filename": meta.get("input_filename"),
                "target_language": meta.get("target_language"),
                "lipsync_backend": meta.get("lipsync_backend"),
            })
            if len(jobs) >= limit:
                break
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
        artifacts.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "url": f"/jobs/{job_id}/artifacts/{p.name}",
        })
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
