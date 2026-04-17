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


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_job(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None),
    lipsync_backend: str | None = Form(None),
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


@router.get("/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str):
    try:
        path = storage.job_artifact_path(job_id, name)
    except ValueError:
        raise HTTPException(400, "invalid artifact name")
    if not path.exists():
        raise HTTPException(404, "artifact not found")
    return FileResponse(path, filename=name)
