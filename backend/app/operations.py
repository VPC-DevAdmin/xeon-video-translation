"""Local resource accounting, retention and model provenance."""

import os
import shutil
import time
from .config import settings
from . import state_store, storage

_worker_lock = None


def acquire_dispatcher():
    global _worker_lock
    import fcntl

    handle = (settings.job_artifacts_dir / ".dispatcher.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("only one translation dispatcher may own this jobs volume")
    _worker_lock = handle


def release_dispatcher():
    global _worker_lock
    if _worker_lock:
        _worker_lock.close()
        _worker_lock = None


def refresh_usage(identifier):
    """Reconcile one job at stage boundaries; never walk history during admission."""
    total = 0
    for path in storage.job_dir(identifier).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # atomic output replacement or concurrent cleanup
    state_store.set_usage(identifier, total)
    return total


def reconcile_usage(active_only=False):
    identifiers = (
        state_store.active_job_ids()
        if active_only
        else [p.name for p in settings.job_artifacts_dir.iterdir() if p.is_dir()]
    )
    for identifier in identifiers:
        try:
            if storage.read_meta(identifier):
                refresh_usage(identifier)
        except (ValueError, OSError):
            continue


def usage(owner):
    return state_store.usage(owner)


def check_capacity(owner):
    from fastapi import HTTPException

    used = usage(owner)
    if used["active"] >= settings.max_user_jobs:
        raise HTTPException(429, "account job capacity reached")
    if used["bytes"] >= settings.max_user_storage_mb * 1024**2:
        raise HTTPException(413, "account storage quota reached")
    if shutil.disk_usage(settings.job_artifacts_dir).free < settings.min_free_disk_mb * 1024**2:
        raise HTTPException(507, "server disk capacity reached")


def expire_jobs():
    from datetime import datetime

    if not settings.retention_days:
        return
    cutoff = time.time() - settings.retention_days * 86400
    for job in state_store.list_jobs(None, 100000):
        if job["status"] not in state_store.TERMINAL or not job.get("completed_at"):
            continue
        if datetime.fromisoformat(job["completed_at"]).timestamp() < cutoff:
            shutil.rmtree(storage.job_dir(job["job_id"]), ignore_errors=True)
            state_store.remove(job["job_id"])


def provenance():
    # Only allow-listed, nonsecret configuration belongs in public job metadata.
    keys = (
        "model_revision",
        "device",
        "whisper_model",
        "whisper_compute_type",
        "nllb_model",
        "translate_backend",
        "quality_translate_backend",
        "tts_backend",
        "tts_max_speed",
        "window_seconds",
        "window_overlap_seconds",
        "video_encoder",
    )
    return {
        "configuration": {k: getattr(settings, k) for k in keys},
        "release": os.getenv("RELEASE_ID", "local-unreleased"),
        "models_manifest": os.getenv("MODEL_MANIFEST_SHA256", "unverified"),
    }


def resource_sample():
    """Process peak RSS and host GPU sample; no model loading on diagnostics."""
    import resource
    import subprocess
    import sys

    result = {
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "disk_free_bytes": shutil.disk_usage(settings.job_artifacts_dir).free,
        "gpus": [],
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            output = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            for line in output.stdout.splitlines():
                index, used, total, utilization = map(int, line.split(","))
                result["gpus"].append(
                    dict(index=index, used_mb=used, total_mb=total, utilization_percent=utilization)
                )
        except (subprocess.SubprocessError, ValueError):
            result["gpu_sample_error"] = True
    return result
