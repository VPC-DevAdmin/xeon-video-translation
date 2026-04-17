"""Local-filesystem job artifact store.

Each job gets a directory: {JOB_ARTIFACTS_DIR}/{job_id}/
- input.<ext>          original upload
- audio.wav            stage 1 output
- transcript.json      stage 2 output
- translation.json     stage 3 output
- translated_audio.wav stage 4 output (future)
- lipsynced.mp4        stage 5 output (future)
- final.mp4            stage 6 output (future)
- meta.json            job metadata + stage results
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import settings


def new_job_id() -> str:
    return uuid4().hex


def job_dir(job_id: str) -> Path:
    d = settings.job_artifacts_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_artifact_path(job_id: str, name: str) -> Path:
    # Refuse path traversal: artifact names must be plain filenames.
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid artifact name: {name!r}")
    return job_dir(job_id) / name


def write_meta(job_id: str, meta: dict[str, Any]) -> None:
    path = job_dir(job_id) / "meta.json"
    payload = json.dumps(meta, indent=2, default=_json_default)
    path.write_text(payload, encoding="utf-8")


def read_meta(job_id: str) -> dict[str, Any] | None:
    path = job_dir(job_id) / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
