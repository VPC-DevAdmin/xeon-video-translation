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
import re
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import settings


def new_job_id() -> str:
    return uuid4().hex


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("invalid job id")
    return settings.job_artifacts_dir / job_id


def job_artifact_path(job_id: str, name: str) -> Path:
    # Refuse path traversal: artifact names must be plain filenames.
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid artifact name: {name!r}")
    return job_dir(job_id) / name


def write_meta(job_id: str, meta: dict[str, Any]) -> None:
    path = job_dir(job_id) / "meta.json"
    from .state_store import save
    save(job_id, meta)
    payload = json.dumps(meta, indent=2, default=_json_default)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".meta-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_meta(job_id: str) -> dict[str, Any] | None:
    job_dir(job_id)  # validate before querying the registry
    from .state_store import read, save
    found = read(job_id)
    if found is not None:
        return found
    path = job_dir(job_id) / "meta.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    save(job_id, meta)
    return meta


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
