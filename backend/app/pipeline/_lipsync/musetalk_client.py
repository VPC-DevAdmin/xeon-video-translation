"""HTTP client for the MuseTalk lipsync microservice.

The service runs in its own container with a conflicting dep stack (MuseTalk
upstream pins `transformers==4.39.2`). We invoke it via HTTP and pass file
paths — not bytes — because both services mount the same `/jobs` volume.

PR 1a: the service returns 501 with a structured body. We translate that into
a clear LipsyncError pointing the user at docs/lipsync.md.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from ...config import settings

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def run(
    video_in: Path,
    audio_in: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
    quality_overrides: dict | None = None,
):
    """Call the MuseTalk service and wait for the lipsynced MP4.

    Returns a LipsyncResult on success, raises LipsyncError otherwise.

    `quality_overrides`, if present, gets merged into the JSON body so the
    service can vary blend mode, face restore, etc. per request. Missing
    keys fall through to the service's env-driven defaults.
    """
    # Imported here to avoid a circular import with the dispatcher.
    from ..lipsync import LipsyncError, LipsyncResult

    if progress is not None:
        # Indeterminate while we wait — real progress lands in PR 1c when the
        # service streams per-frame updates.
        progress(0.05)

    payload: dict = {
        "video_path": str(video_in),
        "audio_path": str(audio_in),
        "output_path": str(output_path),
    }
    if quality_overrides:
        # Only forward keys the service understands. Drops `None`s so they
        # don't override existing env defaults on the service side.
        for key in (
            "blend_mode", "blend_feather",
            "face_restore", "face_restore_fidelity", "face_restore_blend",
        ):
            val = quality_overrides.get(key)
            if val is not None:
                payload[key] = val
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.musetalk_service_url.rstrip('/')}/lipsync",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=settings.musetalk_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_from_http_error(e, LipsyncError)
    except urllib.error.URLError as e:
        raise LipsyncError(
            f"MuseTalk service unreachable at {settings.musetalk_service_url}: "
            f"{e.reason}. Is the lipsync-musetalk container running?"
        ) from e

    if progress is not None:
        progress(1.0)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise LipsyncError(
            f"MuseTalk returned 200 but no output file was written at {output_path}."
        )

    return LipsyncResult(
        backend="musetalk",
        output_path=output_path.name,
        passthrough=False,
    )


def _raise_from_http_error(e: urllib.error.HTTPError, LipsyncError: type) -> None:
    """Translate structured service errors into a clean LipsyncError."""
    try:
        body = json.loads(e.read().decode("utf-8", errors="replace"))
        detail = body.get("detail", body)
    except Exception:
        detail = e.reason

    if e.code == 501:
        # PR 1a / 1b legacy — should not appear after PR 1c.
        msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
        raise LipsyncError(
            f"MuseTalk service not implemented: {msg}. See docs/lipsync.md."
        )
    if e.code == 503:
        # Missing weights.
        fix = detail.get("fix") if isinstance(detail, dict) else None
        raise LipsyncError(
            f"MuseTalk weights missing: {detail}. "
            f"{f'Fix: {fix}' if fix else 'See docs/lipsync.md.'}"
        )
    if e.code == 422:
        # Input-level problem (no face detected, corrupt video, …).
        err = detail.get("error", detail) if isinstance(detail, dict) else detail
        raise LipsyncError(f"MuseTalk couldn't process the clip: {err}")
    if e.code == 400:
        raise LipsyncError(f"MuseTalk rejected request: {detail}")
    if e.code == 500:
        err = detail.get("error", detail) if isinstance(detail, dict) else detail
        raise LipsyncError(f"MuseTalk crashed: {err}. See service logs.")
    raise LipsyncError(f"MuseTalk failed (HTTP {e.code}): {detail}")
