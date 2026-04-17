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
):
    """Call the MuseTalk service and wait for the lipsynced MP4.

    Returns a LipsyncResult on success, raises LipsyncError otherwise.
    """
    # Imported here to avoid a circular import with the dispatcher.
    from ..lipsync import LipsyncError, LipsyncResult

    if progress is not None:
        # Indeterminate while we wait — real progress lands in PR 1c when the
        # service streams per-frame updates.
        progress(0.05)

    body = json.dumps({
        "video_path": str(video_in),
        "audio_path": str(audio_in),
        "output_path": str(output_path),
    }).encode("utf-8")

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
        # PR 1a / 1b return this — the scaffold is up but inference isn't wired.
        msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
        raise LipsyncError(
            f"MuseTalk service is not yet able to run inference: {msg}. "
            f"See docs/lipsync.md for the PR roadmap. Fall back to "
            f"LIPSYNC_BACKEND=wav2lip or none for now."
        )
    if e.code == 400:
        raise LipsyncError(f"MuseTalk rejected request: {detail}")
    raise LipsyncError(f"MuseTalk failed (HTTP {e.code}): {detail}")
