"""HTTP client for the LatentSync lipsync microservice.

The service runs in its own container with a conflicting dep stack
(LatentSync's diffusers/transformers pinning doesn't match either the
main backend or lipsync-musetalk). We invoke it via HTTP and pass file
paths — not bytes — because both services mount the same `/jobs` volume.

PR-LS-1a (current): the service returns 501 with a structured body. We
translate that into a clear LipsyncError pointing the user at
docs/lipsync.md. The rest of this client is already shaped for the
eventual 1c response so only the server-side flip is needed later.

Mirrors backend/app/pipeline/_lipsync/musetalk_client.py; the two stay
shaped the same on purpose — factoring to a shared base class is cheap
once we have a third consumer, overkill for two.
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
    """Call the LatentSync service and wait for the lipsynced MP4.

    Returns a LipsyncResult on success, raises LipsyncError otherwise.

    `quality_overrides`, if present, is forwarded selectively — only the
    keys the service recognizes make it into the payload so callers don't
    accidentally poison the request with MuseTalk-specific settings.
    """
    # Imported here to avoid a circular import with the dispatcher.
    from ..lipsync import LipsyncError, LipsyncResult

    if progress is not None:
        # Indeterminate while we wait — real progress lands in PR-LS-1c
        # when the service streams per-frame updates.
        progress(0.05)

    payload: dict = {
        "video_path": str(video_in),
        "audio_path": str(audio_in),
        "output_path": str(output_path),
    }
    if quality_overrides:
        # LatentSync-specific per-request knobs. The server validates
        # ranges; we just forward the value when present. Missing keys
        # fall through to the service's env-driven defaults
        # (LATENTSYNC_STEPS / LATENTSYNC_GUIDANCE).
        for key in ("num_inference_steps", "guidance_scale", "seed"):
            val = quality_overrides.get(key)
            if val is not None:
                payload[key] = val
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.latentsync_service_url.rstrip('/')}/lipsync",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=settings.latentsync_timeout_seconds,
        ) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
            log.info("latentsync responded: %s", resp_json.get("status"))
    except urllib.error.HTTPError as e:
        _raise_from_http_error(e, LipsyncError)
    except urllib.error.URLError as e:
        raise LipsyncError(
            f"LatentSync service unreachable at "
            f"{settings.latentsync_service_url}: {e.reason}. "
            f"Is the lipsync-latentsync container running? "
            f"(`make logs-latentsync` for clues.)"
        ) from e

    if progress is not None:
        progress(1.0)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise LipsyncError(
            f"LatentSync returned 200 but no output file was written at "
            f"{output_path}."
        )

    return LipsyncResult(
        backend="latentsync",
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
        # PR-LS-1a / 1b legacy — should not appear after PR-LS-1c. The
        # service body carries a structured message explaining staging;
        # surface that so the user sees the same "coming in PR-LS-1c"
        # context whether they hit the API or `make run-latentsync`.
        if isinstance(detail, dict):
            msg = detail.get("message", str(detail))
            phase = detail.get("phase", "")
            next_step = detail.get("next_step", "")
            full = msg
            if next_step:
                full = f"{msg} {next_step}"
            raise LipsyncError(
                f"LatentSync service not implemented "
                f"({phase or 'scaffold'}): {full} See docs/lipsync.md."
            )
        raise LipsyncError(
            f"LatentSync service not implemented: {detail}. "
            f"See docs/lipsync.md."
        )
    if e.code == 503:
        # Missing weights. Only hit in PR-LS-1b and later.
        fix = detail.get("fix") if isinstance(detail, dict) else None
        raise LipsyncError(
            f"LatentSync weights missing: {detail}. "
            f"{f'Fix: {fix}' if fix else 'See docs/lipsync.md.'}"
        )
    if e.code == 422:
        err = detail.get("error", detail) if isinstance(detail, dict) else detail
        raise LipsyncError(f"LatentSync couldn't process the clip: {err}")
    if e.code == 400:
        raise LipsyncError(f"LatentSync rejected request: {detail}")
    if e.code == 500:
        err = detail.get("error", detail) if isinstance(detail, dict) else detail
        raise LipsyncError(f"LatentSync crashed: {err}. See service logs.")
    raise LipsyncError(f"LatentSync failed (HTTP {e.code}): {detail}")
