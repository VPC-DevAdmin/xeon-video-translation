"""MuseTalk lipsync microservice.

Endpoints:
- GET  /health         — liveness + what this build can do
- POST /lipsync        — run lipsync. PR 1a returns 501 with structured body.
                          PR 1b adds weights + preprocessing. PR 1c adds inference.

Input paths live under /jobs which is a shared volume also mounted by the
main backend. No bytes travel over HTTP; only path references do.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger(__name__)

VERSION = "0.1.0"

# PR scaffold: flipped to True by PR 1c once inference lands.
INFERENCE_IMPLEMENTED = False

app = FastAPI(
    title="polyglot-demo lipsync (MuseTalk)",
    version=VERSION,
    description="CPU MuseTalk lipsync service. PR 1a: HTTP scaffold only.",
)


class LipsyncRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video (shared /jobs volume).")
    audio_path: str = Field(..., description="Absolute path to the new audio (translated_audio.wav).")
    output_path: str = Field(..., description="Where to write the lipsynced MP4.")


class LipsyncResponse(BaseModel):
    status: Literal["ok", "not_implemented"]
    output_path: str | None = None
    frames_processed: int | None = None
    duration_ms: int | None = None
    detail: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "lipsync-musetalk",
        "version": VERSION,
        "inference_implemented": INFERENCE_IMPLEMENTED,
        "milestone": "PR 1a — HTTP scaffold only",
    }


@app.post("/lipsync", response_model=LipsyncResponse)
def lipsync(req: LipsyncRequest) -> LipsyncResponse:
    # PR 1a check: inputs must exist in the shared volume. This validates the
    # volume wiring end-to-end before we invest in inference code.
    video = Path(req.video_path)
    audio = Path(req.audio_path)

    if not video.exists():
        raise HTTPException(
            status_code=400,
            detail=f"video_path not visible to this service: {video!s}. "
                   "Both services must mount the same /jobs volume.",
        )
    if not audio.exists():
        raise HTTPException(
            status_code=400,
            detail=f"audio_path not visible to this service: {audio!s}.",
        )

    if not INFERENCE_IMPLEMENTED:
        log.info("lipsync requested for %s + %s — returning 501 (PR 1a scaffold)",
                 video.name, audio.name)
        raise HTTPException(
            status_code=501,
            detail={
                "phase": "not-implemented",
                "message": (
                    "MuseTalk inference is not implemented yet. This service is "
                    "currently the PR 1a scaffold — it validates the HTTP + "
                    "shared-volume wiring only. Inference lands in PR 1c."
                ),
                "see_also": "docs/lipsync.md",
            },
        )

    # PR 1c will replace this with the real call.
    raise HTTPException(status_code=500, detail="unreachable in PR 1a")
