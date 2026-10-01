"""WebRTC webcam ingest for polyglot-demo (GPU track).

Browser                      this service                      backend
-------                      ------------                      -------
getUserMedia()
RTCPeerConnection  --offer-> POST /sessions/{id}/offer
                   <-answer- (aiortc answers, starts a MediaRecorder
                              writing /jobs/ingest/<id>/input.mp4)
... webcam streams over SRTP ...
                   ---------> POST /sessions/{id}/stop
                              (closes recorder, submits the file to
                               POST {BACKEND_URL}/jobs with the mode's
                               parameters, returns job_id)

This is the *record-then-submit* shape, which is what mode 1 ("real-time":
done within a few minutes of stopping) and mode 2 (batch, best quality)
need. Mode 3 (live avatar) and any sub-utterance streaming need the
media to be consumed as it arrives rather than written to a file; that
is a separate consumer registered on the same track and is described,
not implemented, in docs/gpu/mode3-avatar.md.

Why a separate service: aiortc wants its own event loop and UDP ports,
and the backend's job model is "a file appeared". Keeping ingest apart
means the backend never learns about ICE.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")

JOB_ARTIFACTS_DIR = Path(os.environ.get("JOB_ARTIFACTS_DIR", "./jobs")).resolve()
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3030").split(",") if o.strip()]
INGEST_DIR = JOB_ARTIFACTS_DIR / "ingest"

# What each user-facing mode means in backend job parameters. Mode 1 trades
# quality for speed; mode 2 is "everything on". The exact knobs will move as
# the GPU numbers come in — keep this table the single place they live.
MODE_PARAMS: dict[str, dict[str, str]] = {
    # Mode 1 — real-time-ish: finish within a few minutes of stopping.
    "fast": {
        "lipsync_backend": "musetalk",
        "tts_backend": "auto",
        "musetalk_face_restore": "none",
    },
    # Mode 2 — batch: highest quality we can get, time is not a constraint.
    "quality": {
        "lipsync_backend": "latentsync",
        "tts_backend": "auto",
        "enable_output_stabilization": "true",
    },
}


@dataclass
class Session:
    session_id: str
    target_language: str
    mode: str
    source_language: str | None
    pc: RTCPeerConnection
    recorder: MediaRecorder | None = None
    input_path: Path | None = None
    tracks: list[str] = field(default_factory=list)
    job_id: str | None = None


_sessions: dict[str, Session] = {}

app = FastAPI(title="polyglot-ingest-webrtc", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OfferIn(BaseModel):
    sdp: str
    type: str = "offer"
    target_language: str = Field(..., min_length=2, max_length=8)
    source_language: str | None = None
    mode: str = Field("fast", pattern="^(fast|quality)$")


class OfferOut(BaseModel):
    session_id: str
    sdp: str
    type: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backend_url": BACKEND_URL,
        "active_sessions": len(_sessions),
        "modes": sorted(MODE_PARAMS),
    }


@app.post("/sessions/{session_id}/offer", response_model=OfferOut)
async def offer(session_id: str, body: OfferIn) -> OfferOut:
    """Accept the browser's SDP offer, start recording, return the answer."""
    if session_id in _sessions:
        raise HTTPException(409, f"session {session_id} already exists")

    session_dir = INGEST_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    input_path = session_dir / "input.mp4"

    pc = RTCPeerConnection()
    recorder = MediaRecorder(str(input_path))
    sess = Session(
        session_id=session_id,
        target_language=body.target_language.lower(),
        source_language=body.source_language.lower() if body.source_language else None,
        mode=body.mode,
        pc=pc,
        recorder=recorder,
        input_path=input_path,
    )
    _sessions[session_id] = sess

    @pc.on("track")
    def on_track(track):
        log.info("session %s: %s track received", session_id, track.kind)
        sess.tracks.append(track.kind)
        recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            log.info("session %s: %s track ended", session_id, track.kind)

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("session %s: connection %s", session_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed") and session_id in _sessions:
            # Browser went away without calling /stop. Finalize what we have
            # so the file is playable, but don't auto-submit a job.
            await _close(sess, submit=False)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    await recorder.start()
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return OfferOut(session_id=session_id, sdp=pc.localDescription.sdp, type=pc.localDescription.type)


@app.post("/sessions/{session_id}/stop")
async def stop(session_id: str) -> dict:
    """Stop recording and hand the file to the backend as a job."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(404, f"unknown session {session_id}")
    job = await _close(sess, submit=True)
    return {"session_id": session_id, "input_path": str(sess.input_path), **job}


@app.get("/sessions")
async def list_sessions() -> list[dict]:
    return [
        {"session_id": s.session_id, "mode": s.mode, "tracks": s.tracks, "job_id": s.job_id}
        for s in _sessions.values()
    ]


async def _close(sess: Session, submit: bool) -> dict:
    _sessions.pop(sess.session_id, None)
    if sess.recorder is not None:
        await sess.recorder.stop()
        sess.recorder = None
    await sess.pc.close()
    if not submit or sess.input_path is None:
        return {}
    if not sess.input_path.exists() or sess.input_path.stat().st_size == 0:
        raise HTTPException(422, "no media was recorded for this session")
    job_id = await _submit_job(sess)
    sess.job_id = job_id
    return {"job_id": job_id}


async def _submit_job(sess: Session) -> str:
    """POST the recorded file to the backend's existing upload endpoint.

    The backend and this service share the /jobs volume, but the backend's
    API is upload-shaped, so we re-upload over loopback rather than teach
    the backend a second "adopt this path" entrypoint. That is a few MB over
    localhost; if clips get long, add a path-based endpoint on the backend.
    """
    assert sess.input_path is not None
    params = dict(MODE_PARAMS[sess.mode])
    params["target_language"] = sess.target_language
    if sess.source_language:
        params["source_language"] = sess.source_language
    async with httpx.AsyncClient(timeout=120) as client:
        with sess.input_path.open("rb") as f:
            resp = await client.post(
                f"{BACKEND_URL}/jobs",
                data=params,
                files={"video": ("input.mp4", f, "video/mp4")},
            )
    if resp.status_code != 201:
        raise HTTPException(502, f"backend rejected job: {resp.status_code} {resp.text[:300]}")
    job_id = resp.json()["job_id"]
    log.info("session %s -> job %s (mode=%s)", sess.session_id, job_id, sess.mode)
    return job_id


@app.on_event("shutdown")
async def _shutdown() -> None:
    await asyncio.gather(*(_close(s, submit=False) for s in list(_sessions.values())), return_exceptions=True)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
