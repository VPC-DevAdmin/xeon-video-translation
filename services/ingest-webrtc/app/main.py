"""Bounded WebRTC recording with idempotent job submission."""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import httpx
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaRecorder, MediaRelay
from .identity import owner, require
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger("ingest")
JOB_ARTIFACTS_DIR = Path(os.getenv("JOB_ARTIFACTS_DIR", "./jobs")).resolve()
INGEST_DIR = JOB_ARTIFACTS_DIR / "ingest"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CORS_ORIGINS = [
    x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3030").split(",")
]
MAX_SECONDS = int(os.getenv("INGEST_MAX_SECONDS", "120"))
MAX_BYTES = int(os.getenv("INGEST_MAX_MB", "100")) * 1024 * 1024
MAX_SESSIONS = int(os.getenv("INGEST_MAX_SESSIONS", "4"))
SESSION_TTL = int(os.getenv("INGEST_SESSION_TTL", "3600"))
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "")
_sessions = {}
_sweeper = None


def ice_servers():
    servers = json.loads(os.getenv("ICE_SERVERS_JSON", "[]"))
    secret = os.getenv("TURN_SHARED_SECRET", "")
    urls = json.loads(os.getenv("TURN_URLS_JSON", "[]"))
    if secret and urls:
        import hmac, hashlib, base64

        username = f"{int(time.time()) + 600}:{owner.get()}"
        credential = base64.b64encode(
            hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
        ).decode()
        servers.append({"urls": urls, "username": username, "credential": credential})
    return servers


def rtc_configuration():
    return RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ice_servers()])


def validate_id(value):
    if not re.fullmatch(r"[a-f0-9]{12,32}", value):
        raise HTTPException(400, "invalid session identifier")
    return value


@dataclass
class Session:
    session_id: str
    target_language: str
    mode: str
    source_language: str | None
    pc: RTCPeerConnection
    recorder: MediaRecorder | None
    input_path: Path
    tracks: list = field(default_factory=list)
    job_id: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.monotonic)
    stopped: bool = False
    error: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    owner: str = "local"
    caption: str = ""
    caption_task: asyncio.Task | None = None
    preview: object | None = None


app = FastAPI(title="polyglot-ingest-webrtc", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def internal_access(request: Request, call_next):
    from fastapi.responses import JSONResponse

    if INTERNAL_KEY and not secrets.compare_digest(
        request.headers.get("x-internal-key", ""), INTERNAL_KEY
    ):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if (
        not INTERNAL_KEY
        and request.client
        and request.client.host not in ("127.0.0.1", "::1", "testclient")
    ):
        return JSONResponse(
            {"detail": "set INTERNAL_API_KEY for remote access"}, status_code=503
        )
    token = owner.set(
        request.headers.get("x-owner-id", "local") if INTERNAL_KEY else "local"
    )
    try:
        return await call_next(request)
    finally:
        owner.reset(token)


class OfferIn(BaseModel):
    sdp: str = Field(..., max_length=65536)
    type: str = Field("offer", pattern="^offer$")
    target_language: str = Field(..., min_length=2, max_length=8)
    source_language: str | None = None
    live_captions: bool = False
    mode: str = Field("fast", pattern="^(fast|quality|dub)$")


@app.get("/config")
async def config():
    return {
        "iceServers": ice_servers(),
        "maxSeconds": MAX_SECONDS,
        "modes": ["fast", "quality", "dub", "avatar"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": sum(not s.stopped for s in _sessions.values()),
        "gpu_codecs": os.getenv("WEBRTC_GPU_CODECS", "0") == "1",
        "host_frame_exchange": True,
    }


@app.post("/sessions/{session_id}/offer")
async def offer(session_id: str, body: OfferIn):
    validate_id(session_id)
    if session_id in _sessions:
        raise HTTPException(409, "session already exists")
    if sum(not s.stopped for s in _sessions.values()) >= MAX_SESSIONS:
        raise HTTPException(429, "recording capacity reached")
    directory = INGEST_DIR / session_id
    directory.mkdir(parents=True, exist_ok=False)
    pc = RTCPeerConnection(rtc_configuration())
    path = directory / "input.mp4"
    from .gpu_media import recorder as make_recorder
    recorder = make_recorder(str(path))
    session = Session(
        session_id,
        body.target_language.lower(),
        body.mode,
        body.source_language,
        pc,
        recorder,
        path,
    )
    session.owner = owner.get()
    (directory / "owner.json").write_text(json.dumps({"owner_id": session.owner}))
    _sessions[session_id] = session
    relay = MediaRelay()

    @pc.on("track")
    def track_received(track):
        session.tracks.append(track.kind)
        recorder.addTrack(relay.subscribe(track))
        if track.kind == "audio" and body.live_captions:
            from .preview import TranscriptPreview

            session.preview = TranscriptPreview(
                directory,
                BACKEND_URL,
                body.source_language,
                lambda text: setattr(session, "caption", text),
                session.owner,
            )
            session.preview.start()

            async def captions():
                import av

                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                incoming = relay.subscribe(track)
                try:
                    while True:
                        frame = await incoming.recv()
                        for audio in resampler.resample(frame):
                            session.preview.push(audio.to_ndarray().reshape(-1))
                finally:
                    incoming.stop()

            session.caption_task = asyncio.create_task(captions())

    @pc.on("connectionstatechange")
    async def connection_changed():
        if pc.connectionState == "failed" and not session.stopped:
            async with session.lock:
                session.error = "media connection failed"
                await close_recording(session)

    from .gpu_media import prefer_h264
    prefer_h264(pc)
    try:
        await asyncio.wait_for(
            pc.setRemoteDescription(RTCSessionDescription(body.sdp, body.type)), 15
        )
        if not {"audio", "video"}.issubset(session.tracks):
            raise HTTPException(422, "both microphone and camera tracks are required")
        await recorder.start()
        await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()), 15)
    except BaseException:
        await close_recording(session)
        raise
    return {
        "session_id": session_id,
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


async def close_recording(session):
    if session.stopped:
        return
    session.stopped = True
    if session.caption_task:
        session.caption_task.cancel()
        await asyncio.gather(session.caption_task, return_exceptions=True)
    if session.preview:
        await session.preview.close()
    try:
        if session.recorder:
            await session.recorder.stop()
    except Exception:
        session.error = "video recording failed; check media service logs"
        log.exception("recorder failed")
    finally:
        session.recorder = None
        await session.pc.close()


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(validate_id(session_id))
    require(session)
    return {
        "stopped": session.stopped,
        "error": session.error,
        "job_id": session.job_id,
        "caption": session.caption,
        "seconds": min(MAX_SECONDS, time.monotonic() - session.created),
        "max_seconds": MAX_SECONDS,
    }


@app.post("/sessions/{session_id}/stop")
async def stop(session_id: str):
    session = _sessions.get(validate_id(session_id))
    require(session)
    async with session.lock:
        if session.job_id:
            return {"session_id": session_id, "job_id": session.job_id}
        await close_recording(session)
        if session.error:
            raise HTTPException(422, session.error)
        if not session.input_path.exists() or not session.input_path.stat().st_size:
            raise HTTPException(422, "no media recorded")
        if session.input_path.stat().st_size > MAX_BYTES:
            raise HTTPException(413, "recording exceeds size limit")
        params = {
            "target_language": session.target_language,
            "mode": session.mode,
            "request_id": session.request_id,
        }
        if session.source_language:
            params["source_language"] = session.source_language
        try:
            async with httpx.AsyncClient(
                timeout=120,
                headers={"x-internal-key": INTERNAL_KEY, "x-owner-id": session.owner},
            ) as client:
                with session.input_path.open("rb") as video:
                    response = await client.post(
                        f"{BACKEND_URL}/jobs",
                        data=params,
                        files={"video": ("input.mp4", video, "video/mp4")},
                    )
            if response.status_code != 201:
                raise HTTPException(
                    502, f"job submission failed ({response.status_code}); retry Stop"
                )
            session.job_id = response.json()["job_id"]
        except httpx.HTTPError as exc:
            raise HTTPException(
                502, "backend unavailable; recording retained, retry Stop"
            ) from exc
        return {"session_id": session_id, "job_id": session.job_id}


@app.delete("/sessions/{session_id}")
async def discard(session_id: str):
    import shutil

    session = _sessions.get(validate_id(session_id))
    if session:
        require(session)
        async with session.lock:
            await close_recording(session)
            _sessions.pop(session_id, None)
            shutil.rmtree(session.input_path.parent, ignore_errors=True)
    return {"status": "closed"}


def clean_orphans():
    import shutil
    from .avatar import ROOT, _sessions as avatars

    for root, active in ((INGEST_DIR, _sessions), (ROOT, avatars)):
        if not root.exists():
            continue
        for directory in root.iterdir():
            if not directory.is_dir() or directory.name in active:
                continue
            # A 24-hour grace period is longer than bounded inference timeouts.
            if (
                re.fullmatch(r"[a-f0-9]{12,32}", directory.name)
                and time.time() - directory.stat().st_mtime > 86400
            ):
                shutil.rmtree(directory, ignore_errors=True)


async def sweep():
    ticks = 0
    while True:
        await asyncio.sleep(2)
        ticks += 1
        if ticks % 300 == 0:
            await asyncio.to_thread(clean_orphans)
        for session in list(_sessions.values()):
            try:
                age = time.monotonic() - session.created
                size = (
                    session.input_path.stat().st_size
                    if session.input_path.exists()
                    else 0
                )
                if not session.stopped and (age >= MAX_SECONDS or size >= MAX_BYTES):
                    async with session.lock:
                        session.error = "recording limit reached; press Stop to submit"
                        await close_recording(session)
                if session.stopped and age >= SESSION_TTL:
                    token = owner.set(session.owner)
                    try:
                        await discard(session.session_id)
                    finally:
                        owner.reset(token)
            except Exception:
                log.exception("session cleanup failed")


@app.on_event("startup")
async def startup():
    global _sweeper
    from .gpu_media import probe
    await asyncio.to_thread(probe)
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(clean_orphans)
    _sweeper = asyncio.create_task(sweep())


@app.on_event("shutdown")
async def shutdown():
    if _sweeper:
        _sweeper.cancel()
        await asyncio.gather(_sweeper, return_exceptions=True)
    await asyncio.gather(
        *(close_recording(s) for s in list(_sessions.values())), return_exceptions=True
    )


from .avatar import router as avatar_router

app.include_router(avatar_router)
