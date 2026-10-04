"""Live microphone turns and prepared-image avatar output over WebRTC."""

from __future__ import annotations
import asyncio
import json
import os
import uuid
import wave
from collections import deque
from io import BytesIO
from pathlib import Path
import av
import httpx
import numpy as np
from PIL import Image, ImageDraw
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from .playback import Playback, AudioOutput, VideoOutput
from .identity import owner, require

router = APIRouter(prefix="/avatar", tags=["avatar"])
_sessions = {}
MAX_SESSIONS = int(os.getenv("AVATAR_MAX_SESSIONS", "1"))
BACKEND = os.getenv(
    "AVATAR_BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000")
)
MUSETALK = os.getenv("AVATAR_MUSETALK_URL", "http://localhost:8089")
ROOT = Path(os.getenv("JOB_ARTIFACTS_DIR", "./jobs")).resolve() / "avatars"


class Avatar:
    def __init__(self, identifier, directory, image, language):
        self.id, self.directory, self.language = identifier, directory, language
        self.playback = Playback(image)
        self.pc = None
        self.channel = None
        self.consumer = None
        self.turn = None
        self.history = []
        self.expiry = None
        self.closed = False
        self.owner = "local"
        self.voice = None
        self.preview = None
        self.pending_played = {}
        self.turn_history = {}
        self.metrics = {
            "turns": 0,
            "interruptions": 0,
            "render_fallbacks": 0,
            "first_reply_seconds": [],
        }

    def notify(self, kind, **values):
        if self.channel and self.channel.readyState == "open":
            self.channel.send(json.dumps({"type": kind, **values}))

    def interrupt(self):
        self.metrics["interruptions"] += 1
        self.pending_played.clear()
        self.playback.interrupt()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        self.notify("listening")

    def acknowledge(self, token):
        item = self.pending_played.pop(token, None)
        if item is None or item["generation"] != self.playback.generation:
            return
        entry = self.turn_history.get(item["generation"])
        if entry is not None:
            entry["sentences"][item["sentence"]] = item["text"]
            self.history = self.history[-12:]
            assistant = entry["assistant"]
            assistant["content"] = " ".join(
                entry["sentences"][i] for i in sorted(entry["sentences"])
            )

    async def reply(self, samples, generation):
        import time

        started = time.monotonic()
        self.metrics["turns"] += 1
        retained = sum(
            p.stat().st_size for p in self.directory.rglob("*") if p.is_file()
        )
        if retained > int(os.getenv("AVATAR_MAX_STORAGE_MB", "256")) * 1024**2:
            self.notify(
                "error", message="Session storage limit reached. Start a new session."
            )
            return
        turn_directory = self.directory / uuid.uuid4().hex
        turn_directory.mkdir()
        path = turn_directory / "input.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(samples.astype(np.int16).tobytes())
        self.notify("thinking")
        queue = asyncio.Queue(maxsize=2)
        first = True
        headers = {
            "x-internal-key": os.getenv("INTERNAL_API_KEY", ""),
            "x-owner-id": self.owner,
        }

        async def receive(client):
            try:
                async with client.stream(
                    "POST",
                    f"{BACKEND}/avatar/respond",
                    json={
                        "audio_path": str(path),
                        "language": self.language,
                        "voice": self.voice,
                        "history": [m for m in self.history[-12:] if m.get("content")],
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if generation != self.playback.generation:
                            return
                        event = json.loads(line)
                        if event["type"] == "transcript":
                            self.notify("transcript", text=event["text"])
                            assistant = {"role": "assistant", "content": ""}
                            self.history += [
                                {"role": "user", "content": event["text"]},
                                assistant,
                            ]
                            self.turn_history = {
                                generation: {"assistant": assistant, "sentences": {}}
                            }
                        elif event["type"] == "text":
                            self.notify("reply", text=event["text"])
                        elif event["type"] == "error":
                            raise RuntimeError(event["message"])
                        elif event["type"] == "audio":
                            await queue.put(event)
                await queue.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put(exc)

        producer = None
        tail = np.empty(0, dtype=np.int16)
        previous_frame = None
        try:
            async with httpx.AsyncClient(timeout=90, headers=headers) as client:
                producer = asyncio.create_task(receive(client))
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    if isinstance(event, Exception):
                        raise event
                    if generation != self.playback.generation:
                        break
                    if (
                        sum(
                            p.stat().st_size
                            for p in self.directory.rglob("*")
                            if p.is_file()
                        )
                        > int(os.getenv("AVATAR_MAX_STORAGE_MB", "256")) * 1024**2
                    ):
                        raise RuntimeError(
                            "Session storage limit reached; start a new session."
                        )
                    audio_path = Path(event["path"]).resolve()
                    if not audio_path.is_relative_to(turn_directory):
                        raise RuntimeError("invalid audio artifact path")
                    audio = await asyncio.to_thread(read_audio, audio_path)
                    context_path = audio_path.with_name(
                        audio_path.stem + "-context.wav"
                    )
                    frames_path = audio_path.with_suffix(".npy")
                    context = len(tail)
                    with wave.open(str(context_path), "wb") as wav:
                        wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                        wav.writeframes(np.concatenate([tail, audio]).tobytes())
                    rendered_safely = False
                    try:
                        render = await client.post(
                            f"{MUSETALK}/avatar/render",
                            json={
                                "image_path": str(self.directory / "image.png"),
                                "audio_path": str(context_path),
                                "output_path": str(frames_path),
                            },
                            timeout=float(os.getenv("AVATAR_RENDER_TIMEOUT", "5")),
                        )
                        render.raise_for_status()
                        frames = np.load(frames_path, allow_pickle=False)[
                            round(context / 48000 * 25) :
                        ]
                        if not len(frames):
                            raise RuntimeError("renderer returned no frames")
                        if previous_frame is not None:
                            for i in range(min(3, len(frames))):
                                alpha = (i + 1) / 3
                                frames[i] = (
                                    previous_frame * (1 - alpha) + frames[i] * alpha
                                ).astype(np.uint8)
                        previous_frame = frames[-1].copy()
                        rendered_safely = True
                    except (httpx.HTTPError, RuntimeError, OSError, ValueError):
                        self.metrics["render_fallbacks"] += 1
                        self.notify(
                            "degraded",
                            message="Video rendering delayed; continuing with audio.",
                        )
                        frames = self.playback.image[None, :, :, :]
                    # Successful HTTP completion means the renderer released files.
                    if rendered_safely:
                        for artifact in (audio_path, context_path, frames_path):
                            artifact.unlink(missing_ok=True)
                    timing = await self.playback.enqueue(audio, frames, generation)
                    if timing is None:
                        break
                    if first:
                        elapsed = time.monotonic() - started
                        self.metrics["first_reply_seconds"].append(elapsed)
                        self.metrics["first_reply_seconds"] = self.metrics[
                            "first_reply_seconds"
                        ][-100:]
                        self.notify("latency", seconds=elapsed)
                        first = False
                    if event.get("final"):
                        token = uuid.uuid4().hex
                        self.pending_played[token] = {
                            "generation": generation,
                            "sentence": event.get("sentence_id", 0),
                            "text": event.get("text", ""),
                            "end": timing["end"],
                        }
                        self.notify(
                            "playout",
                            token=token,
                            end=timing["end"] - timing["base"],
                            generation=generation,
                        )
                    self.notify("speaking")
                    tail = audio[-9600:].copy()
                    # A timed-out renderer may still own these paths; cleanup waits
                    # until the session expires rather than unlinking underneath it.
            self.history = self.history[-12:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.notify("error", message=str(exc))
        finally:
            if producer:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)

    async def consume(self, track):
        # Configurable energy VAD baseline. 600 ms endpoint, 200 ms pre-roll.
        # Echo cancellation is requested in the browser. Tune on actual microphones.
        from .vad import Detector

        try:
            detector = Detector()
        except Exception:
            self.notify(
                "error",
                message="VAD configuration unavailable; check server readiness.",
            )
            await self.close()
            return
        endpoint = float(os.getenv("AVATAR_ENDPOINT_SECONDS", "0.6"))
        from .preview import TranscriptPreview

        if os.getenv("AVATAR_LIVE_CAPTIONS", "false").lower() == "true":
            self.preview = TranscriptPreview(
                self.directory,
                BACKEND,
                self.language,
                lambda text: self.notify("partial_transcript", text=text),
                self.owner,
            )
            self.preview.start()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        preroll = deque(maxlen=10)
        utterance = []
        silence = 0
        total = 0
        active = False
        try:
            while True:
                frame = await track.recv()
                for audio_frame in resampler.resample(frame):
                    samples = audio_frame.to_ndarray().reshape(-1).copy()
                    duration = len(samples) / 16000
                    if self.preview:
                        self.preview.push(samples)
                    voiced = detector.speech(samples)
                    if voiced and not active:
                        self.interrupt()
                        utterance = list(preroll)
                        total = sum(len(x) for x in utterance) / 16000
                        active = True
                    if active:
                        utterance.append(samples)
                        total += duration
                        silence = 0 if voiced else silence + duration
                        if silence >= endpoint or total >= 15:
                            if total - silence >= 0.2:
                                self.turn = asyncio.create_task(
                                    self.reply(
                                        np.concatenate(utterance),
                                        self.playback.generation,
                                    )
                                )
                            active, utterance, total, silence = False, [], 0, 0
                            preroll.clear()
                    else:
                        preroll.append(samples)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.notify("error", message="microphone stream ended")

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.interrupt()
        current = asyncio.current_task()
        tasks = [
            t for t in (self.consumer, self.turn, self.expiry) if t and t is not current
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.preview:
            await self.preview.close()
        if self.pc:
            await self.pc.close()
        _sessions.pop(self.id, None)
        # In-flight GPU calls may still own artifacts. Remove only aged session
        # directories at next startup; never delete files under a running worker.


def read_audio(path):
    samples = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                samples.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            samples.append(out.to_ndarray().reshape(-1))
    return np.concatenate(samples)


async def require_warm_models():
    if os.getenv("AVATAR_REQUIRE_WARM_MODELS", "0") != "1":
        return
    headers = {"x-internal-key": os.getenv("INTERNAL_API_KEY", ""),
               "x-owner-id": owner.get()}
    try:
        async with httpx.AsyncClient(timeout=3, headers=headers) as client:
            speech, renderer = await asyncio.gather(
                client.get(f"{BACKEND}/health"), client.get(f"{MUSETALK}/health")
            )
            speech.raise_for_status()
            renderer.raise_for_status()
            if (speech.json().get("warmup", {}).get("status") == "done"
                    and speech.json().get("warmup", {}).get("avatar_inference_ready") is True
                    and renderer.json().get("avatar_inference_warm") is True):
                return
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    raise HTTPException(503, "Avatar models are not ready yet; retry shortly.",
                        headers={"Retry-After": "5"})


@router.post("/sessions")
async def create(
    image: UploadFile = File(...),
    language: str = Form("en"),
    voice: str | None = Form(None),
):
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, "avatar capacity reached")
    await require_warm_models()
    languages = {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "hu",
        "ko",
        "ja",
        "hi",
        "zh",
    }
    if language not in languages:
        raise HTTPException(400, "unsupported avatar language")
    payload = await image.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise HTTPException(413, "image exceeds 5 MB")
    try:
        picture = Image.open(BytesIO(payload))
        if picture.width * picture.height > 20_000_000:
            raise ValueError("image too large")
        picture = picture.convert("RGB")
        picture.thumbnail((512, 512))
        picture = picture.resize(
            (max(2, picture.width // 2 * 2), max(2, picture.height // 2 * 2))
        )
    except Exception as exc:
        raise HTTPException(422, "upload a valid portrait image") from exc
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, "avatar capacity reached")
    identifier = uuid.uuid4().hex
    directory = ROOT / identifier
    directory.mkdir(parents=True)
    picture.save(directory / "image.png")
    ImageDraw.Draw(picture).text((8, picture.height - 20), "AI avatar", fill="white")
    session = Avatar(
        identifier, directory, np.asarray(picture)[:, :, ::-1].copy(), language
    )
    session.owner = owner.get()
    session.voice = voice if isinstance(voice, str) else None
    (directory / "owner.json").write_text(json.dumps({"owner_id": session.owner}))
    _sessions[identifier] = session

    async def expire():
        await asyncio.sleep(1800)
        await session.close()

    session.expiry = asyncio.create_task(expire())
    return {"session_id": identifier}


class Offer(BaseModel):
    sdp: str = Field(..., max_length=65536)
    type: str = Field("offer", pattern="^offer$")


@router.post("/sessions/{identifier}/offer")
async def offer(identifier: str, body: Offer):
    from .main import rtc_configuration

    session = _sessions.get(identifier)
    require(session)
    if session.pc:
        raise HTTPException(409, "avatar already connected")
    pc = session.pc = RTCPeerConnection(rtc_configuration())
    pc.addTrack(AudioOutput(session.playback))
    pc.addTrack(VideoOutput(session.playback))
    from .gpu_media import prefer_h264
    prefer_h264(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            session.consumer = asyncio.create_task(session.consume(track))

    @pc.on("datachannel")
    def on_channel(channel):
        session.channel = channel

        @channel.on("message")
        def message(value):
            if value == "interrupt":
                session.interrupt()
            else:
                try:
                    event = json.loads(value)
                    if event.get("type") == "played":
                        session.acknowledge(event.get("token"))
                except (ValueError, TypeError):
                    pass

    @pc.on("connectionstatechange")
    async def changed():
        if pc.connectionState == "failed":
            await session.close()

    try:
        await pc.setRemoteDescription(RTCSessionDescription(body.sdp, body.type))
        await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()), 15)
    except BaseException:
        await session.close()
        raise
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@router.delete("/sessions/{identifier}")
async def delete(identifier: str):
    session = _sessions.get(identifier)
    if session:
        require(session)
        await session.close()
    return {"status": "closed"}


@router.on_event("shutdown")
async def shutdown():
    await asyncio.gather(*(s.close() for s in list(_sessions.values())))


@router.get("/sessions/{identifier}")
async def status(identifier: str):
    session = require(_sessions.get(identifier))
    return {
        "metrics": session.metrics,
        "history": session.history,
        "generation": session.playback.generation,
        "playback": {"video_frames_sent": session.playback.video_frames_sent,
                     "video_frames_skipped": session.playback.video_frames_skipped},
    }
