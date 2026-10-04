"""Video assistant sessions over WebRTC.

Turn: Silero endpoint -> prepared acknowledgement plays at once -> the backend plans the
whole reply text and streams TTS audio -> FlashHead renders it chunk by chunk ->
chunks are scheduled on one timeline at a head start chosen from the renderer's
measured rate -> idle footage covers the gap -> playout at 25 fps with stall counting.
Speaking again or pressing Interrupt clears everything and resets the renderer.

Services (ingest runs with host networking):
  ASSISTANT_BACKEND_URL   backend with /assistant/respond and /assistant/speak
  ASSISTANT_RENDERER_URL  FlashHead render service (services/flashhead)
Tuning: ASSISTANT_HEAD_START (s, default 10), ASSISTANT_MAX_HEAD_START (20),
ASSISTANT_IDLE_CHUNKS (2), ASSISTANT_ACK_TEXT_<LANG>, ASSISTANT_RENDER_TIMEOUT (30).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
import wave
from collections import deque
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import httpx
import numpy as np
from PIL import Image
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .identity import owner, require
from .timeline import Timeline, head_start_required

router = APIRouter(prefix="/assistant", tags=["assistant"])
_sessions: dict = {}
MAX_SESSIONS = int(os.getenv("ASSISTANT_MAX_SESSIONS", os.getenv("AVATAR_MAX_SESSIONS", "1")))
BACKEND = os.getenv("ASSISTANT_BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8088"))
RENDERER = os.getenv("ASSISTANT_RENDERER_URL", "http://localhost:8094")
ROOT = Path(os.getenv("JOB_ARTIFACTS_DIR", "./jobs")).resolve() / "avatars"
HEAD_START = float(os.getenv("ASSISTANT_HEAD_START", "10"))
MAX_HEAD_START = float(os.getenv("ASSISTANT_MAX_HEAD_START", "20"))
IDLE_CHUNKS = int(os.getenv("ASSISTANT_IDLE_CHUNKS", "2"))          # rendered before the session answers
IDLE_SECONDS = float(os.getenv("ASSISTANT_IDLE_SECONDS", "12"))      # grown to this in the background, then looped
CACHE_DIR = Path(os.getenv("JOB_ARTIFACTS_DIR", "./jobs")).resolve() / "personas"
RENDER_TIMEOUT = float(os.getenv("ASSISTANT_RENDER_TIMEOUT", "30"))
FPS = 25
ACK_TEXT = {
    "en": "Let me think about that for a second.", "es": "Déjame pensarlo un momento.",
    "fr": "Laissez-moi réfléchir un instant.", "de": "Lass mich kurz nachdenken.",
    "it": "Fammi pensare un attimo.", "pt": "Deixe-me pensar um instante.",
    "zh": "让我想一想。", "ja": "少し考えさせてください。",
}


def ack_text(language: str) -> str:
    return os.getenv(f"ASSISTANT_ACK_TEXT_{language.upper()}") or ACK_TEXT.get(language) or ACK_TEXT["en"]


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    """PCM16 mono resample through libswresample."""
    if src == dst or not len(samples):
        return samples.astype(np.int16, copy=False)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=dst)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples.astype(np.int16))[None, :], format="s16", layout="mono")
    frame.sample_rate = src
    frame.pts = 0
    frame.time_base = Fraction(1, src)
    out = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    out += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(out) if out else np.zeros(0, np.int16)


class AudioOut(MediaStreamTrack):
    kind = "audio"

    def __init__(self, timeline: Timeline):
        super().__init__()
        self.timeline, self.index = timeline, 0

    async def recv(self):
        tl = self.timeline
        self.index = max(self.index, int(tl.now() * 50))
        pts = self.index * 960
        seconds = pts / 48000
        if tl.first_audio_seconds is None:
            tl.first_audio_seconds = seconds
        await asyncio.sleep(max(0, tl.epoch + seconds - time.monotonic()))
        frame = av.AudioFrame.from_ndarray(tl.audio_packet(seconds, 960)[None, :], format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 48000, pts, Fraction(1, 48000)
        self.index += 1
        return frame


class VideoOut(MediaStreamTrack):
    kind = "video"

    def __init__(self, timeline: Timeline):
        super().__init__()
        self.timeline, self.index = timeline, 0

    async def recv(self):
        tl = self.timeline
        scheduled = max(self.index, int(tl.now() * tl.fps))
        if tl.frames_sent:
            tl.frames_skipped += scheduled - self.index
        self.index = scheduled
        seconds = self.index / tl.fps
        await asyncio.sleep(max(0, tl.epoch + seconds - time.monotonic()))
        image, source = tl.frame_at(seconds)
        tl.frames_sent += 1
        if source != "clip":
            tl.idle_frames_sent += 1
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame.pts, frame.time_base = self.index * (90000 // tl.fps), Fraction(1, 90000)
        self.index += 1
        return frame


class Renderer:
    """Client for one FlashHead session: audio chunk in, RGB frames out."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.session_id = None
        self.spec = None
        self.chunk_seconds = deque(maxlen=8)
        self.first_chunk_seconds = None

    async def open(self, image_path: Path) -> dict:
        response = await self.client.post(f"{RENDERER}/sessions", json={"image_path": str(image_path), "seed": 42}, timeout=120)
        response.raise_for_status()
        info = response.json()
        self.session_id = info["session_id"]
        self.spec = info
        return info

    @property
    def samples(self) -> int:
        return int(self.spec["samples_per_chunk"])

    @property
    def seconds(self) -> float:
        return float(self.spec["seconds_per_chunk"])

    @property
    def ratio(self) -> float:
        """Seconds of video produced per wall second (r), from recent chunks; 0.8 until measured."""
        if not self.chunk_seconds:
            return 0.8
        return self.seconds / (sum(self.chunk_seconds) / len(self.chunk_seconds))

    async def render(self, pcm16k: np.ndarray, reset: bool, generation: int) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        response = await self.client.post(
            f"{RENDERER}/sessions/{self.session_id}/render",
            json={"pcm_b64": base64.b64encode(np.ascontiguousarray(pcm16k.astype(np.int16)).tobytes()).decode(),
                  "reset": reset, "generation": generation},
            timeout=RENDER_TIMEOUT)
        response.raise_for_status()
        elapsed = time.monotonic() - started
        count, height, width = (int(response.headers[k]) for k in ("X-Frames", "X-Height", "X-Width"))
        frames = np.frombuffer(response.content, dtype=np.uint8).reshape(count, height, width, 3) if count else np.zeros((0, height, width, 3), np.uint8)
        self.chunk_seconds.append(elapsed)
        if self.first_chunk_seconds is None:
            self.first_chunk_seconds = elapsed
        return frames, elapsed

    async def reset(self) -> None:
        if self.session_id:
            try:
                await self.client.post(f"{RENDERER}/sessions/{self.session_id}/reset", timeout=30)
            except httpx.HTTPError:
                pass

    async def close(self) -> None:
        if self.session_id:
            try:
                await self.client.delete(f"{RENDERER}/sessions/{self.session_id}", timeout=10)
            except httpx.HTTPError:
                pass


class Assistant:
    def __init__(self, identifier, directory: Path, still, language, voice, persona_id=None):
        self.id, self.directory, self.language, self.voice = identifier, directory, language, voice
        self.persona_id = persona_id
        self.timeline = Timeline(FPS, still=still)
        self.pc = None
        self.channel = None
        self.consumer = None
        self.turn = None
        self.background = None
        self.monitor_task = None
        self.last_playout_start = -10.0
        self.expiry = None
        self.closed = False
        self.owner = "local"
        self.history: list = []
        self.ack = None                       # (audio48k, frames)
        self.client = httpx.AsyncClient(headers={"x-internal-key": os.getenv("INTERNAL_API_KEY", "")})
        self.renderer = Renderer(self.client)
        self.metrics = {"turns": 0, "interruptions": 0, "renderer_errors": 0, "prepare": {},
                        "ack_start_seconds": [], "reply_start_seconds": [], "head_start_seconds": [],
                        "reply_seconds": [], "render_ratio": [], "stalls": 0}

    def notify(self, kind, **values):
        if self.channel and self.channel.readyState == "open":
            self.channel.send(json.dumps({"type": kind, **values}))

    # ------------------------------------------------------------- session setup
    async def speak(self, text: str) -> np.ndarray:
        """Synthesize `text` with the session voice; PCM16 at 24 kHz."""
        parts = []
        async with self.client.stream("POST", f"{BACKEND}/assistant/speak",
                                      json={"text": text, "language": self.language, "voice": self.voice,
                                            "persona_id": self.persona_id, "verify": True},
                                      headers={"x-owner-id": self.owner}, timeout=120) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "audio":
                    parts.append(np.frombuffer(base64.b64decode(event["pcm_b64"]), dtype=np.int16))
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])
        return np.concatenate(parts) if parts else np.zeros(0, np.int16)

    async def render_all(self, pcm16k: np.ndarray, reset: bool = True) -> np.ndarray:
        """Render a whole clip now (idle loop, acknowledgement); frames trimmed to the audio."""
        frames = []
        step = self.renderer.samples
        for index, start in enumerate(range(0, max(len(pcm16k), 1), step)):
            chunk = pcm16k[start: start + step]
            rendered, _ = await self.renderer.render(chunk, reset and index == 0, self.timeline.generation)
            frames.append(rendered)
        return np.concatenate(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)

    def _cache_dir(self) -> Path | None:
        if not self.persona_id:
            return None
        return CACHE_DIR / self.persona_id / "assistant-cache"

    def _cache_file(self, kind: str) -> Path | None:
        directory = self._cache_dir()
        if directory is None:
            return None
        # Keyed by the portrait's content so a re-cropped or replaced portrait is re-rendered.
        import hashlib
        digest = hashlib.sha256((self.directory / "image.png").read_bytes()).hexdigest()[:12]
        return directory / f"{kind}-{self.language}-{self.voice or 'persona'}-{digest}.npz"

    async def _save_cache(self, kind: str, **arrays) -> None:
        path = self._cache_file(kind)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        await asyncio.to_thread(np.savez, str(tmp), **arrays)
        tmp.rename(path)

    async def prepare(self) -> dict:
        """Open the renderer and get a face on screen fast: the cached idle loop and
        acknowledgement when this persona has been used before, otherwise two chunks
        of idle motion now and the rest in the background."""
        started = time.monotonic()
        info = await self.renderer.open(self.directory / "image.png")
        t_open = time.monotonic()
        cached = {"idle": False, "ack": False}
        for kind in ("idle", "ack"):
            path = self._cache_file(kind)
            if path and path.exists():
                try:
                    data = await asyncio.to_thread(np.load, str(path))
                    if kind == "idle":
                        self.timeline.idle_frames = data["idle"]
                    else:
                        self.ack = (data["audio48"], data["frames"])
                    cached[kind] = True
                except Exception:
                    pass
        if not cached["idle"]:
            self.timeline.idle_frames = await self.render_all(np.zeros(self.renderer.samples * IDLE_CHUNKS, np.int16))
        self.metrics["prepare"] = {
            "cached": cached, "renderer_open_seconds": round(t_open - started, 2),
            "idle_seconds_ready": round(len(self.timeline.idle_frames) / FPS, 2),
            "ack_ready": self.ack is not None,
            "chunk": {k: info[k] for k in ("frames_per_chunk", "samples_per_chunk", "seconds_per_chunk")},
            "total_seconds": round(time.monotonic() - started, 2)}
        if not (cached["idle"] and cached["ack"]):
            self.background = asyncio.create_task(self.finish_prepare(grow_idle=not cached["idle"], make_ack=not cached["ack"]))
        return self.metrics["prepare"]

    def turn_active(self) -> bool:
        return self.turn is not None and not self.turn.done()

    async def finish_prepare(self, grow_idle: bool = True, make_ack: bool = True) -> None:
        """Grow the idle loop to IDLE_SECONDS (continuing the motion) and render the
        acknowledgement, yielding to turns; cache each as soon as it is complete."""
        try:
            if grow_idle:
                target = int(IDLE_SECONDS * FPS)
                continuous = True                   # motion state still follows the last idle chunk
                while len(self.timeline.idle_frames) < target and not self.closed:
                    if self.turn_active():
                        continuous = False          # a reply moved the renderer's motion state
                        await asyncio.sleep(0.5)
                        continue
                    frames, _ = await self.renderer.render(np.zeros(self.renderer.samples, np.int16), not continuous, self.timeline.generation)
                    continuous = True
                    if len(frames):
                        self.timeline.idle_frames = np.concatenate([self.timeline.idle_frames, frames])
                if not self.closed:
                    await self._save_cache("idle", idle=self.timeline.idle_frames)
                    self.metrics["prepare"]["idle_seconds_final"] = round(len(self.timeline.idle_frames) / FPS, 2)
            if make_ack:
                while self.turn_active() and not self.closed:
                    await asyncio.sleep(0.5)
                if self.closed:
                    return
                t_ack = time.monotonic()
                ack_pcm = await self.speak(ack_text(self.language))
                ack_frames = await self.render_all(resample(ack_pcm, 24000, 16000))
                self.ack = (resample(ack_pcm, 24000, 48000), ack_frames)
                self.metrics["prepare"].update({"ack_ready": True, "ack_seconds": round(time.monotonic() - t_ack, 2),
                                                "ack_audio_seconds": round(len(self.ack[0]) / 48000, 2)})
                await self._save_cache("ack", audio48=self.ack[0], frames=self.ack[1])
            self.notify("ready", idle_seconds=round(len(self.timeline.idle_frames) / FPS, 1))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics["renderer_errors"] += 1
            self.notify("error", message=f"background preparation failed: {exc}")

    # ------------------------------------------------------------------- turns
    def interrupt(self):
        self.metrics["interruptions"] += 1
        self.timeline.interrupt()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        asyncio.create_task(self.renderer.reset())
        self.notify("listening")

    async def reply(self, samples: np.ndarray, generation: int):
        tl = self.timeline
        t0 = tl.now()
        self.metrics["turns"] += 1
        turn_dir = self.directory / uuid.uuid4().hex
        turn_dir.mkdir()
        path = turn_dir / "input.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(samples.astype(np.int16).tobytes())
        self.notify("thinking")

        # 1. The prepared acknowledgement lands immediately (when this persona's is ready).
        ack_end = tl.now() + 0.2
        if self.ack is None:
            self.notify("acknowledging", seconds=0, pending=True)
        if self.ack is not None:
            placed = tl.schedule(tl.now() + 0.1, self.ack[0], self.ack[1], generation)
            if placed:
                ack_end = placed[1]
                self.last_playout_start = placed[0]
                self.metrics["ack_start_seconds"].append(round(placed[0] - t0, 2))
                self.notify("acknowledging", seconds=round(placed[1] - placed[0], 2))

        # 2. Plan then speak on the backend; audio arrives as PCM events.
        queue: asyncio.Queue = asyncio.Queue()
        reply_text = {"text": ""}

        async def receive():
            try:
                async with self.client.stream(
                    "POST", f"{BACKEND}/assistant/respond",
                    json={"audio_path": str(path), "language": self.language, "voice": self.voice,
                          "persona_id": self.persona_id,
                          "history": [m for m in self.history[-12:] if m.get("content")]},
                    headers={"x-owner-id": self.owner}, timeout=120,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or generation != tl.generation:
                            continue
                        event = json.loads(line)
                        kind = event["type"]
                        if kind == "transcript":
                            self.notify("transcript", text=event["text"])
                            self.history += [{"role": "user", "content": event["text"]}]
                        elif kind == "reply":
                            reply_text["text"] = event["text"]
                            self.notify("reply", text=event["text"])
                        elif kind == "audio":
                            await queue.put(np.frombuffer(base64.b64decode(event["pcm_b64"]), dtype=np.int16))
                        elif kind == "error":
                            raise RuntimeError(event["message"])
                await queue.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put(exc)

        producer = asyncio.create_task(receive())
        pending24 = np.zeros(0, np.int16)       # TTS audio not yet rendered
        rendered: list = []                      # (frames, audio48k) waiting for a start time
        state = {"reply_start": None, "offset": 0.0, "total": 0, "tts_done": False, "chunks": 0}
        need = self.renderer.samples * 24000 // 16000      # 24 kHz samples per renderer chunk

        def take(item):
            nonlocal pending24
            if item is None:
                state["tts_done"] = True
            elif isinstance(item, Exception):
                raise item
            else:
                pending24 = np.concatenate([pending24, item])
                state["total"] += len(item)

        def decide_start():
            """Pick the reply start once: at the default head start, pushed later only when
            the measured render rate cannot sustain a reply of this length."""
            total_seconds = state["total"] / 24000
            words = len(reply_text["text"].split())
            reply_seconds = total_seconds if state["tts_done"] else max(total_seconds, words / 2.5)
            required = head_start_required(reply_seconds, self.renderer.ratio, self.renderer.first_chunk_seconds or 1.5)
            start = max(ack_end + 0.3, t0 + min(MAX_HEAD_START, max(HEAD_START, required)))
            # A late decision (slow transcript or first chunk) must not schedule into the
            # past, or the opening of the reply would be skipped.
            start = max(start, tl.now() + 0.15)
            state["reply_start"] = start
            tl.promise(start, start + reply_seconds, generation)
            self.metrics["reply_start_seconds"].append(round(start - t0, 2))
            self.metrics["head_start_seconds"].append(round(required, 2))
            self.metrics["render_ratio"].append(round(self.renderer.ratio, 3))
            self.notify("reply_scheduled", start_in=round(start - tl.now(), 2), head_start=round(start - t0, 2),
                        required=round(required, 2), render_ratio=round(self.renderer.ratio, 3),
                        reply_seconds=round(reply_seconds, 2))

            async def announce():
                await asyncio.sleep(max(0.0, start - tl.now()))
                if generation == tl.generation:
                    self.last_playout_start = tl.now()
                    self.notify("speaking")
            asyncio.create_task(announce())

        def place_rendered():
            while rendered:
                frames, audio48 = rendered.pop(0)
                tl.schedule(state["reply_start"] + state["offset"], audio48, frames, generation)
                state["offset"] += len(audio48) / 48000

        try:
            while True:
                while not state["tts_done"]:
                    try:
                        take(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if len(pending24) < need and not (state["tts_done"] and len(pending24)):
                    if state["tts_done"]:
                        break
                    take(await queue.get())
                    continue
                piece, pending24 = pending24[:need], pending24[need:]
                frames, _ = await self.renderer.render(resample(piece, 24000, 16000), state["chunks"] == 0, generation)
                if generation != tl.generation:
                    return
                state["chunks"] += 1
                covered = int(round(len(frames) / FPS * 48000))
                rendered.append((frames, resample(piece, 24000, 48000)[:covered]))
                if state["reply_start"] is None and (state["tts_done"] or tl.now() >= t0 + HEAD_START - 2.0):
                    decide_start()
                if state["reply_start"] is not None:
                    place_rendered()
            if rendered:
                if state["reply_start"] is None:
                    decide_start()
                place_rendered()
            if reply_text["text"]:
                self.history += [{"role": "assistant", "content": reply_text["text"]}]
                self.history = self.history[-12:]
            self.metrics["reply_seconds"].append(round(state["total"] / 24000, 2))
            end = tl.last_end()
            while tl.now() < end and generation == tl.generation:
                await asyncio.sleep(0.1)
            if generation == tl.generation:
                self.metrics["stalls"] = tl.stalls
                self.notify("listening", stalls=tl.stalls, frames_sent=tl.frames_sent)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics["renderer_errors"] += 1
            self.notify("error", message=str(exc))
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def consume(self, track):
        from .vad import Detector

        try:
            detector = Detector()
        except Exception:
            self.notify("error", message="VAD configuration unavailable; check server readiness.")
            await self.close()
            return
        endpoint = float(os.getenv("AVATAR_ENDPOINT_SECONDS", "0.6"))
        # Echo guard: while the assistant is audible, the microphone also hears it
        # through the speakers. Speech then has to be sustained (and louder than the
        # recent echo level) before it counts as the user talking over the reply.
        barge_in = float(os.getenv("ASSISTANT_BARGE_IN_SECONDS", "0.6"))
        barge_in_gain = float(os.getenv("ASSISTANT_BARGE_IN_GAIN", "2.0"))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        preroll = deque(maxlen=10)
        utterance, silence, total, active = [], 0.0, 0.0, False
        candidate, echo_level = 0.0, 0.0
        try:
            while True:
                frame = await track.recv()
                for audio_frame in resampler.resample(frame):
                    samples = audio_frame.to_ndarray().reshape(-1).copy()
                    duration = len(samples) / 16000
                    voiced = detector.speech(samples)
                    level = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                    speaking = self.timeline.active(self.timeline.now()) is not None
                    if speaking and not active and self.timeline.now() - self.last_playout_start < 0.4:
                        preroll.append(samples)
                        continue
                    if speaking and not active:
                        # Track how loud the echo is; only sustained, clearly louder sound starts a turn.
                        if not voiced:
                            echo_level = 0.9 * echo_level + 0.1 * level
                            candidate = 0.0
                        else:
                            echo_level = max(echo_level, 0.0)
                            candidate = candidate + duration if level > barge_in_gain * max(echo_level, 50.0) else 0.0
                        if candidate < barge_in:
                            preroll.append(samples)
                            continue
                        self.metrics["barge_ins"] = self.metrics.get("barge_ins", 0) + 1
                    if voiced and not active:
                        self.interrupt()
                        utterance = list(preroll)
                        total = sum(len(x) for x in utterance) / 16000
                        active = True
                        candidate = 0.0
                    if active:
                        utterance.append(samples)
                        total += duration
                        silence = 0 if voiced else silence + duration
                        if silence >= endpoint or total >= 15:
                            if total - silence >= 0.2:
                                self.turn = asyncio.create_task(self.reply(np.concatenate(utterance), self.timeline.generation))
                            active, utterance, total, silence = False, [], 0.0, 0.0
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
        self.timeline.interrupt()
        current = asyncio.current_task()
        tasks = [t for t in (self.consumer, self.turn, self.expiry, self.background, self.monitor_task) if t and t is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.renderer.close()
        await self.client.aclose()
        if self.pc:
            await self.pc.close()
        _sessions.pop(self.id, None)

    async def rtc_stats(self) -> dict:
        """Sender-side view of the video path: what we sent and what the browser reported back."""
        out: dict = {}
        if not self.pc:
            return out
        for sender in self.pc.getSenders():
            if sender.track is None:
                continue
            try:
                report = await sender.getStats()
            except Exception:
                continue
            entry: dict = {}
            for stat in report.values():
                if stat.type == "outbound-rtp":
                    entry.update(packets_sent=stat.packetsSent, bytes_sent=stat.bytesSent)
                elif stat.type == "remote-inbound-rtp":
                    entry.update(packets_lost=stat.packetsLost, fraction_lost=round(float(stat.fractionLost), 4),
                                 jitter_ms=round(float(stat.jitter) * 1000, 1), rtt_ms=round(float(stat.roundTripTime) * 1000, 1))
            out[sender.track.kind] = entry
        return out

    async def monitor(self) -> None:
        """Log the transport and timeline every 10 s while connected (operators read docker logs)."""
        try:
            while not self.closed and self.pc and self.pc.connectionState != "closed":
                await asyncio.sleep(10)
                stats = await self.rtc_stats()
                tl = self.timeline
                print(json.dumps({"event": "assistant_rtc", "session": self.id, "connection": self.pc.connectionState if self.pc else None,
                                  "rtc": stats, "frames_sent": tl.frames_sent, "frames_skipped": tl.frames_skipped,
                                  "stalls": tl.stalls, "now": round(tl.now(), 1)}), flush=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def status(self) -> dict:
        tl = self.timeline
        return {"metrics": {**self.metrics, "stalls": tl.stalls}, "history": self.history, "generation": tl.generation,
                "timeline": {"frames_sent": tl.frames_sent, "frames_skipped": tl.frames_skipped,
                             "idle_frames_sent": tl.idle_frames_sent, "scheduled_seconds": round(tl.scheduled_seconds, 2),
                             "now": round(tl.now(), 2)},
                "renderer": {"session": self.renderer.session_id, "ratio": round(self.renderer.ratio, 3),
                             "first_chunk_seconds": self.renderer.first_chunk_seconds,
                             "chunk_seconds": [round(s, 3) for s in self.renderer.chunk_seconds]}}


# ------------------------------------------------------------------- endpoints
async def require_warm():
    if os.getenv("ASSISTANT_REQUIRE_WARM", "1") != "1":
        return
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            health = await client.get(f"{RENDERER}/health")
            health.raise_for_status()
            if health.json().get("status") == "ok":
                return
    except (httpx.HTTPError, ValueError):
        pass
    raise HTTPException(503, "The renderer is not ready yet; retry shortly.", headers={"Retry-After": "5"})


async def _persona_portrait(persona_id: str, owner_id: str) -> bytes:
    """The persona's portrait as the backend recorded it (same /jobs volume)."""
    async with httpx.AsyncClient(timeout=10, headers={"x-internal-key": os.getenv("INTERNAL_API_KEY", ""),
                                                      "x-owner-id": owner_id}) as client:
        response = await client.get(f"{BACKEND}/personas/{persona_id}")
        if response.status_code == 404:
            raise HTTPException(404, "persona not found")
        response.raise_for_status()
        record = response.json()
    if record.get("status") != "ready":
        raise HTTPException(409, "persona is not ready")
    portrait = Path(record["files"].get("portrait_render") or record["files"]["portrait"])
    if not portrait.is_relative_to(ROOT.parent) or not portrait.exists():
        raise HTTPException(502, "persona portrait is not visible to the media service")
    return portrait.read_bytes()


@router.post("/sessions")
async def create(image: UploadFile | None = File(None), language: str = Form("en"), voice: str | None = Form(None),
                 persona_id: str | None = Form(None)):
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, "assistant capacity reached")
    await require_warm()
    if language not in {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "hu", "ko", "ja", "hi", "zh"}:
        raise HTTPException(400, "unsupported language")
    if persona_id and not __import__("re").fullmatch(r"[0-9a-f]{32}", persona_id):
        raise HTTPException(400, "invalid persona id")
    if persona_id:
        payload = await _persona_portrait(persona_id, owner.get())
    elif image is not None:
        payload = await image.read(5 * 1024 * 1024 + 1)
        if len(payload) > 5 * 1024 * 1024:
            raise HTTPException(413, "image exceeds 5 MB")
    else:
        raise HTTPException(400, "upload a portrait or choose a persona")
    original = payload
    enhanced = False
    if os.getenv("ASSISTANT_ENHANCE_PORTRAIT", "1") == "1":
        # Restore the still once before it conditions the renderer (see /portrait/enhance).
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{RENDERER}/portrait/enhance", json={
                    "image_b64": base64.b64encode(payload).decode(),
                    "fidelity": float(os.getenv("ASSISTANT_ENHANCE_FIDELITY", "0.7")),
                    "blend": float(os.getenv("ASSISTANT_ENHANCE_BLEND", "0.85"))})
            if response.status_code == 200 and response.content:
                payload = response.content
                enhanced = True
        except httpx.HTTPError:
            pass
    try:
        picture = Image.open(BytesIO(payload))
        if picture.width * picture.height > 20_000_000:
            raise ValueError("image too large")
        picture = picture.convert("RGB")
        picture.thumbnail((1024, 1024))
    except Exception as exc:
        raise HTTPException(422, "upload a valid portrait image") from exc
    identifier = uuid.uuid4().hex
    directory = ROOT / identifier
    directory.mkdir(parents=True)
    picture.save(directory / "image.png")
    if enhanced:
        (directory / "image-original.png").write_bytes(original)
    still = np.asarray(picture.resize((512, 512))).copy()
    session = Assistant(identifier, directory, still, language, voice if isinstance(voice, str) and voice else None,
                        persona_id or None)
    session.owner = owner.get()
    (directory / "owner.json").write_text(json.dumps({"owner_id": session.owner}))
    _sessions[identifier] = session
    session.metrics["portrait_enhanced"] = enhanced
    try:
        prepared = await session.prepare()
    except Exception as exc:
        await session.close()
        raise HTTPException(502, f"could not prepare the assistant: {type(exc).__name__}: {exc}")

    async def expire():
        await asyncio.sleep(1800)
        await session.close()

    session.expiry = asyncio.create_task(expire())
    return {"session_id": identifier, "prepare": prepared}


class Offer(BaseModel):
    sdp: str = Field(..., max_length=65536)
    type: str = Field("offer", pattern="^offer$")


@router.post("/sessions/{identifier}/offer")
async def offer(identifier: str, body: Offer):
    from .main import rtc_configuration
    from .gpu_media import prefer_h264

    session = _sessions.get(identifier)
    require(session)
    if session.pc:
        raise HTTPException(409, "assistant already connected")
    pc = session.pc = RTCPeerConnection(rtc_configuration())
    pc.addTrack(AudioOut(session.timeline))
    pc.addTrack(VideoOut(session.timeline))
    prefer_h264(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            session.consumer = asyncio.create_task(session.consume(track))

    @pc.on("datachannel")
    def on_channel(channel):
        session.channel = channel
        if session.ack is not None and (session.background is None or session.background.done()):
            # Everything was cached: tell the client now that the channel exists.
            session.notify("ready", idle_seconds=round(len(session.timeline.idle_frames) / FPS, 1), cached=True)

        @channel.on("message")
        def message(value):
            if value == "interrupt":
                session.interrupt()

    @pc.on("connectionstatechange")
    async def changed():
        if pc.connectionState == "connected" and session.monitor_task is None:
            session.monitor_task = asyncio.create_task(session.monitor())
            print(json.dumps({"event": "assistant_connected", "session": session.id,
                              "video_codec": next((t.sender._rtp_codec.mimeType if getattr(t.sender, "_rtp_codec", None) else None
                                                   for t in pc.getTransceivers() if t.kind == "video"), None)}), flush=True)
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


@router.get("/sessions/{identifier}")
async def status(identifier: str):
    session = require(_sessions.get(identifier))
    return {**session.status(), "rtc": await session.rtc_stats()}


@router.on_event("shutdown")
async def shutdown():
    await asyncio.gather(*(s.close() for s in list(_sessions.values())))
