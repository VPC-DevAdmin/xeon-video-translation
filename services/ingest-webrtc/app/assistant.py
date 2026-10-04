"""Video assistant sessions over WebRTC.

Turn: Silero endpoint -> a prepared opener plays at once ("I am going to look that up,
give me a second") -> the persona turns to its tablet (a second, posed portrait) and
makes short progress utterances between pauses -> the backend plans the whole reply
text and streams TTS audio -> FlashHead renders it chunk by chunk -> the reply is
scheduled at the earliest start that cannot stall given the renderer's measured rate
-> whatever filler is still talking is cut off with a fade, a short closer plays, the
persona turns back and the reply plays at 25 fps. Speaking again or pressing
Interrupt clears everything.

Services (ingest runs with host networking):
  ASSISTANT_BACKEND_URL   backend with /assistant/respond and /assistant/speak
  ASSISTANT_RENDERER_URL  FlashHead render service (services/flashhead), also /portrait/pose
Tuning: ASSISTANT_HEAD_START (minimum reply start after the endpoint, s, default 7),
ASSISTANT_MAX_HEAD_START (20), ASSISTANT_IDLE_CHUNKS (2), ASSISTANT_IDLE_SECONDS (12),
ASSISTANT_WORKING_IDLE_SECONDS (6), ASSISTANT_WORKING_POSE ("pitch,yaw,roll,eyes_x,eyes_y"),
ASSISTANT_<KIND>_TEXT_<LANG> ("|"-separated phrases; kinds OPENER, BEAT, BRIDGE, CLOSER),
ASSISTANT_RENDER_TIMEOUT (30).

Every renderer call of a session goes through one lock (`Renderer.lock`): idle growth,
filler clips and reply chunks never interleave, and the renderer remembers which
footage its motion state continues (`Renderer.motion`) so idle growth knows whether
to extend the current segment or start a new one.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import os
import random
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
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
HEAD_START = float(os.getenv("ASSISTANT_HEAD_START", "7"))             # the reply never starts earlier than this
MAX_HEAD_START = float(os.getenv("ASSISTANT_MAX_HEAD_START", "20"))
IDLE_CHUNKS = int(os.getenv("ASSISTANT_IDLE_CHUNKS", "2"))          # rendered before the session answers
IDLE_SECONDS = float(os.getenv("ASSISTANT_IDLE_SECONDS", "12"))      # grown to this in the background, then looped
WORKING_IDLE_SECONDS = float(os.getenv("ASSISTANT_WORKING_IDLE_SECONDS", "6"))
_POSE_VALUES = [float(v) for v in os.getenv("ASSISTANT_WORKING_POSE", "14,-14,0,-6,10").split(",")]
WORKING_POSE = dict(zip(("pitch", "yaw", "roll", "eyes_x", "eyes_y"), _POSE_VALUES + [0.0] * 5))
WORKING_POSE_ENABLED = os.getenv("ASSISTANT_WORKING_POSE_ENABLED", "1") == "1"
CACHE_DIR = Path(os.getenv("JOB_ARTIFACTS_DIR", "./jobs")).resolve() / "personas"
CACHE_VERSION = 3                                                    # bump when cached footage changes meaning
TURN_FRAMES = int(os.getenv("ASSISTANT_TURN_FRAMES", "12"))          # head turn to/from the tablet, at 25 fps
RENDER_TIMEOUT = float(os.getenv("ASSISTANT_RENDER_TIMEOUT", "30"))
FPS = 25

# Filler repertoire. Openers play the moment the user stops; beats are short progress
# utterances and bridges longer ones, both said while looking at the tablet; closers
# bring the gaze back just before the reply. Every phrase is at least three words so
# the speech verifier can recognize it (XTTS babbles on one-word prompts).
FILLERS = {
    "en": {
        "opener": ["Let me think about that for a second.", "Good question, give me a moment.",
                   "I am going to look that up, please give me a second.", "Sure, let me check on that for you.",
                   "Hold on, let me find that for you."],
        "beat": ["Hmm, let me see.", "Okay, almost there.", "Right, one second.", "Mm-hmm, getting closer.",
                 "Okay, nearly there.", "Let's see here."],
        "bridge": ["I'm pulling that up now, it should only take a moment.", "Bear with me, I want to make sure I get this right.",
                   "I'm checking a couple of things so I give you a proper answer.", "Still looking, this one deserves a careful answer.",
                   "Just making sure I have the details straight."],
        "closer": ["Okay, got it.", "Alright, here we go.", "Right, here's what I have.", "Okay, so."],
    },
    "es": {
        "opener": ["Déjame pensarlo un momento.", "Buena pregunta, dame un segundo.", "Voy a buscarlo, dame un segundo por favor.",
                   "Claro, déjame comprobarlo."],
        "beat": ["A ver, un momento.", "Vale, casi está.", "Mm, ya casi.", "Un segundo más."],
        "bridge": ["Lo estoy buscando ahora, solo tardará un momento.", "Ten paciencia, quiero asegurarme de que sea correcto.",
                   "Estoy comprobando un par de cosas para darte una buena respuesta."],
        "closer": ["Vale, ya lo tengo.", "Bien, aquí está.", "Listo, esto es lo que tengo."],
    },
    "fr": {
        "opener": ["Laissez-moi réfléchir un instant.", "Bonne question, donnez-moi un moment.", "Je vais chercher ça, une seconde s'il vous plaît.",
                   "Bien sûr, laissez-moi vérifier."],
        "beat": ["Voyons voir, un instant.", "D'accord, presque fini.", "Hmm, j'y suis presque.", "Encore une seconde."],
        "bridge": ["Je cherche ça maintenant, ça ne prendra qu'un instant.", "Un peu de patience, je veux être sûr de bien répondre.",
                   "Je vérifie deux ou trois choses pour vous répondre correctement."],
        "closer": ["Voilà, je l'ai.", "Bon, c'est parti.", "D'accord, voici ce que j'ai."],
    },
    "de": {
        "opener": ["Lass mich kurz nachdenken.", "Gute Frage, einen Moment bitte.", "Das schaue ich kurz nach, einen Moment bitte.",
                   "Klar, lass mich das prüfen."],
        "beat": ["Mal sehen, einen Moment.", "Okay, fast fertig.", "Hm, gleich hab ich es.", "Noch eine Sekunde."],
        "bridge": ["Ich rufe das gerade auf, es dauert nur einen Moment.", "Einen Augenblick, ich will sichergehen, dass es stimmt.",
                   "Ich prüfe noch zwei Dinge, damit die Antwort passt."],
        "closer": ["Okay, hab es.", "Gut, hier ist es.", "Also, das habe ich gefunden."],
    },
    "it": {"opener": ["Fammi pensare un attimo.", "Bella domanda, dammi un momento.", "Certo, un attimo che controllo."]},
    "pt": {"opener": ["Deixe-me pensar um instante.", "Boa pergunta, me dê um momento.", "Claro, um momento enquanto verifico."]},
    "zh": {"opener": ["让我想一想。", "好问题，请稍等一下。"]},
    "ja": {"opener": ["少し考えさせてください。", "いい質問ですね、少々お待ちください。"]},
}
KIND_POSE = {"opener": "front", "beat": "working", "bridge": "working", "closer": "front"}


def filler_texts(language: str, kind: str) -> list[str]:
    """Phrases of one kind for a language; env overrides win, English openers are the
    last resort so every language has at least an acknowledgement."""
    override = os.getenv(f"ASSISTANT_{kind.upper()}_TEXT_{language.upper()}") or (
        os.getenv(f"ASSISTANT_ACK_TEXT_{language.upper()}") if kind == "opener" else None)
    if override:
        return [t.strip() for t in override.split("|") if t.strip()]
    table = FILLERS.get(language) or {}
    texts = list(table.get(kind, []))
    if not texts and kind == "opener":
        texts = list(FILLERS["en"]["opener"])
    return texts


def ack_texts(language: str) -> list[str]:
    return filler_texts(language, "opener")


def ack_text(language: str) -> str:
    return ack_texts(language)[0]


@dataclass
class Clip:
    kind: str
    text: str
    audio48: np.ndarray
    frames: np.ndarray
    pose: str

    @property
    def seconds(self) -> float:
        return max(len(self.audio48) / 48000, len(self.frames) / FPS)


class PcmQueue:
    """PCM16 pieces awaiting the renderer; pops whole renderer chunks without
    re-concatenating everything that is still pending."""

    def __init__(self):
        self.parts: deque = deque()
        self.length = 0

    def push(self, samples: np.ndarray) -> None:
        if len(samples):
            self.parts.append(samples)
            self.length += len(samples)

    def __len__(self) -> int:
        return self.length

    def pop(self, count: int) -> np.ndarray:
        out, taken = [], 0
        while self.parts and taken < count:
            part = self.parts.popleft()
            room = count - taken
            if len(part) > room:
                self.parts.appendleft(part[room:])
                part = part[:room]
            out.append(part)
            taken += len(part)
        self.length -= taken
        return np.concatenate(out) if out else np.zeros(0, np.int16)


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
    """Client for the FlashHead sessions of one assistant: audio chunk in, RGB frames out.
    One renderer session per portrait pose ("front" listening/talking, "working" looking
    at the tablet); the service switches portraits in about 0.2 s."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.sessions: dict[str, str] = {}
        self.spec = None
        self.model: dict = {}
        self.chunk_seconds = deque(maxlen=8)
        self.first_chunk_seconds = None
        self.lock = asyncio.Lock()        # one request per assistant at a time, in call order
        self.motion = None                # footage the renderer's motion state continues, e.g. "idle:front", "reply:front"

    @property
    def session_id(self):
        return self.sessions.get("front")

    async def open(self, image_path: Path, pose: str = "front") -> dict:
        if not self.model:
            try:
                health = await self.client.get(f"{RENDERER}/health", timeout=5)
                self.model = {k: health.json().get(k) for k in ("model", "compile")}
            except (httpx.HTTPError, ValueError):
                self.model = {}
        response = await self.client.post(f"{RENDERER}/sessions", json={"image_path": str(image_path), "seed": 42}, timeout=120)
        response.raise_for_status()
        info = response.json()
        self.sessions[pose] = info["session_id"]
        if pose == "front" or self.spec is None:
            self.spec = info
        self.motion = None
        return info

    def has_pose(self, pose: str) -> bool:
        return pose in self.sessions

    def fingerprint(self) -> dict:
        return {**self.model, "chunk": {k: self.spec.get(k) for k in ("frames_per_chunk", "samples_per_chunk", "seconds_per_chunk")}}

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

    async def render(self, pcm16k: np.ndarray, reset: bool, generation: int, motion: str = "reply:front",
                     pose: str = "front") -> tuple[np.ndarray, float]:
        async with self.lock:
            return await self.render_locked(pcm16k, reset, generation, motion, pose)

    async def render_locked(self, pcm16k: np.ndarray, reset: bool, generation: int, motion: str,
                            pose: str = "front") -> tuple[np.ndarray, float]:
        """Render one chunk; the caller holds `self.lock`."""
        session_id = self.sessions.get(pose) or self.sessions["front"]
        started = time.monotonic()
        response = await self.client.post(
            f"{RENDERER}/sessions/{session_id}/render",
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
        self.motion = motion
        return frames, elapsed

    async def reset(self) -> None:
        if self.session_id:
            async with self.lock:
                try:
                    await self.client.post(f"{RENDERER}/sessions/{self.session_id}/reset", timeout=30)
                except httpx.HTTPError:
                    pass
                self.motion = None

    async def close(self) -> None:
        for session_id in list(self.sessions.values()):
            try:
                await self.client.delete(f"{RENDERER}/sessions/{session_id}", timeout=10)
            except httpx.HTTPError:
                pass
        self.sessions.clear()


class Assistant:
    def __init__(self, identifier, directory: Path, still, language, voice, persona_id=None,
                 portrait_digest: str = "", voice_digest: str = "", enhance: dict | None = None):
        self.id, self.directory, self.language, self.voice = identifier, directory, language, voice
        self.persona_id = persona_id
        self.portrait_digest = portrait_digest          # of the portrait as uploaded, before any restoration
        self.voice_digest = voice_digest                # of the persona's voice conditioning (or the stock voice name)
        self.enhance = enhance or {}
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
        self.clips: dict[str, list[Clip]] = {kind: [] for kind in KIND_POSE}
        self.last_used: dict[str, str] = {}
        self.working_pose = False                       # a posed portrait is open on the renderer
        self.turn_down: np.ndarray | None = None        # head turning from the camera to the tablet (frames)
        self.turn_up: np.ndarray | None = None
        self.client = httpx.AsyncClient(headers={"x-internal-key": os.getenv("INTERNAL_API_KEY", "")})
        self.renderer = Renderer(self.client)
        self.metrics = {"turns": 0, "interruptions": 0, "renderer_errors": 0, "prepare": {},
                        "ack_start_seconds": [], "reply_start_seconds": [], "head_start_seconds": [],
                        "reply_seconds": [], "render_ratio": [], "stalls": 0, "fillers": []}

    def notify(self, event: str, **values):
        """Send an event to the browser over the data channel (dropped when it is not open)."""
        if self.channel and self.channel.readyState == "open":
            self.channel.send(json.dumps({"type": event, **values}))

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

    def turn_active(self) -> bool:
        return self.turn is not None and not self.turn.done()

    async def render_all(self, pcm16k: np.ndarray, motion: str, pose: str = "front") -> np.ndarray:
        """Render a whole clip (idle footage, a filler) from the portrait pose. A reply that
        takes the renderer in between would break the clip's motion, so the clip waits
        for the turn and starts over when its motion state was moved."""
        step = self.renderer.samples
        starts = list(range(0, max(len(pcm16k), 1), step))
        frames: list = []
        while len(frames) < len(starts) and not self.closed:
            if self.turn_active():
                await asyncio.sleep(0.25)
                continue
            async with self.renderer.lock:
                if self.turn_active():
                    continue
                if frames and self.renderer.motion != motion:
                    frames = []                          # someone else rendered since our last chunk: restart the clip
                index = len(frames)
                chunk = pcm16k[starts[index]: starts[index] + step]
                rendered, _ = await self.renderer.render_locked(chunk, index == 0, self.timeline.generation, motion, pose)
                frames.append(rendered)
        return np.concatenate(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)

    # ------------------------------------------------------------------- cache
    def _cache_dir(self) -> Path | None:
        if not self.persona_id:
            return None
        return CACHE_DIR / self.persona_id / "assistant-cache"

    def _fingerprint(self, **extra) -> str:
        """Cached footage is only valid for the exact inputs that produced it: the
        portrait as uploaded, the restoration settings, the renderer model and chunk
        spec, the frame rate, plus whatever the caller adds (voice, pose, phrases)."""
        parts = {"v": CACHE_VERSION, "portrait": self.portrait_digest, "enhance": self.enhance,
                 "renderer": self.renderer.fingerprint(), "fps": FPS, **extra}
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _idle_file(self, pose: str) -> Path | None:
        directory = self._cache_dir()
        if directory is None:
            return None
        seconds = IDLE_SECONDS if pose == "front" else WORKING_IDLE_SECONDS
        return directory / f"idle-{pose}-{self._fingerprint(kind='idle', pose=pose, seconds=seconds, posed=WORKING_POSE if pose != 'front' else None)}.npz"

    def _clip_prefix(self) -> str:
        return self._fingerprint(kind="clip", language=self.language, voice=self.voice or "", voice_digest=self.voice_digest,
                                 posed=WORKING_POSE if self.working_pose else None)

    def _clip_file(self, kind: str, text: str) -> Path | None:
        directory = self._cache_dir()
        if directory is None:
            return None
        return directory / "clips" / f"{self._clip_prefix()}-{hashlib.sha256(f'{kind}|{text}'.encode()).hexdigest()[:10]}.npz"

    def _working_portrait_file(self) -> Path | None:
        directory = self._cache_dir()
        if directory is None:
            return None
        return directory / f"portrait-working-{self._fingerprint(kind='portrait', posed=WORKING_POSE)}.png"

    def _turn_file(self) -> Path | None:
        directory = self._cache_dir()
        if directory is None:
            return None
        return directory / f"turn-{self._fingerprint(kind='turn', posed=WORKING_POSE, steps=TURN_FRAMES)}.npz"

    async def load_turn(self) -> bool:
        path = self._turn_file()
        if path is None or not path.exists():
            return False
        try:
            data = await asyncio.to_thread(np.load, str(path))
            self.turn_down = data["down"]
            self.turn_up = self.turn_down[::-1].copy()
            return True
        except Exception:
            return False

    async def make_turn(self) -> bool:
        """Frames of the head turning from the camera to the tablet (LivePortrait), sized
        like the renderer's frames; played forward to look down and backward to look up."""
        if not self.working_pose:
            return False
        try:
            response = await self.client.post(f"{RENDERER}/portrait/pose", json={
                "image_b64": base64.b64encode((self.directory / "image.png").read_bytes()).decode(), **WORKING_POSE,
                "steps": TURN_FRAMES, "size": int(self.renderer.spec.get("height") or 512)}, timeout=120)
            if response.status_code != 200:
                return False
            count, height, width = (int(response.headers[k]) for k in ("X-Frames", "X-Height", "X-Width"))
            self.turn_down = np.frombuffer(response.content, dtype=np.uint8).reshape(count, height, width, 3).copy()
            self.turn_up = self.turn_down[::-1].copy()
            path = self._turn_file()
            if path is not None:
                await self._save_npz(path, down=self.turn_down)
                for stale in path.parent.glob("turn-*.npz"):
                    if stale != path:
                        stale.unlink(missing_ok=True)
            return True
        except (httpx.HTTPError, ValueError, OSError) as exc:
            self.metrics["prepare"]["turn_error"] = f"{type(exc).__name__}: {exc}"
            return False

    def turn(self, at: float, direction: str, generation: int) -> float:
        """Schedule the head turn starting at `at` and switch the idle loop with it.
        Returns when the turn ends (= `at` when no turn footage exists)."""
        frames = self.turn_down if direction == "down" else self.turn_up
        mode = "working" if direction == "down" else "front"
        if frames is None or not len(frames):
            self.timeline.set_mode(at, mode, generation)
            return at
        placed = self.timeline.schedule(at, np.zeros(int(len(frames) / FPS * 48000), np.int16), frames, generation, tag="turn")
        self.timeline.set_mode(at, mode, generation)
        return placed[1] if placed else at

    async def _save_npz(self, path: Path | None, **arrays) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        await asyncio.to_thread(np.savez, str(tmp), **arrays)
        tmp.rename(path)

    async def _save_idle(self, pose: str) -> None:
        segments = self.timeline.loop(pose).segments
        path = self._idle_file(pose)
        if path is None or not segments:
            return
        await self._save_npz(path, count=len(segments), **{f"idle_{i}": seg for i, seg in enumerate(segments)})
        for stale in path.parent.glob(f"idle-{pose}-*.npz"):
            if stale != path:
                stale.unlink(missing_ok=True)
        if pose == "front":
            for legacy in path.parent.glob("idle-??-*.npz"):   # files from before per-pose naming
                legacy.unlink(missing_ok=True)

    async def _load_idle(self, pose: str) -> bool:
        path = self._idle_file(pose)
        if path is None or not path.exists():
            return False
        try:
            data = await asyncio.to_thread(np.load, str(path))
            self.timeline.loop(pose).segments = [data[f"idle_{i}"] for i in range(int(data["count"]))]
            return True
        except Exception:
            return False

    async def _save_clip(self, clip: Clip) -> None:
        path = self._clip_file(clip.kind, clip.text)
        if path is None:
            return
        await self._save_npz(path, audio48=clip.audio48, frames=clip.frames, text=np.array(clip.text),
                             kind=np.array(clip.kind), pose=np.array(clip.pose))
        prefix = self._clip_prefix()
        for stale in path.parent.glob("*.npz"):
            if not stale.name.startswith(prefix):
                stale.unlink(missing_ok=True)
        for legacy in path.parent.parent.glob("ack-*.npz"):
            legacy.unlink(missing_ok=True)

    async def _load_clips(self) -> int:
        directory = self._cache_dir()
        if directory is None or not (directory / "clips").exists():
            return 0
        prefix = self._clip_prefix()
        loaded = 0
        for path in sorted((directory / "clips").glob(f"{prefix}-*.npz")):
            try:
                data = await asyncio.to_thread(np.load, str(path))
                kind = str(data["kind"])
                if kind in self.clips:
                    self.clips[kind].append(Clip(kind, str(data["text"]), data["audio48"], data["frames"], str(data["pose"])))
                    loaded += 1
            except Exception:
                continue
        return loaded

    def has_clip(self, kind: str, text: str) -> bool:
        return any(c.text == text for c in self.clips[kind])

    def clip_count(self) -> int:
        return sum(len(v) for v in self.clips.values())

    # --------------------------------------------------------------- preparing
    async def prepare(self) -> dict:
        """Open the renderer and get a face on screen fast: cached idle footage and clips
        when this persona has been used before, otherwise two chunks of idle motion now
        and the rest in the background."""
        started = time.monotonic()
        info = await self.renderer.open(self.directory / "image.png")
        t_open = time.monotonic()
        working_file = self._working_portrait_file()
        if WORKING_POSE_ENABLED and working_file is not None and working_file.exists():
            # The posed portrait is cached: open it now so cached working clips match it.
            (self.directory / "image-working.png").write_bytes(working_file.read_bytes())
            try:
                await self.renderer.open(self.directory / "image-working.png", "working")
                self.working_pose = True
                await self.load_turn()
            except httpx.HTTPError:
                self.working_pose = False
        cached = {"idle": await self._load_idle("front"), "working_idle": self.working_pose and await self._load_idle("working"),
                  "clips": await self._load_clips()}
        if not cached["idle"]:
            self.timeline.add_idle(await self.render_all(np.zeros(self.renderer.samples * IDLE_CHUNKS, np.int16), "idle:front"), False)
        self.metrics["prepare"] = {
            "cached": cached, "renderer_open_seconds": round(t_open - started, 2),
            "idle_seconds_ready": round(self.timeline.idle_seconds(), 2),
            "ack_ready": bool(self.clips["opener"]), "clips": self.clip_count(), "working_pose": self.working_pose,
            "chunk": {k: info[k] for k in ("frames_per_chunk", "samples_per_chunk", "seconds_per_chunk")},
            "total_seconds": round(time.monotonic() - started, 2)}
        self.background = asyncio.create_task(self.finish_prepare())
        return self.metrics["prepare"]

    def _plan_texts(self) -> list[tuple[str, str]]:
        """Clips in the order the user needs them: first opener, a few beats and closers so
        a working phase can happen, bridges, then the rest of the repertoire."""
        texts = {kind: filler_texts(self.language, kind) for kind in KIND_POSE}
        order = [("opener", 0)]
        order += [("beat", i) for i in range(3)] + [("closer", i) for i in range(2)] + [("bridge", i) for i in range(2)]
        order += [("opener", i) for i in range(1, 8)] + [("beat", i) for i in range(3, 8)]
        order += [("bridge", i) for i in range(2, 8)] + [("closer", i) for i in range(2, 8)]
        plan, seen = [], set()
        for kind, index in order:
            if index < len(texts[kind]) and (kind, texts[kind][index]) not in seen:
                seen.add((kind, texts[kind][index]))
                plan.append((kind, texts[kind][index]))
        return plan

    async def make_clip(self, kind: str, text: str) -> None:
        """Synthesize and render one filler clip in the pose its kind calls for."""
        pose = KIND_POSE[kind] if (self.working_pose and self.renderer.has_pose("working")) else "front"
        pcm = await self.speak(text)
        frames = await self.render_all(resample(pcm, 24000, 16000), f"{kind}:{pose}", pose)
        if not len(frames):
            return
        clip = Clip(kind, text, resample(pcm, 24000, 48000), frames, pose)
        self.clips[kind].append(clip)
        await self._save_clip(clip)
        self.notify("filler_ready", kind=kind, count=len(self.clips[kind]), clips=self.clip_count())

    async def grow_idle(self, pose: str, target_seconds: float) -> None:
        """Grow the idle loop of `pose` to `target_seconds`, yielding to turns. Footage
        extends the current segment while the renderer's motion still follows it and
        starts a new segment after anything else rendered."""
        loop = self.timeline.loop(pose)
        target = int(target_seconds * FPS)
        motion = f"idle:{pose}"
        while loop.frame_count < target and not self.closed:
            if self.turn_active():
                await asyncio.sleep(0.5)
                continue
            async with self.renderer.lock:
                if self.turn_active():
                    continue
                continuous = self.renderer.motion == motion
                frames, _ = await self.renderer.render_locked(np.zeros(self.renderer.samples, np.int16), not continuous,
                                                              self.timeline.generation, motion, pose)
            loop.add(frames, continuous)
        if not self.closed:
            await self._save_idle(pose)

    async def open_working_pose(self) -> bool:
        """Make (or load) the portrait looking down at a tablet and open it on the renderer."""
        if not WORKING_POSE_ENABLED or self.working_pose:
            return self.working_pose
        target = self.directory / "image-working.png"
        cached = self._working_portrait_file()
        try:
            if cached is not None and cached.exists():
                target.write_bytes(cached.read_bytes())
            else:
                started = time.monotonic()
                response = await self.client.post(f"{RENDERER}/portrait/pose", json={
                    "image_b64": base64.b64encode((self.directory / "image.png").read_bytes()).decode(), **WORKING_POSE}, timeout=120)
                if response.status_code != 200:
                    self.metrics["prepare"]["working_pose_error"] = response.status_code
                    return False
                target.write_bytes(response.content)
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(response.content)
                self.metrics["prepare"]["working_pose_seconds"] = round(time.monotonic() - started, 2)
            await self.renderer.open(target, "working")
            self.working_pose = True
        except (httpx.HTTPError, OSError) as exc:
            self.metrics["prepare"]["working_pose_error"] = f"{type(exc).__name__}: {exc}"
            return False
        self.metrics["prepare"]["working_pose"] = True
        return True

    async def finish_prepare(self) -> None:
        """Background preparation, in the order the user needs it: the first opener (what
        they hear when they stop talking), the idle loop, the posed portrait and its idle
        loop, then the filler repertoire. Yields to turns; everything is cached."""
        try:
            plan = self._plan_texts()
            if plan and not self.clips["opener"]:
                kind, text = plan[0]
                await self.make_clip(kind, text)
            await self.grow_idle("front", IDLE_SECONDS)
            self.metrics["prepare"].update(idle_seconds_final=round(self.timeline.idle_seconds(), 2),
                                           idle_segments=len(self.timeline.idle_segments))
            if await self.open_working_pose():
                if self.turn_down is None:
                    await self.make_turn()
                await self.grow_idle("working", WORKING_IDLE_SECONDS)
                self.metrics["prepare"]["working_idle_seconds"] = round(self.timeline.idle_seconds("working"), 2)
            for kind, text in plan:
                if self.closed:
                    return
                if not self.has_clip(kind, text):
                    await self.make_clip(kind, text)
            self.notify("ready", idle_seconds=round(self.timeline.idle_seconds(), 1), clips=self.clip_count(),
                        working_pose=self.working_pose)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics["renderer_errors"] += 1
            self.metrics["prepare"]["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"event": "assistant_prepare_failed", "session": self.id, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            self.notify("error", message=f"background preparation failed: {exc}")

    # ------------------------------------------------------------------- turns
    def interrupt(self):
        self.metrics["interruptions"] += 1
        self.timeline.interrupt()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        self.notify("listening")

    def pick(self, kind: str) -> Clip | None:
        """A prepared clip of `kind`, not the one used last time when there is a choice."""
        clips = self.clips.get(kind) or []
        if not clips:
            return None
        choices = [c for c in clips if c.text != self.last_used.get(kind)] or clips
        clip = random.choice(choices)
        self.last_used[kind] = clip.text
        return clip

    async def fill_gap(self, plan: dict, state: dict, generation: int, t0: float) -> None:
        """Fill the wait between the opener and the reply the way a person looking
        something up would: turn to the tablet, say short progress utterances with
        pauses between them (beats and the occasional longer bridge), and when the
        reply is ready cut whatever is still being said, say a short closer facing the
        user, and hand over. Clips are scheduled just in time so the plan can adapt."""
        tl = self.timeline
        cursor = plan["opener_end"]
        pattern = itertools.cycle(["beat", "bridge", "beat", "beat", "bridge"])
        working = False
        try:
            while generation == tl.generation and not self.closed:
                start = state["reply_start"]
                if start is not None:
                    closer = self.pick("closer")
                    closer_len = closer.seconds if closer else 0.0
                    turn_len = len(self.turn_up) / FPS if (working and self.turn_up is not None) else 0.0
                    cut_at = max(plan["opener_end"], start - 0.25 - turn_len - (closer_len + 0.15 if closer else 0.0))
                    cut = tl.truncate(cut_at, "filler")
                    tl.truncate(cut_at, "turn")
                    back = self.turn(cut_at + 0.05, "up", generation) if working else cut_at
                    if closer and back + 0.1 + closer_len + 0.15 <= start:
                        tl.schedule(back + 0.1, closer.audio48, closer.frames, generation, tag="filler")
                        self.notify("filler", kind="closer", text=closer.text)
                        self.metrics["fillers"].append({"kind": "closer", "text": closer.text, "at": round(back + 0.1 - t0, 2)})
                    self.metrics["fillers"].append({"kind": "cut", "at": round(cut_at - t0, 2), "clips_cut": cut, "turn_back": working})
                    return
                now = tl.now()
                if not working and tl.loop("working").ready and now >= plan["opener_end"] - 0.5:
                    cursor = max(cursor, self.turn(plan["opener_end"] + 0.1, "down", generation))
                    working = True
                    self.notify("working")
                if cursor - now < 1.5 and cursor < t0 + MAX_HEAD_START - 1.0:
                    kind = next(pattern)
                    clip = self.pick(kind) or self.pick("beat") or self.pick("bridge")
                    if clip is not None:
                        at = max(cursor + random.uniform(0.7, 1.5), now + 0.3)
                        placed = tl.schedule(at, clip.audio48, clip.frames, generation, tag="filler")
                        if placed:
                            cursor = placed[1]
                            self.notify("filler", kind=clip.kind, text=clip.text)
                            self.metrics["fillers"].append({"kind": clip.kind, "text": clip.text, "at": round(at - t0, 2)})
                    else:
                        cursor = now + 1.0
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            import traceback
            print(json.dumps({"event": "assistant_filler_failed", "session": self.id, "error": f"{type(exc).__name__}: {exc}",
                              "trace": traceback.format_exc()[-1500:]}), flush=True)
            self.metrics["renderer_errors"] += 1
            self.notify("error", message=f"filler plan failed: {exc}")

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
        state = {"reply_start": None, "offset": 0.0, "total": 0, "tts_done": False, "chunks": 0}
        plan = {"opener_end": tl.now() + 0.2}

        # 1. The prepared opener lands immediately (when this persona's is ready), then the
        #    filler plan takes over the gap.
        opener = self.pick("opener")
        if opener is None:
            self.notify("acknowledging", seconds=0, pending=True)
        else:
            placed = tl.schedule(tl.now() + 0.1, opener.audio48, opener.frames, generation, tag="filler")
            if placed:
                plan["opener_end"] = placed[1]
                self.last_playout_start = placed[0]
                self.metrics["ack_start_seconds"].append(round(placed[0] - t0, 2))
                self.metrics["fillers"].append({"kind": "opener", "text": opener.text, "at": round(placed[0] - t0, 2)})
                self.notify("acknowledging", seconds=round(placed[1] - placed[0], 2), text=opener.text)
        filler = asyncio.create_task(self.fill_gap(plan, state, generation, t0))

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
                            if "verified_match" in event and (event.get("fallback") or event["takes"] > 1):
                                check = {"text": event.get("text", "")[:80], "match": round(float(event["verified_match"]), 2),
                                         "takes": int(event["takes"]), "fallback": bool(event.get("fallback"))}
                                self.metrics.setdefault("speech_checks", []).append(check)
                                self.notify("speech_check", **check)
                        elif kind == "error":
                            raise RuntimeError(event["message"])
                await queue.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put(exc)

        producer = asyncio.create_task(receive())
        pending24 = PcmQueue()                   # TTS audio not yet rendered
        rendered: list = []                      # (frames, audio48k) waiting for a start time
        need = self.renderer.samples * 24000 // 16000      # 24 kHz samples per renderer chunk

        def take(item):
            if item is None:
                state["tts_done"] = True
            elif isinstance(item, Exception):
                raise item
            else:
                pending24.push(item)
                state["total"] += len(item)

        def decide_start():
            """Pick the reply start once: the earliest moment the reply can play without
            stalling given the renderer's measured rate, but not before the opener and the
            minimum head start, and not after the cap. The filler plan sees it and wraps up."""
            total_seconds = state["total"] / 24000
            words = len(reply_text["text"].split())
            reply_seconds = total_seconds if state["tts_done"] else max(total_seconds, words / 2.5)
            required = head_start_required(reply_seconds, self.renderer.ratio, self.renderer.first_chunk_seconds or 1.5)
            start = max(plan["opener_end"] + 0.3, t0 + min(MAX_HEAD_START, max(HEAD_START, required)))
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
                piece = pending24.pop(need)
                frames, _ = await self.renderer.render(resample(piece, 24000, 16000), state["chunks"] == 0, generation, "reply:front")
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
            filler.cancel()
            producer.cancel()
            await asyncio.gather(filler, producer, return_exceptions=True)

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
            clock = 90000 if sender.track.kind == "video" else 48000     # RTCP jitter is in RTP clock ticks
            for stat in report.values():
                if stat.type == "outbound-rtp":
                    entry.update(packets_sent=stat.packetsSent, bytes_sent=stat.bytesSent)
                elif stat.type == "remote-inbound-rtp":
                    entry.update(packets_lost=stat.packetsLost, fraction_lost=round(float(stat.fractionLost), 4),
                                 jitter_ms=round(float(stat.jitter) / clock * 1000, 1), rtt_ms=round(float(stat.roundTripTime) * 1000, 1))
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
                "idle": {"seconds": round(tl.idle_seconds(), 2), "segments": len(tl.idle_segments),
                         "working_seconds": round(tl.idle_seconds("working"), 2)},
                "clips": {kind: [c.text for c in clips] for kind, clips in self.clips.items()}, "working_pose": self.working_pose,
                "renderer": {"sessions": self.renderer.sessions, "ratio": round(self.renderer.ratio, 3), "motion": self.renderer.motion,
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


async def _persona_portrait(persona_id: str, owner_id: str) -> tuple[bytes, dict]:
    """The persona's portrait as the backend recorded it (same /jobs volume), and the record."""
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
    return portrait.read_bytes(), record


def _voice_digest(record: dict) -> str:
    """What the acknowledgement's voice depends on: the stock voice name, or the content
    of the cloned conditioning file."""
    if record.get("voice_mode") == "stock" or not (record.get("files") or {}).get("conditioning"):
        return f"stock:{record.get('stock_voice') or ''}"
    path = Path(record["files"]["conditioning"])
    if path.is_relative_to(ROOT.parent) and path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"file:{path.name}"


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
    record: dict = {}
    if persona_id:
        payload, record = await _persona_portrait(persona_id, owner.get())
    elif image is not None:
        payload = await image.read(5 * 1024 * 1024 + 1)
        if len(payload) > 5 * 1024 * 1024:
            raise HTTPException(413, "image exceeds 5 MB")
    else:
        raise HTTPException(400, "upload a portrait or choose a persona")
    original = payload
    portrait_digest = hashlib.sha256(original).hexdigest()[:16]
    enhanced = False
    enhance = {"fidelity": float(os.getenv("ASSISTANT_ENHANCE_FIDELITY", "0.7")),
               "blend": float(os.getenv("ASSISTANT_ENHANCE_BLEND", "0.85"))}
    if os.getenv("ASSISTANT_ENHANCE_PORTRAIT", "1") == "1":
        # Restore the still once before it conditions the renderer (see /portrait/enhance).
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{RENDERER}/portrait/enhance", json={"image_b64": base64.b64encode(payload).decode(), **enhance})
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
                        persona_id or None, portrait_digest=portrait_digest,
                        voice_digest=_voice_digest(record) if record else "", enhance={**enhance, "applied": enhanced})
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
        if session.clips["opener"] and (session.background is None or session.background.done()):
            # Everything was cached: tell the client now that the channel exists.
            session.notify("ready", idle_seconds=round(session.timeline.idle_seconds(), 1), clips=session.clip_count(),
                           working_pose=session.working_pose, cached=True)

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
