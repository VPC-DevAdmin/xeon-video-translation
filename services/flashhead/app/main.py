"""FlashHead render service: a talking head from a portrait, one audio chunk at a time.

Session contract for the video assistant:
  POST /sessions                 {image_path, seed}      -> session id; prepares the portrait
  POST /sessions/{id}/render     {pcm_b64, reset}        -> raw RGB frames for that audio chunk
  POST /sessions/{id}/reset                              -> restart motion from the portrait
  DELETE /sessions/{id}
  POST /portrait/enhance         {image_b64, ...}        -> CodeFormer-restored portrait (PNG)
  POST /portrait/pose            {image_b64, pitch, yaw} -> head pose / gaze edit (PNG, LivePortrait)
  GET  /health                                           -> readiness, chunk spec, timings

One pipeline per process (one GPU). Motion state (the last frames' latents and the
rolling audio window) is kept per session so consecutive chunks of a reply are
continuous; `reset` starts a new reply from the portrait pose. Frames come back as
`application/octet-stream` uint8 RGB with the shape in headers: no files cross
between services.

Environment: FLASHHEAD_MODEL (pro|lite), FLASHHEAD_COMPILE (1), FLASHHEAD_MODEL_ROOT,
FLASHHEAD_SOURCE (directory containing the flash_head package), FLASHHEAD_WARM_CHUNKS,
JOB_ARTIFACTS_DIR (portraits must live under it).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .chunking import AudioContext, ChunkSpec, decode_pcm, fit_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("flashhead")

SOURCE = os.environ.get("FLASHHEAD_SOURCE", "/experiment")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
MODEL_ROOT = Path(os.environ.get("FLASHHEAD_MODEL_ROOT", "/experiment-models"))
MODEL = os.environ.get("FLASHHEAD_MODEL", "pro")
COMPILE = os.environ.get("FLASHHEAD_COMPILE", "1") == "1"
WARM_CHUNKS = int(os.environ.get("FLASHHEAD_WARM_CHUNKS", "2"))
JOBS = Path(os.environ.get("JOB_ARTIFACTS_DIR", "/jobs")).resolve()
MAX_SESSIONS = int(os.environ.get("FLASHHEAD_MAX_SESSIONS", "4"))

app = FastAPI(title="polyglot-flashhead", version="0.1.0")
_LOCK = threading.Lock()
_STATE: dict = {"ready": False, "warm": False, "error": None, "load_seconds": None, "warm_seconds": None,
                "chunks_rendered": 0, "chunk_seconds_ema": None}
_pipeline = None
_spec: ChunkSpec | None = None
_sessions: dict[str, dict] = {}
_active_session: str | None = None   # whose portrait and motion state the pipeline currently holds


class Session(BaseModel):
    image_path: str
    seed: int = 42


class EnhanceRequest(BaseModel):
    image_b64: str = Field(..., description="PNG or JPEG portrait")
    fidelity: float = Field(0.7, ge=0.0, le=1.0, description="CodeFormer w: 1 keeps identity, 0 maximises detail")
    blend: float = Field(0.85, ge=0.0, le=1.0, description="how much of the restored face replaces the original")


class PoseRequest(BaseModel):
    image_b64: str = Field(..., description="PNG or JPEG portrait")
    pitch: float = Field(12.0, ge=-30, le=30, description="degrees; positive looks down")
    yaw: float = Field(-10.0, ge=-40, le=40)
    roll: float = Field(0.0, ge=-30, le=30)
    eyes_x: float = Field(-4.0, ge=-20, le=20)
    eyes_y: float = Field(9.0, ge=-20, le=20, description="positive lowers the gaze")


class RenderRequest(BaseModel):
    pcm_b64: str = Field(..., description="PCM16 mono at the chunk sample rate; up to samples_per_chunk")
    reset: bool = Field(False, description="start this reply from the portrait pose")
    generation: int = 0


def _load() -> None:
    global _pipeline, _spec
    import torch
    import flash_head.src.pipeline.flash_head_pipeline as implementation
    from flash_head.inference import get_pipeline, get_infer_params

    implementation.COMPILE_MODEL = COMPILE
    implementation.COMPILE_VAE = COMPILE
    started = time.perf_counter()
    _pipeline = get_pipeline(1, str(MODEL_ROOT / "SoulX-FlashHead-1_3B"), MODEL, str(MODEL_ROOT / "wav2vec2-base-960h"))
    params = get_infer_params()
    _spec = ChunkSpec(params["frame_num"], params["motion_frames_num"], params["tgt_fps"],
                      params["sample_rate"], params["cached_audio_duration"])
    torch.cuda.synchronize()
    _STATE["load_seconds"] = round(time.perf_counter() - started, 2)
    _STATE["ready"] = True
    log.info("pipeline loaded in %.1fs: model=%s compile=%s chunk=%s", _STATE["load_seconds"], MODEL, COMPILE, _spec.as_dict())


def _prepare(image_path: str, seed: int) -> None:
    from flash_head.inference import get_base_data
    get_base_data(_pipeline, image_path, seed, False)


def _generate(context: AudioContext, audio: np.ndarray) -> np.ndarray:
    import torch
    from flash_head.inference import get_audio_embedding, run_pipeline

    window = context.push(audio)
    start, end = context.window
    embedding = get_audio_embedding(_pipeline, window, start, end)
    frames = run_pipeline(_pipeline, embedding)[_spec.frame_num - _spec.frames:].to(torch.uint8)
    torch.cuda.synchronize()
    return frames.cpu().numpy()


def _warm() -> None:
    warm_image = os.environ.get("FLASHHEAD_WARMUP_IMAGE", "/models/avatar-warmup/portrait.png")
    if not Path(warm_image).exists():
        log.warning("no warm-up portrait at %s; first session pays the compile", warm_image)
        return
    started = time.perf_counter()
    with _LOCK:
        _prepare(warm_image, 42)
        context = AudioContext(_spec)
        for _ in range(WARM_CHUNKS):
            _generate(context, np.zeros(_spec.samples, np.float32))
    _STATE["warm_seconds"] = round(time.perf_counter() - started, 2)
    _STATE["warm"] = True
    log.info("warm-up done in %.1fs (%d chunks)", _STATE["warm_seconds"], WARM_CHUNKS)


def _startup_thread() -> None:
    try:
        _load()
        _warm()
    except Exception as exc:  # keep /health answering with the reason
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("startup failed")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_startup_thread, name="flashhead-startup", daemon=True).start()


@app.get("/health")
def health() -> dict:
    from . import pose as pose_module
    return {"status": "ok" if _STATE["ready"] and not _STATE["error"] else "starting" if not _STATE["error"] else "error",
            "model": MODEL, "compile": COMPILE, **_STATE, "pose_edit": pose_module.available(),
            "chunk": _spec.as_dict() if _spec else None, "sessions": len(_sessions), "active_session": _active_session}


def _require_ready() -> None:
    if _STATE["error"]:
        raise HTTPException(500, {"phase": "startup", "error": _STATE["error"]})
    if not _STATE["ready"]:
        raise HTTPException(503, "renderer still loading", headers={"Retry-After": "5"})


def _activate(session_id: str) -> dict:
    """Make the pipeline hold this session's portrait and motion state."""
    global _active_session
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    if _active_session != session_id:
        started = time.perf_counter()
        _prepare(session["image_path"], session["seed"])
        session["context"].reset()
        _active_session = session_id
        session["prepare_seconds"] = round(time.perf_counter() - started, 2)
    return session


@app.post("/sessions")
def create(body: Session) -> dict:
    _require_ready()
    image = Path(body.image_path).resolve()
    if not image.is_relative_to(JOBS) or not image.is_file():
        raise HTTPException(400, "image_path must be a file under JOB_ARTIFACTS_DIR")
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, "renderer session capacity reached")
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {"image_path": str(image), "seed": body.seed, "context": AudioContext(_spec),
                             "created": time.time(), "chunks": 0, "prepare_seconds": None}
    with _LOCK:
        _activate(session_id)
    return {"session_id": session_id, **_spec.as_dict(), "prepare_seconds": _sessions[session_id]["prepare_seconds"]}


@app.post("/sessions/{session_id}/reset")
def reset(session_id: str) -> dict:
    _require_ready()
    with _LOCK:
        session = _activate(session_id)
        _pipeline.reset_person_name()
        session["context"].reset()
    return {"status": "reset"}


@app.post("/sessions/{session_id}/render")
def render(session_id: str, body: RenderRequest) -> Response:
    _require_ready()
    samples = decode_pcm(body.pcm_b64)
    if len(samples) > _spec.samples:
        raise HTTPException(400, f"at most {_spec.samples} samples per render call")
    audio, covered = fit_chunk(samples, _spec)
    started = time.perf_counter()
    with _LOCK:
        session = _activate(session_id)
        if body.reset:
            _pipeline.reset_person_name()
            session["context"].reset()
        frames = _generate(session["context"], audio)
        session["chunks"] += 1
    seconds = time.perf_counter() - started
    _STATE["chunks_rendered"] += 1
    ema = _STATE["chunk_seconds_ema"]
    _STATE["chunk_seconds_ema"] = round(seconds if ema is None else 0.8 * ema + 0.2 * seconds, 3)
    frames = np.ascontiguousarray(frames[:covered] if covered else frames[:0])
    headers = {"X-Frames": str(frames.shape[0]), "X-Height": str(frames.shape[1]) if frames.size else str(_pipeline.target_h),
               "X-Width": str(frames.shape[2]) if frames.size else str(_pipeline.target_w),
               "X-Seconds": f"{seconds:.3f}", "X-Generation": str(body.generation), "X-Fps": str(_spec.fps)}
    return Response(content=frames.tobytes(), media_type="application/octet-stream", headers=headers)


def _five_points(image_bgr: np.ndarray):
    """Eyes, nose tip and mouth corners from MediaPipe FaceMesh, in CodeFormer's order
    (left-in-image eye, right eye, nose, left mouth corner, right mouth corner)."""
    import cv2
    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        result = mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    pts = np.array([(v.x * w, v.y * h) for v in result.multi_face_landmarks[0].landmark], dtype=np.float32)
    eyes = sorted([pts[[33, 133]].mean(axis=0), pts[[362, 263]].mean(axis=0)], key=lambda p: p[0])
    mouth = sorted([pts[61], pts[291]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], pts[1], mouth[0], mouth[1]], dtype=np.float32)


@app.post("/portrait/enhance")
def enhance(body: EnhanceRequest) -> Response:
    """One-time face restoration of a portrait before it conditions the renderer.
    FlashHead reproduces the texture of its reference, so a soft webcam frame yields a
    soft face; restoring the still once is temporally safe (no per-frame flicker)."""
    import base64
    import cv2
    import torch
    from .codeformer import restore_frame

    raw = np.frombuffer(base64.b64decode(body.image_b64), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "could not decode the portrait")
    kps = _five_points(image)
    if kps is None:
        raise HTTPException(422, "no face found in the portrait")
    started = time.perf_counter()
    with _LOCK:
        restored = restore_frame(image, kps, torch.device("cuda"), fidelity=body.fidelity, blend=body.blend)
    ok, png = cv2.imencode(".png", restored)
    if not ok:
        raise HTTPException(500, "could not encode the restored portrait")
    return Response(content=png.tobytes(), media_type="image/png",
                    headers={"X-Seconds": f"{time.perf_counter() - started:.2f}", "X-Face-Points": "mediapipe"})


@app.post("/portrait/pose")
def pose(body: PoseRequest) -> Response:
    """Head pose and gaze edit of a portrait (LivePortrait), used once per persona for the
    assistant's "looking something up" footage. 503 when LivePortrait is not installed."""
    import base64
    import cv2
    from . import pose as pose_module

    if not pose_module.available():
        raise HTTPException(503, "pose editing is not available on this renderer (LivePortrait not installed)")
    raw = np.frombuffer(base64.b64decode(body.image_b64), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "could not decode the portrait")
    started = time.perf_counter()
    try:
        with _LOCK:
            posed = pose_module.pose_portrait(image, body.pitch, body.yaw, body.roll, body.eyes_x, body.eyes_y)
    except Exception as exc:
        raise HTTPException(422, f"pose edit failed: {type(exc).__name__}: {exc}")
    ok, png = cv2.imencode(".png", posed)
    if not ok:
        raise HTTPException(500, "could not encode the posed portrait")
    return Response(content=png.tobytes(), media_type="image/png", headers={"X-Seconds": f"{time.perf_counter() - started:.2f}"})


@app.delete("/sessions/{session_id}")
def delete(session_id: str) -> dict:
    global _active_session
    _sessions.pop(session_id, None)
    if _active_session == session_id:
        _active_session = None
    return {"status": "closed"}
