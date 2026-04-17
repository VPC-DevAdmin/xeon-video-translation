"""MuseTalk lipsync microservice.

Endpoints:
- GET  /health         — liveness + top-level readiness summary
- GET  /ready          — imports the ML stack; reports which deps are live
- GET  /weights        — reports which weight files are present on disk
- POST /lipsync        — run lipsync. PR 1b still returns 501 (weights + deps
                          are now present; inference wiring lands in PR 1c).

Input paths live under /jobs (shared volume). No bytes cross the wire.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger(__name__)

VERSION = "0.2.0"
INFERENCE_IMPLEMENTED = False  # flipped to True by PR 1c

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
WEIGHTS_ROOT = MODEL_CACHE_DIR / "musetalk"


app = FastAPI(
    title="polyglot-demo lipsync (MuseTalk)",
    version=VERSION,
    description="CPU MuseTalk lipsync service. PR 1b: deps + weights ready; inference pending.",
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class LipsyncRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video (shared /jobs volume).")
    audio_path: str = Field(..., description="Absolute path to the new audio.")
    output_path: str = Field(..., description="Where to write the lipsynced MP4.")


class LipsyncResponse(BaseModel):
    status: Literal["ok", "not_implemented"]
    output_path: str | None = None
    frames_processed: int | None = None
    duration_ms: int | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- #
# Readiness introspection
# --------------------------------------------------------------------------- #


# Python modules the inference code will need. Import failures here point at
# a broken image rather than a broken request.
_REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "diffusers",
    "transformers",
    "accelerate",
    "einops",
    "librosa",
    "soundfile",
    "cv2",
    "face_alignment",
    "huggingface_hub",
)


def _dep_status() -> dict[str, dict]:
    status: dict[str, dict] = {}
    for name in _REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            status[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            continue
        version = getattr(mod, "__version__", None)
        status[name] = {"ok": True, "version": version}
    return status


# Each entry: (relative path under /models/musetalk, minimum plausible size).
# Sizes are defensive — catch accidentally-truncated downloads without being
# brittle across mirror revisions.
_REQUIRED_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("musetalkV15/unet.pth",                         800_000_000),
    ("musetalkV15/musetalk.json",                    100),
    ("sd-vae/config.json",                           100),
    ("sd-vae/diffusion_pytorch_model.bin",           300_000_000),
    ("whisper/config.json",                          100),
    ("whisper/pytorch_model.bin",                    50_000_000),
    ("whisper/preprocessor_config.json",             50),
    ("face-parse-bisent/79999_iter.pth",             40_000_000),
    ("face-parse-bisent/resnet18-5c106cde.pth",      40_000_000),
)


def _weight_status() -> dict[str, dict]:
    status: dict[str, dict] = {}
    for rel, min_bytes in _REQUIRED_WEIGHTS:
        path = WEIGHTS_ROOT / rel
        if not path.exists():
            status[rel] = {"ok": False, "reason": "missing"}
            continue
        size = path.stat().st_size
        if size < min_bytes:
            status[rel] = {
                "ok": False,
                "reason": f"truncated? {size} < expected {min_bytes}",
                "size_bytes": size,
            }
            continue
        status[rel] = {"ok": True, "size_bytes": size}
    return status


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict:
    weights = _weight_status()
    all_weights_present = all(w["ok"] for w in weights.values())
    return {
        "status": "ok",
        "service": "lipsync-musetalk",
        "version": VERSION,
        "inference_implemented": INFERENCE_IMPLEMENTED,
        "weights_ready": all_weights_present,
        "milestone": "PR 1b — deps + weights; inference pending in PR 1c",
    }


@app.get("/ready")
def ready() -> dict:
    deps = _dep_status()
    missing = [name for name, info in deps.items() if not info["ok"]]
    return {
        "status": "ok" if not missing else "degraded",
        "deps": deps,
        "missing_or_broken": missing,
    }


@app.get("/weights")
def weights() -> dict:
    w = _weight_status()
    missing = [rel for rel, info in w.items() if not info["ok"]]
    return {
        "status": "ok" if not missing else "incomplete",
        "cache_dir": str(WEIGHTS_ROOT),
        "weights": w,
        "missing_or_truncated": missing,
        "download_command": "docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh",
    }


@app.post("/lipsync", response_model=LipsyncResponse)
def lipsync(req: LipsyncRequest) -> LipsyncResponse:
    video = Path(req.video_path)
    audio = Path(req.audio_path)

    if not video.exists():
        raise HTTPException(
            status_code=400,
            detail=f"video_path not visible to this service: {video!s}. "
                   "Both services must mount the same /jobs volume.",
        )
    if not audio.exists():
        raise HTTPException(status_code=400, detail=f"audio_path not visible: {audio!s}.")

    if not INFERENCE_IMPLEMENTED:
        log.info("lipsync requested for %s + %s — returning 501 (PR 1b; weights ready, inference pending)",
                 video.name, audio.name)
        raise HTTPException(
            status_code=501,
            detail={
                "phase": "not-implemented",
                "message": (
                    "MuseTalk inference is not yet wired up. Deps and weights "
                    "are in place as of PR 1b; the UNet/VAE forward pass lands "
                    "in PR 1c."
                ),
                "see_also": "docs/lipsync.md",
            },
        )

    raise HTTPException(status_code=500, detail="unreachable in PR 1b")
