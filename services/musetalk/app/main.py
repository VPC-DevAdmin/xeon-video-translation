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

VERSION = "0.7.0"
INFERENCE_IMPLEMENTED = True

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
WEIGHTS_ROOT = MODEL_CACHE_DIR / "musetalk"


app = FastAPI(
    title="polyglot-demo lipsync (MuseTalk)",
    version=VERSION,
    description="CPU MuseTalk lipsync service. PR 1c: first runnable inference.",
)


@app.on_event("startup")
def _log_startup() -> None:
    log.info("lipsync-musetalk %s starting — inference_implemented=%s", VERSION, INFERENCE_IMPLEMENTED)


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
    "insightface",
    "onnxruntime",
    "huggingface_hub",
    # IPEX is nice-to-have: if it's missing, inference just runs vanilla
    # PyTorch. `/ready` still treats an import failure as non-fatal.
    "intel_extension_for_pytorch",
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
    # codeformer.pth lives at MODEL_CACHE_DIR/codeformer/ — see runner.py.
    # It's a *sibling* of the musetalk/ tree but tracked here because the
    # service needs it for the face-restoration stage. Missing = warning,
    # not error: restoration is skipped and MuseTalk output is shipped raw.
    ("../codeformer/codeformer.pth",                 300_000_000),
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
        "ipex_dtype": os.environ.get("MUSETALK_IPEX_DTYPE", "fp32"),
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "blend_mode": os.environ.get("MUSETALK_BLEND_MODE", "jaw"),
        "blend_feather": os.environ.get("MUSETALK_BLEND_FEATHER", "0.04"),
        "face_restore": os.environ.get("MUSETALK_FACE_RESTORE", "codeformer"),
        "face_restore_fidelity": os.environ.get("MUSETALK_FACE_RESTORE_FIDELITY", "0.7"),
        "face_restore_blend": os.environ.get("MUSETALK_FACE_RESTORE_BLEND", "0.6"),
        "milestone": "MuseTalk + SCRFD + IPEX + CodeFormer face restore",
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
    import time

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

    # Import here so the service stays responsive to /health even if torch
    # imports fail on boot.
    from .musetalk.inference import WeightPaths, run

    weights = WeightPaths.from_cache(Path(os.environ.get("MODEL_CACHE_DIR", "/models")))
    missing = weights.missing()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "phase": "weights-missing",
                "missing": [str(p) for p in missing],
                "fix": "docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh",
            },
        )

    started = time.perf_counter()
    try:
        result = run(
            video_path=video,
            audio_path=audio,
            output_path=Path(req.output_path),
            weight_paths=weights,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"phase": "weights-missing", "error": str(e)})
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail={"phase": "input-rejected", "error": str(e)})
    except Exception as e:
        log.exception("MuseTalk inference failed")
        raise HTTPException(
            status_code=500,
            detail={"phase": "inference-error", "error": f"{type(e).__name__}: {e}"},
        )
    duration_ms = int((time.perf_counter() - started) * 1000)
    log.info("lipsync done in %d ms — %d frames", duration_ms, result.frames_processed)
    return LipsyncResponse(
        status="ok",
        output_path=result.output_path,
        frames_processed=result.frames_processed,
        duration_ms=duration_ms,
    )
