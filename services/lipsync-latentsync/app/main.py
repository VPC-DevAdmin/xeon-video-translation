"""LatentSync lipsync microservice.

Endpoints:
- GET  /health         — liveness + top-level readiness summary
- GET  /ready          — imports the ML stack; reports which deps are live.
                         In PR-LS-1a (this PR) the ML stack isn't installed
                         yet so every entry reports `ok=false`. That's
                         expected; the response is still 200.
- GET  /weights        — reports which weight files are present on disk.
                         All missing in PR-LS-1a; PR-LS-1b adds the
                         download script.
- POST /lipsync        — run lipsync. Returns 501 in PR-LS-1a and PR-LS-1b.
                         PR-LS-1c flips it on.

Input paths live under /jobs (shared volume). No bytes cross the wire.

Staging:

  PR-LS-1a (this PR)  HTTP scaffold, /lipsync returns structured 501
  PR-LS-1b            ML deps + weight download; /ready and /weights green
  PR-LS-1c            First runnable inference (single-frame diffusion,
                      expect ~10 min/second of source video on CPU)

CPU realism: LatentSync is SD 1.5 latent diffusion per frame. Even with
every CPU optimization we have, a 10-second clip will take ~1.5–2 hours.
This service is designed for **batch workflows**, not live demos. Budget
accordingly.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger(__name__)

VERSION = "0.1.0"
# Flipped to True in PR-LS-1c. Until then, /lipsync returns a clean 501.
INFERENCE_IMPLEMENTED = False

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
WEIGHTS_ROOT = MODEL_CACHE_DIR / "latentsync"


app = FastAPI(
    title="polyglot-demo lipsync (LatentSync)",
    version=VERSION,
    description=(
        "CPU LatentSync lipsync service. "
        "PR-LS-1a: HTTP scaffold. /lipsync returns 501 until PR-LS-1c."
    ),
)


@app.on_event("startup")
def _log_startup() -> None:
    log.info(
        "lipsync-latentsync %s starting — inference_implemented=%s",
        VERSION, INFERENCE_IMPLEMENTED,
    )


# --------------------------------------------------------------------------- #
# Request / response models
#
# The shape matches lipsync-musetalk's request so the backend can hand
# either service the same payload. Per-request quality knobs for LatentSync
# (denoising step count, CFG scale, guidance, ...) land in PR-LS-1c; they
# live as optional fields here so future requests are forward-compatible.
# --------------------------------------------------------------------------- #


class LipsyncRequest(BaseModel):
    video_path: str = Field(
        ..., description="Absolute path to the source video (shared /jobs volume).",
    )
    audio_path: str = Field(
        ..., description="Absolute path to the new audio.",
    )
    output_path: str = Field(
        ..., description="Where to write the lipsynced MP4.",
    )

    # Per-request quality knobs. Forward-compat placeholders today; wired
    # up in PR-LS-1c when inference lands. Omit to use service env defaults.
    num_inference_steps: int | None = Field(
        None, ge=1, le=100,
        description=(
            "Number of latent-diffusion steps per frame. Higher = better "
            "quality, linearly slower. LatentSync default is 20; 10 is a "
            "common 'draft' setting. Omit to use LATENTSYNC_STEPS env."
        ),
    )
    guidance_scale: float | None = Field(
        None, ge=0.0, le=15.0,
        description=(
            "Classifier-free guidance scale. Default 1.5 in LatentSync; "
            "higher pushes mouth shapes harder at the cost of identity "
            "drift. Omit to use LATENTSYNC_GUIDANCE env."
        ),
    )
    seed: int | None = Field(
        None,
        description="Diffusion seed for reproducibility. Omit for random.",
    )


class LipsyncResponse(BaseModel):
    status: Literal["ok", "not_implemented"]
    output_path: str | None = None
    frames_processed: int | None = None
    duration_ms: int | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- #
# Readiness introspection
#
# Python modules the inference code will need. In PR-LS-1a every one of
# these imports fails because the service only ships FastAPI + Pydantic.
# That's fine — /ready still returns 200, just with `status="degraded"`
# and every dep marked `ok=false`. This gives the operator a one-line
# view of exactly how far the image has been built out.
# --------------------------------------------------------------------------- #


_REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "einops",
    "omegaconf",
    "librosa",
    "soundfile",
    "cv2",
    "insightface",
    "onnxruntime",
    "huggingface_hub",
    # IPEX is nice-to-have. Inference still works without it, just slower.
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


# Each entry: (relative path under /models/latentsync, minimum plausible
# size). All missing in PR-LS-1a; PR-LS-1b adds the download script. Paths
# and sizes are provisional based on LatentSync upstream's release layout
# and will be tightened once the download script actually pulls them.
_REQUIRED_WEIGHTS: tuple[tuple[str, int], ...] = (
    # LatentSync UNet checkpoint.
    ("latentsync_unet.pt",          1_000_000_000),
    # SD 1.5 VAE reused for latent encode/decode. Same file lipsync-musetalk
    # uses but we keep a dedicated copy to keep the services independent.
    ("sd-vae/config.json",                     100),
    ("sd-vae/diffusion_pytorch_model.bin", 300_000_000),
    # Whisper tiny for audio conditioning (same pattern as MuseTalk).
    ("whisper/config.json",                     100),
    ("whisper/pytorch_model.bin",        50_000_000),
    ("whisper/preprocessor_config.json",         50),
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
        "service": "lipsync-latentsync",
        "version": VERSION,
        "inference_implemented": INFERENCE_IMPLEMENTED,
        "weights_ready": all_weights_present,
        "phase": "PR-LS-1a (scaffold only)",
        "milestone": (
            "HTTP scaffold reachable; /lipsync returns 501. "
            "Weights + deps land in PR-LS-1b; inference in PR-LS-1c."
        ),
        "num_inference_steps": os.environ.get("LATENTSYNC_STEPS", "20"),
        "guidance_scale": os.environ.get("LATENTSYNC_GUIDANCE", "1.5"),
    }


@app.get("/ready")
def ready() -> dict:
    deps = _dep_status()
    missing = [name for name, info in deps.items() if not info["ok"]]
    # PR-LS-1a: every dep is missing on purpose. `status="degraded"` is
    # the honest answer. Callers (including `make health`) that just want
    # a boolean should look at `missing_or_broken` being empty.
    return {
        "status": "ok" if not missing else "degraded",
        "deps": deps,
        "missing_or_broken": missing,
        "note": (
            "PR-LS-1a ships no ML deps; every module will import-fail. "
            "PR-LS-1b installs torch + diffusers and this endpoint turns green."
        ),
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
        "download_command": (
            "docker compose exec lipsync-latentsync "
            "bash /app/scripts/download_models.sh"
        ),
        "note": (
            "PR-LS-1a ships no download script yet; this endpoint will "
            "report everything missing until PR-LS-1b lands."
        ),
    }


@app.post("/lipsync", response_model=LipsyncResponse)
def lipsync(req: LipsyncRequest) -> LipsyncResponse:
    """Run lipsync. Returns 501 until PR-LS-1c lands.

    We still validate the input paths so integration tests exercise the
    same failure modes as the real inference path will. The 501 response
    carries a structured body so the backend client can translate it into
    a clean user-facing error.
    """
    video = Path(req.video_path)
    audio = Path(req.audio_path)

    # Input validation runs even in PR-LS-1a so the backend client's
    # 400-path plumbing is exercised in integration tests.
    if not video.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"video_path not visible to this service: {video!s}. "
                "Both services must mount the same /jobs volume."
            ),
        )
    if not audio.exists():
        raise HTTPException(
            status_code=400,
            detail=f"audio_path not visible: {audio!s}.",
        )

    if not INFERENCE_IMPLEMENTED:
        raise HTTPException(
            status_code=501,
            detail={
                "phase": "PR-LS-1a",
                "message": (
                    "LatentSync inference is not yet implemented in this "
                    "service. PR-LS-1a only ships the HTTP scaffold — the "
                    "ML deps land in PR-LS-1b and the inference path in "
                    "PR-LS-1c. This 501 is expected behavior."
                ),
                "next_step": (
                    "Either wait for PR-LS-1c, or use LIPSYNC=musetalk "
                    "(slow but works) / LIPSYNC=none (fastest)."
                ),
                "docs": "docs/lipsync.md",
            },
        )

    # Once PR-LS-1c lands, the real inference path takes over here.
    raise HTTPException(status_code=500, detail="unreachable: INFERENCE_IMPLEMENTED True but no dispatch")
