"""LatentSync lipsync microservice.

Endpoints:
- GET  /health         — liveness + top-level readiness summary
- GET  /ready          — imports the ML stack; reports which deps are live.
- GET  /weights        — reports which weight files are present on disk.
- POST /lipsync        — run lipsync. Returns 501 through PR-LS-1b; flips
                         on in PR-LS-1c.

Input paths live under /jobs (shared volume). No bytes cross the wire.

Staging:

  PR-LS-1a (shipped)  HTTP scaffold, /lipsync returns structured 501
  PR-LS-1b (current)  ML deps installed + weight download live. Once the
                      user runs `make models-latentsync`, /ready and
                      /weights both turn green. /lipsync still 501.
  PR-LS-1c            First runnable inference. Expect ~10 min/sec of
                      source video on CPU — batch workflow territory.

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

VERSION = "0.2.0"
# Flipped to True in PR-LS-1c. Until then, /lipsync returns a clean 501.
INFERENCE_IMPLEMENTED = False
# Phase string surfaced in /health and the 501 body so operators can tell
# at a glance which staged PR built the image they're poking at.
PHASE = "PR-LS-1b (ML stack + weights download)"

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
WEIGHTS_ROOT = MODEL_CACHE_DIR / "latentsync"


app = FastAPI(
    title="polyglot-demo lipsync (LatentSync)",
    version=VERSION,
    description=(
        "CPU LatentSync lipsync service. "
        f"{PHASE}. /lipsync still returns 501 — inference lands in PR-LS-1c."
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
# Python modules the inference code will need. PR-LS-1a shipped no ML
# stack so every import failed; PR-LS-1b (this PR) should have them all
# importable. If /ready still shows failures after a clean rebuild,
# that's a Dockerfile or pyproject problem, not a code problem.
# --------------------------------------------------------------------------- #


_REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "einops",
    "omegaconf",
    "librosa",
    "soundfile",
    "decord",
    "cv2",
    "mediapipe",
    "insightface",
    "onnxruntime",
    "huggingface_hub",
    # LatentSync uses the original openai/whisper package (loads
    # whisper/tiny.pt directly), not transformers.WhisperModel.
    "whisper",
    "imageio",
    "scenedetect",
    "kornia",
    "face_alignment",
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
# size). The manifest was realigned in PR-LS-1b against the actual layout
# published at https://huggingface.co/ByteDance/LatentSync-1.6 — upstream
# README's "if the download is successful, the checkpoints should appear
# as follows" section.
#
# Sizes are defensive lower bounds so we catch truncated downloads
# without becoming brittle across minor release size drift.
_REQUIRED_WEIGHTS: tuple[tuple[str, int], ...] = (
    # Main diffusion UNet. SD 1.5 at heart, fine-tuned for lipsync.
    # Current release is ~5 GB at float32.
    ("latentsync_unet.pt",       4_000_000_000),
    # SyncNet auxiliary model, used for audio-visual supervision. Loaded
    # separately from the UNet. Current release is ~1.6 GB.
    ("stable_syncnet.pt",        1_000_000_000),
    # OpenAI Whisper tiny, bundled inside the LatentSync repo. Loaded
    # through the openai-whisper package, not transformers.
    ("whisper/tiny.pt",             50_000_000),
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
        "phase": PHASE,
        "milestone": (
            "ML stack installed; download_models.sh pulls UNet + SyncNet + "
            "Whisper tiny from ByteDance/LatentSync-1.6. /lipsync still "
            "returns 501 — inference wiring lands in PR-LS-1c."
        ),
        "latentsync_repo": os.environ.get(
            "LATENTSYNC_HF_REPO", "ByteDance/LatentSync-1.6",
        ),
        "num_inference_steps": os.environ.get("LATENTSYNC_STEPS", "20"),
        "guidance_scale": os.environ.get("LATENTSYNC_GUIDANCE", "1.5"),
    }


@app.get("/ready")
def ready() -> dict:
    deps = _dep_status()
    missing = [name for name, info in deps.items() if not info["ok"]]
    return {
        "status": "ok" if not missing else "degraded",
        "deps": deps,
        "missing_or_broken": missing,
        "note": (
            "All listed modules should import cleanly after PR-LS-1b's "
            "image build. Any `ok=false` here means a Dockerfile or "
            "pyproject.toml regression — check `docker compose logs "
            "lipsync-latentsync` and `make rebuild`."
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
            "Run `make models-latentsync` (or the download_command above) "
            "to pull the ~6.6 GB of weights into the shared /models volume. "
            "First pull takes a few minutes on a decent connection; "
            "subsequent container restarts reuse the volume."
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
        # PR-LS-1b still returns 501 by design; the container now has
        # torch, diffusers, and the weights on disk, but nothing is
        # wired to actually call the UNet. Inference lands in PR-LS-1c.
        # We include a weights-present hint so the user can confirm their
        # `make models-latentsync` ran even while inference is blocked.
        weights_present = all(w["ok"] for w in _weight_status().values())
        raise HTTPException(
            status_code=501,
            detail={
                "phase": PHASE,
                "message": (
                    "LatentSync inference is not yet implemented in this "
                    "service. PR-LS-1b ships the ML stack + weight download "
                    "but not the inference path — that lands in PR-LS-1c. "
                    "This 501 is expected behavior."
                ),
                "weights_present": weights_present,
                "next_step": (
                    "Either wait for PR-LS-1c, or use LIPSYNC=musetalk "
                    "(slow but works) / LIPSYNC=none (fastest)."
                ),
                "docs": "docs/lipsync.md",
            },
        )

    # Once PR-LS-1c lands, the real inference path takes over here.
    raise HTTPException(status_code=500, detail="unreachable: INFERENCE_IMPLEMENTED True but no dispatch")
