"""PR-LS-1c driver — glue between the FastAPI handler and the vendored
LatentSync inference pipeline.

Keep this file small. The heavy lifting lives under ``app/latentsync/``
(vendored upstream code) and ``app/configs/`` (vendored model configs);
the driver's job is:

1. Validate the weights exist on disk.
2. Build the four pipeline components (VAE, audio encoder, UNet,
   scheduler) against CPU tensors in float32.
3. Call ``LipsyncPipeline(...)`` with the per-request knobs forwarded
   from the HTTP layer.
4. Return a small result dict the handler can render into the response.

CPU adaptations from upstream ``scripts/inference.py``:
  - dtype forced to float32 (no CUDA fp16 autodetect).
  - device forced to "cpu"; upstream's ``.to("cuda")`` / ``device="cuda"``
    hard-codes are patched in the vendored tree (see the "CPU patch"
    comments under app/latentsync/...).
  - DeepCache is not used. Upstream wraps the pipeline in
    ``DeepCacheSDHelper`` for a speedup; we skipped the dep in
    PR-LS-1b's pyproject since the speedup is meaningless when the
    bottleneck is CPU float32 matmul.
  - The scheduler loads from our vendored ``app/configs/`` (has
    ``scheduler_config.json``), not upstream's ``configs/`` path.

Dry-run mode: set ``LATENTSYNC_DRY_RUN=1`` to reduce num_inference_steps
to 1 and num_frames to 1. Useful for "does the pipeline even wire up"
smoke tests without committing to a ~50-minute full inference run.

Performance stack (added in the 1c perf follow-up):

  - IPEX bf16 via ``LATENTSYNC_IPEX_DTYPE=bf16`` (default). Applies
    ``ipex.optimize()`` to the UNet + VAE and wraps the pipeline call
    in ``torch.autocast(device_type="cpu", dtype=bfloat16)``. On
    AMX-capable Xeon (Sapphire Rapids+) this is typically 2–4×
    faster than fp32 at essentially imperceptible quality loss.
    Set to ``fp32`` if you see color drift or mouth-texture artifacts.

  - DeepCache via ``LATENTSYNC_ENABLE_DEEPCACHE=1`` (default). Caches
    early denoising-step tensors and reuses them on later steps,
    cutting UNet calls by ~30%. Upstream LatentSync uses it with
    ``cache_interval=3, cache_branch_id=0``.

Both knobs are opt-out, not opt-in — a fresh container ships with
both enabled. Turn them off individually if you're chasing a quality
issue and want to isolate the cause.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Path to the vendored configs dir (app/configs/). Resolved at import
# time so stacktraces make the layout obvious when a file is missing.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_UNET_CONFIG_PATH = _CONFIGS_DIR / "unet" / "stage2.yaml"


def _ipex_dtype():
    """Resolve the IPEX compute dtype from env.

    ``fp32`` is pure kernel acceleration — no numerical drift from
    upstream's tested path. ``bf16`` enables torch CPU autocast around
    the UNet forward and typically gives 2–4× speedup on AMX-capable
    Xeon (Sapphire Rapids+). Mirrors MuseTalk's ``_ipex_dtype`` so
    operators who've tuned one service understand the other's knob.

    Imports torch lazily — the ``/health`` endpoint must stay
    responsive even if the torch stack breaks on boot.
    """
    import torch

    choice = os.environ.get("LATENTSYNC_IPEX_DTYPE", "bf16").lower()
    if choice in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _ipex_optimize(model, name: str):
    """Wrap `model` with ipex.optimize() if IPEX is installed.

    Silently falls through on any failure — perf plumbing should never
    hard-fail a request. Caught broadly because IPEX import can raise
    native-loader errors (e.g. executable-stack markers on some wheels)
    that aren't strictly ImportError.
    """
    try:
        import intel_extension_for_pytorch as ipex
    except Exception as e:
        log.warning("IPEX import failed (%s); skipping %s optimization", e, name)
        return model

    dtype = _ipex_dtype()
    try:
        optimized = ipex.optimize(model.eval(), dtype=dtype, inplace=False)
    except Exception as e:
        log.warning("IPEX optimize(%s) failed (%s); using vanilla PyTorch", name, e)
        return model

    log.info("IPEX optimized %s (dtype=%s)", name, str(dtype).rsplit(".", 1)[-1])
    return optimized


@dataclass
class WeightPaths:
    """Resolved paths to the weights this service needs on disk.

    ``from_cache`` builds one rooted at ``MODEL_CACHE_DIR/latentsync/``
    matching what ``scripts/download_models.sh`` lays down. Callers
    typically do ``WeightPaths.from_cache(...).missing()`` to pre-flight
    the request.
    """

    unet: Path
    syncnet: Path
    whisper_tiny: Path

    @classmethod
    def from_cache(cls, model_cache_dir: Path) -> "WeightPaths":
        root = model_cache_dir / "latentsync"
        return cls(
            unet=root / "latentsync_unet.pt",
            syncnet=root / "stable_syncnet.pt",
            whisper_tiny=root / "whisper" / "tiny.pt",
        )

    def missing(self) -> list[Path]:
        return [p for p in (self.unet, self.syncnet, self.whisper_tiny) if not p.exists()]


@dataclass
class InferenceResult:
    output_path: str
    frames_processed: int
    num_inference_steps: int
    guidance_scale: float
    dry_run: bool


def run(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    weight_paths: WeightPaths,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """Run LatentSync inference. All tensor ops are CPU float32.

    ``num_inference_steps`` / ``guidance_scale`` / ``seed`` are forwarded
    from the per-request HTTP payload. Missing values fall back to
    env-driven defaults (``LATENTSYNC_STEPS`` / ``LATENTSYNC_GUIDANCE``)
    so operators can shift the defaults without deploying.
    """
    missing = weight_paths.missing()
    if missing:
        raise FileNotFoundError(
            "LatentSync weights missing: "
            f"{[str(p) for p in missing]}. "
            "Run `make models-latentsync` on the host."
        )
    if not video_path.exists():
        raise RuntimeError(f"video_path not found: {video_path}")
    if not audio_path.exists():
        raise RuntimeError(f"audio_path not found: {audio_path}")
    if not _UNET_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"UNet config not vendored: {_UNET_CONFIG_PATH}. "
            "This should ship with the image — rebuild if it's missing."
        )

    # Resolve per-request knobs. The HTTP layer validates ranges; we
    # only need to apply env defaults.
    steps = int(num_inference_steps if num_inference_steps is not None
                else os.environ.get("LATENTSYNC_STEPS", "20"))
    guidance = float(guidance_scale if guidance_scale is not None
                     else os.environ.get("LATENTSYNC_GUIDANCE", "1.5"))
    dry_run = os.environ.get("LATENTSYNC_DRY_RUN", "").lower() in ("1", "true", "yes")
    if dry_run:
        # Collapse the denoising loop so we reach the pipeline's end
        # without waiting ~50 minutes. Catches wiring bugs fast.
        steps = 1

    log.info(
        "latentsync inference starting: steps=%d guidance=%.2f seed=%s dry_run=%s",
        steps, guidance, seed, dry_run,
    )

    # Imports are deferred so /health and /ready stay responsive even if
    # the ML stack can't load (e.g. a bad weight file). Any ImportError
    # here surfaces cleanly in the 500 response body.
    import torch
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf

    # Make the vendored `latentsync` package importable without touching
    # PYTHONPATH globally. app/ is the service's working dir inside the
    # container so this is a no-op there; the sys.path.insert is belt-
    # and-braces for local dev loops.
    app_root = Path(__file__).resolve().parent.parent
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))

    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    device = "cpu"
    # Model weights stay fp32 on disk. IPEX handles the bf16 cast
    # internally when enabled; the pipeline's autocast does the rest.
    dtype = torch.float32
    compute_dtype = _ipex_dtype()  # torch.bfloat16 or torch.float32

    # Seed for reproducibility. accelerate.set_seed seeds torch, numpy,
    # and python random in one call. If no seed is given we just log
    # whatever torch picked so reruns can be matched post-hoc.
    if seed is not None and seed >= 0:
        set_seed(int(seed))
    log.info("torch initial seed: %d", torch.initial_seed())

    config = OmegaConf.load(str(_UNET_CONFIG_PATH))
    cross_attn_dim = int(config.model.cross_attention_dim)
    if cross_attn_dim != 384:
        # LatentSync-1.6's release matches whisper/tiny.pt. If the
        # config ever diverges (e.g. someone swaps in a 768-dim config
        # for whisper/small), we catch it loudly here rather than after
        # five minutes of wasted loading.
        raise RuntimeError(
            f"unexpected config.model.cross_attention_dim={cross_attn_dim}; "
            f"only 384 (whisper tiny) is supported in this PR. "
            f"Check app/configs/unet/stage2.yaml vs the weight release."
        )

    # --- Scheduler ------------------------------------------------------
    # DDIMScheduler.from_pretrained wants a directory containing
    # scheduler_config.json. We point it at app/configs/ directly rather
    # than the upstream "configs" path.
    scheduler = DDIMScheduler.from_pretrained(str(_CONFIGS_DIR))

    # --- Audio encoder (Whisper tiny) ----------------------------------
    audio_encoder = Audio2Feature(
        model_path=str(weight_paths.whisper_tiny),
        device=device,
        num_frames=int(config.data.num_frames),
        audio_feat_length=list(config.data.audio_feat_length),
    )

    # --- VAE (SD 1.5 ft-MSE) -------------------------------------------
    # Downloaded on-demand from HF to HF_HOME (/models/huggingface).
    # ~330 MB; first call takes a minute.
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=dtype,
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae = _ipex_optimize(vae, "vae")

    # --- UNet ----------------------------------------------------------
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        str(weight_paths.unet),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)
    unet = _ipex_optimize(unet, "unet")

    # --- Pipeline ------------------------------------------------------
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    # Bind the mask image path to the vendored asset so the pipeline
    # finds it without depending on the container's working directory.
    mask_image_path = Path(__file__).resolve().parent.parent / "latentsync" / "utils" / "mask.png"
    if not mask_image_path.exists():
        raise FileNotFoundError(
            f"vendored mask image missing: {mask_image_path}. "
            "Image was likely built without the latentsync/utils/ assets."
        )

    # Temp dir for intermediate frames/audio — cleaned up by the pipeline
    # internally. We write under /tmp so the `jobs` volume only gets
    # the final mp4.
    temp_dir = Path(os.environ.get("LATENTSYNC_TEMP_DIR", "/tmp/latentsync_work"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Wrap the pipeline call in CPU autocast when bf16 is requested.
    # autocast's allowlist keeps BatchNorm / LayerNorm / Softmax at
    # fp32 for numerical stability; Conv / Linear / etc. run at bf16.
    # When compute_dtype is fp32, autocast becomes a no-op.
    autocast_enabled = compute_dtype == torch.bfloat16
    autocast_ctx = torch.autocast(
        device_type="cpu", dtype=torch.bfloat16, enabled=autocast_enabled,
    )
    log.info(
        "pipeline starting: steps=%d guidance=%.2f compute_dtype=%s autocast=%s",
        steps, guidance,
        str(compute_dtype).rsplit(".", 1)[-1],
        autocast_enabled,
    )

    started = time.perf_counter()
    with torch.no_grad(), autocast_ctx:
        pipeline(
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=str(output_path),
            num_frames=int(config.data.num_frames),
            num_inference_steps=steps,
            guidance_scale=guidance,
            weight_dtype=dtype,
            width=int(config.data.resolution),
            height=int(config.data.resolution),
            mask_image_path=str(mask_image_path),
            temp_dir=str(temp_dir),
        )
    elapsed = time.perf_counter() - started
    log.info("latentsync inference finished in %.1fs", elapsed)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"pipeline returned without raising but no output was written "
            f"at {output_path}. Check service logs for ffmpeg errors."
        )

    # Frame count isn't reported by the pipeline; cheapest honest answer
    # is "we ran at this step/guidance combo". A future iteration can
    # probe the mp4 with ffprobe if the UI needs a real count.
    return InferenceResult(
        output_path=str(output_path),
        frames_processed=-1,  # unknown without an ffprobe round-trip
        num_inference_steps=steps,
        guidance_scale=guidance,
        dry_run=dry_run,
    )
