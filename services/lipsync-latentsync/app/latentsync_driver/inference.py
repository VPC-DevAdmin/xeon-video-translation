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

import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Resume / checkpoint cache
#
# The pipeline's expensive half (face detection + affine transform +
# the 20-step denoising loop per chunk) typically takes 20+ min on CPU.
# The cheap half (restore_video's per-frame kornia warp + ffmpeg mux)
# takes ~1 min. When the cheap half fails (we've seen OOM during
# restore several times), redoing the expensive half is pure waste.
#
# The cache is content-addressable: a sha256 over (video bytes, audio
# bytes, steps, guidance, seed). Any re-run with identical inputs hits
# the cache automatically. Different inputs → different hash → cache
# miss → full run. No manual "resume this job" command needed.
#
# Cache lives under /models/cache/latentsync_denoise/ so it persists
# across container restarts via the shared `models` volume.
# ------------------------------------------------------------------ #

_DENOISE_CACHE_SUBDIR = "cache/latentsync_denoise"

# Built pipeline reused across requests; see run().
_PIPELINE_CACHE: dict = {}
import threading as _threading
_PIPELINE_CACHE_LOCK = _threading.Lock()


def _compute_content_hash(
    video_path: Path, audio_path: Path,
    steps: int, guidance: float, seed: int | None,
) -> str:
    """Return a 16-char hex hash identifying the inputs to the denoise loop.

    Reads the video and audio bytes through a streaming hash so we don't
    pull large files into memory. Keyed on the knobs that affect the
    denoising output — a change in any of them forces a fresh run.

    Also keys on the UNet config name, because caches generated against
    one config (e.g. stage2.yaml at 256 res) are dimensionally
    incompatible with another (stage2_512.yaml at 512 res). Without this
    key component, switching configs would silently hit old incompatible
    caches and produce either crashes or degraded output.
    """
    h = hashlib.sha256()
    for p in (video_path, audio_path):
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                h.update(chunk)
    h.update(
        f"|steps={steps}|guidance={guidance:.4f}"
        f"|seed={seed}|config={_UNET_CONFIG_NAME}".encode()
    )
    return h.hexdigest()[:16]


def _resolve_checkpoint_path(
    model_cache_dir: Path,
    video_path: Path, audio_path: Path,
    steps: int, guidance: float, seed: int | None,
) -> Path | None:
    """Return the cache path for these inputs, or None if caching disabled."""
    if os.environ.get("LATENTSYNC_IGNORE_DENOISE_CACHE", "").lower() in ("1", "true", "yes"):
        return None
    digest = _compute_content_hash(video_path, audio_path, steps, guidance, seed)
    cache_dir = model_cache_dir / _DENOISE_CACHE_SUBDIR
    return cache_dir / f"{digest}.pt"


# Path to the vendored configs dir (app/configs/). Resolved at import
# time so stacktraces make the layout obvious when a file is missing.
#
# We use stage2_512.yaml specifically because that's what the official
# LatentSync gradio_app.py loads for the released LatentSync-1.6
# checkpoint. The checkpoint was trained at 512x512 canonical resolution
# — running it against stage2.yaml (256x256) loads cleanly (architecture
# matches) but produces degraded structural outputs because the model is
# operating at half its intended working resolution. Configurable via
# LATENTSYNC_UNET_CONFIG in case future releases add another variant.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_UNET_CONFIG_NAME = os.environ.get("LATENTSYNC_UNET_CONFIG", "stage2_512.yaml")
_UNET_CONFIG_PATH = _CONFIGS_DIR / "unet" / _UNET_CONFIG_NAME


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

    # LATENTSYNC_DTYPE (fp32|fp16|bf16) is the device-neutral knob;
    # LATENTSYNC_IPEX_DTYPE is honoured as the legacy CPU spelling.
    # Default: fp16 on CUDA (upstream's tested GPU path), fp32 on CPU
    # (the CPU bf16 jitter artifact, see docs/latentsync-pipeline.md §5).
    default = "fp16" if _resolve_device().type == "cuda" else "fp32"
    choice = (
        os.environ.get("LATENTSYNC_DTYPE")
        or os.environ.get("LATENTSYNC_IPEX_DTYPE", default)
    ).lower()
    if choice in ("bf16", "bfloat16"):
        return torch.bfloat16
    if choice in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _resolve_device():
    """Compute device from the DEVICE env var: cpu (default) | cuda | auto.

    docker-compose.gpu.yml sets DEVICE=cuda and pins the container to one
    card via CUDA_VISIBLE_DEVICES, so `cuda` is always cuda:0 in-container.
    """
    import torch

    choice = os.environ.get("DEVICE", "cpu").lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={choice} requested but CUDA is unavailable")
    if choice.startswith("cuda"):
        _enable_cuda_fast_math(torch)
    return torch.device(choice)


def _enable_cuda_fast_math(torch) -> None:
    """Blackwell/Ampere+ settings that are free on this workload: TF32 for
    the fp32 matmuls that remain (VAE scaling, whisper features), cuDNN
    autotuning for the fixed-shape 3D UNet convs, and fp16 reduced-precision
    reductions. Idempotent; also called inside each denoise worker."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = os.environ.get("GPU_CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


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


def _run_impl(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    weight_paths: WeightPaths,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    request_temp_dir: str | None = None,
    face_track_source: Path | str | None = None,
    face_track_offset_frames: int = 0,
    prepare_only: bool = False,
    persona_key: str | None = None,
) -> InferenceResult:
    """Run LatentSync inference. All tensor ops are CPU float32.

    ``face_track_source`` names the full clip a windowed ``video_path`` was
    cut from (25 fps grid, ``face_track_offset_frames`` = first frame of the
    window). The landmark track for that source is built once, cached by
    content, and sliced per window so detection runs once per clip.

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
    if not prepare_only and not video_path.exists():
        raise RuntimeError(f"video_path not found: {video_path}")
    if not prepare_only and not audio_path.exists():
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

    # Hardware encode for the intermediate re-encode and the final write
    # when on CUDA (the container needs the `video` driver capability;
    # util.py falls back to libx264 if ffmpeg rejects NVENC).
    os.environ.setdefault(
        "LATENTSYNC_VIDEO_ENCODER",
        "h264_nvenc" if _resolve_device().type == "cuda" else "libx264",
    )
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

    device = _resolve_device()
    compute_dtype = _ipex_dtype()
    # CPU: weights stay fp32 and IPEX/autocast handle the reduced-precision
    # math. CUDA: mirror upstream, which loads VAE + UNet directly in the
    # compute dtype (fp16) and runs without autocast.
    dtype = compute_dtype if device.type == "cuda" else torch.float32
    optimize = _ipex_optimize if device.type == "cpu" else (lambda m, name: m)
    log.info(
        "LatentSync device=%s weight_dtype=%s compute_dtype=%s",
        device, str(dtype).rsplit(".", 1)[-1], str(compute_dtype).rsplit(".", 1)[-1],
    )

    # Seed for reproducibility. accelerate.set_seed seeds torch, numpy,
    # and python random in one call. If no seed is given we just log
    # whatever torch picked so reruns can be matched post-hoc.
    if seed is not None and seed >= 0:
        set_seed(int(seed))
    log.info("torch initial seed: %d", torch.initial_seed())

    # Debug knob: force deterministic algorithms. Meaningful when
    # hunting "two identical runs produce different jitter" bugs.
    # `warn_only=True` so ops without a deterministic implementation
    # (e.g. some kornia kernels) log a warning rather than raising —
    # lets us still complete a run but flag which ops are sources
    # of non-determinism.
    if os.environ.get("LATENTSYNC_DETERMINISTIC", "").lower() in ("1", "true", "yes"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            log.info("LATENTSYNC_DETERMINISTIC=1 — torch deterministic mode on")
        except Exception as e:
            log.warning("couldn't enable deterministic mode (%s)", e)

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
            f"Check app/configs/unet/{_UNET_CONFIG_NAME} vs the weight release."
        )

    # Models are expensive to build (~18 s on the XE7740 with four UNet
    # replicas) and were rebuilt on every request. Cache the finished
    # pipeline per (device, dtype, config, replica count, deepcache) and
    # reuse it; the per-request knobs (steps, guidance, seed) are passed
    # at call time and do not touch the models.
    cache_key = (
        str(device), str(dtype), _UNET_CONFIG_NAME,
        os.environ.get("LATENTSYNC_UNET_REPLICAS", "0"),
        os.environ.get("LATENTSYNC_SHARD_MODE", "process"),
        os.environ.get("LATENTSYNC_ENABLE_DEEPCACHE", "1"),
        str(weight_paths.unet), str(weight_paths.whisper_tiny),
    )

    def _build_pipeline():
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
        vae = optimize(vae, "vae")

        # --- UNet ----------------------------------------------------------
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            str(weight_paths.unet),
            device=str(device),
        )
        unet = unet.to(dtype=dtype)
        unet = optimize(unet, "unet")

        # --- Pipeline ------------------------------------------------------
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(device)

        # --- UNet replicas for sharded denoise (GPU track, G5) ------------
        # One UNet copy per visible CUDA device (NVIDIA_VISIBLE_DEVICES in the
        # compose overlay). The pipeline splits the 16-frame chunks across
        # them round-robin; everything else stays on cuda:0. ~2.6 GB fp16 per
        # copy. LATENTSYNC_UNET_REPLICAS caps the count (0 = all visible).
        n_visible = torch.cuda.device_count() if device.type == "cuda" else 1
        want = int(os.environ.get("LATENTSYNC_UNET_REPLICAS", "0") or 0)
        n_replicas = max(1, min(want or n_visible, n_visible))
        # process (default): one worker process per GPU, each with its own
        #   interpreter — see shard_workers.py for why threads capped at ~2x.
        # thread: in-process deep copies of the UNet/VAE driven by threads.
        shard_mode = os.environ.get("LATENTSYNC_SHARD_MODE", "process").lower()
        unet_replicas = [unet]
        vae_replicas = [vae]
        pipeline.denoise_pool = None
        if n_replicas > 1 and shard_mode == "process":
            from .shard_workers import DenoisePool, build_args_for

            t_pool = time.perf_counter()
            pipeline.denoise_pool = DenoisePool.start(
                list(range(n_replicas)),
                build_args_for(OmegaConf.to_container(config.model), weight_paths.unet, _CONFIGS_DIR, dtype),
            )
            log.info(
                "denoise pool: %d worker processes on %s, ready in %.1fs",
                n_replicas, [torch.cuda.get_device_name(k) for k in range(n_replicas)],
                time.perf_counter() - t_pool,
            )
        elif n_replicas > 1:
            import copy
            for k in range(1, n_replicas):
                unet_replicas.append(copy.deepcopy(unet).to(torch.device(f"cuda:{k}")))
                vae_replicas.append(copy.deepcopy(vae).to(torch.device(f"cuda:{k}")))
            log.info(
                "UNet replicated to %d GPUs (thread mode): %s",
                n_replicas, [torch.cuda.get_device_name(k) for k in range(n_replicas)],
            )
        pipeline.unet_replicas = unet_replicas
        pipeline.vae_replicas = vae_replicas

        # --- DeepCache -----------------------------------------------------
        # Caches intermediate UNet feature maps on one denoising step and
        # reuses them on subsequent steps ("skip" steps). Net effect: about
        # 30% fewer UNet calls for a given num_inference_steps, with
        # negligible visual drift at cache_interval=3.
        #
        # Upstream LatentSync uses exactly these params (scripts/inference.py):
        #   helper.set_params(cache_interval=3, cache_branch_id=0)
        # We default them on and expose an env toggle for debugging. IPEX
        # and DeepCache stack — the speedup is multiplicative, not additive.
        deepcache_enabled = os.environ.get(
            "LATENTSYNC_ENABLE_DEEPCACHE", "1",
        ).lower() in ("1", "true", "yes")
        if deepcache_enabled and n_replicas > 1:
            # DeepCacheSDHelper patches `pipeline.unet` with per-call cache
            # state; with replicas running concurrently on other devices that
            # state would be wrong for them and racy for this one. Sharding
            # wins ~Nx, DeepCache ~1.3x, so sharding takes precedence.
            log.info("DeepCache disabled: denoise sharded over %d GPUs", n_replicas)
            deepcache_enabled = False
        if deepcache_enabled:
            try:
                from DeepCache import DeepCacheSDHelper
                helper = DeepCacheSDHelper(pipe=pipeline)
                helper.set_params(cache_interval=3, cache_branch_id=0)
                helper.enable()
                log.info("DeepCache enabled (cache_interval=3, cache_branch_id=0)")
            except Exception as e:
                # DeepCache is a speedup, not a correctness piece. If its
                # monkey-patching ever bites an upstream diffusers API
                # change, fall back to vanilla rather than failing the run.
                log.warning(
                    "DeepCache enable failed (%s); running without it", e,
                )


        return pipeline

    with _PIPELINE_CACHE_LOCK:
        pipeline = _PIPELINE_CACHE.get(cache_key)
        pool = getattr(pipeline, "denoise_pool", None) if pipeline is not None else None
        if pool is not None and not pool.alive():
            # A worker died (OOM, shm exhaustion, ...). Rebuild rather than
            # fail every later request until someone restarts the service.
            log.warning("denoise pool has dead workers; rebuilding pipeline + pool")
            try:
                pool.close()
            except Exception:
                pass
            _PIPELINE_CACHE.pop(cache_key, None)
            pipeline = None
        if pipeline is None:
            t_build = time.perf_counter()
            pipeline = _build_pipeline()
            for old in _PIPELINE_CACHE.values():
                old_pool = getattr(old, "denoise_pool", None)
                if old_pool is not None:
                    old_pool.close()
            _PIPELINE_CACHE.clear()
            _PIPELINE_CACHE[cache_key] = pipeline
            log.info("pipeline built in %.1fs and cached", time.perf_counter() - t_build)
        else:
            log.info("pipeline reused from cache")
    # Bind the mask image path to the vendored asset so the pipeline
    # finds it without depending on the container's working directory.
    mask_image_path = Path(__file__).resolve().parent.parent / "latentsync" / "utils" / "mask.png"
    if not mask_image_path.exists():
        raise FileNotFoundError(
            f"vendored mask image missing: {mask_image_path}. "
            "Image was likely built without the latentsync/utils/ assets."
        )

    if prepare_only:
        # Warm-up / prepare path: models, worker pool and the ONNX detector
        # sessions are built and cached; no media is touched here.
        pipeline.ensure_image_processor(int(config.data.resolution), str(mask_image_path))
        return pipeline, config, mask_image_path

    # Temp dir for intermediate frames/audio — cleaned up by the pipeline
    # internally. We write under /tmp so the `jobs` volume only gets
    # the final mp4.
    temp_dir = Path(request_temp_dir or "/tmp/latentsync_work")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume / denoise cache --------------------------------------
    # See the module header for the rationale. One-line summary: we
    # hash (video + audio + knobs), look for a post-denoise tensor
    # cached under /models/cache/latentsync_denoise/<hash>.pt, and
    # skip straight to restore_video if we find one.
    model_cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
    checkpoint_path = _resolve_checkpoint_path(
        model_cache_dir, video_path, audio_path, steps, guidance, seed,
    )
    if (
        checkpoint_path is not None
        and device.type == "cuda"
        and os.environ.get("LATENTSYNC_DENOISE_CACHE", "0").lower() not in ("1", "true", "yes")
    ):
        # The resume cache was a CPU-era necessity (a retry saved hours).
        # On GPU the torch.save of ~2 GB of frames costs ~30 s per job,
        # which is a sizeable slice of the whole run. Opt back in with
        # LATENTSYNC_DENOISE_CACHE=1.
        log.info("denoise cache: off on CUDA (LATENTSYNC_DENOISE_CACHE=1 to enable)")
        checkpoint_path = None
    if checkpoint_path is None:
        log.info("denoise cache: disabled (LATENTSYNC_IGNORE_DENOISE_CACHE)")
    elif checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        log.info(
            "denoise cache HIT: %s (%.1f MB) — skipping expensive half",
            checkpoint_path.name, size_mb,
        )
    else:
        log.info(
            "denoise cache MISS: will write to %s after denoise loop",
            checkpoint_path.name,
        )

    # --- Real-progress reporting -------------------------------------
    # Write a small JSON file on the shared /jobs volume that the
    # backend's orchestrator polls every 5 s. Replaces the "elapsed
    # vs estimated ETA" heuristic that consistently hit 98% two
    # minutes into a run and plateaued for hours.
    #
    # File location mirrors the output path's parent, so each job gets
    # its own progress file. Stale files from prior failed runs are
    # overwritten on the first emit. The backend tolerates a missing
    # file (falls back to the time-based estimate).
    import json as _json
    progress_file = output_path.parent / "latentsync_progress.json"
    # Clear any stale content from a previous attempt before the first
    # emit — otherwise the backend could briefly see 100%-from-a-past-
    # run while this run is at step 0.
    try:
        progress_file.unlink()
    except FileNotFoundError:
        pass
    except Exception as _e:
        log.warning("couldn't clear stale progress file (%s); continuing", _e)

    def _write_progress(phase: str, percent: float) -> None:
        # Best-effort: a progress write failure must never break the
        # inference run.
        try:
            progress_file.write_text(_json.dumps({
                "phase": phase,
                "percent": percent,
                "timestamp": time.time(),
            }))
        except Exception:
            pass

    # Wrap the pipeline call in CPU autocast when bf16 is requested.
    # autocast's allowlist keeps BatchNorm / LayerNorm / Softmax at
    # fp32 for numerical stability; Conv / Linear / etc. run at bf16.
    # When compute_dtype is fp32, autocast becomes a no-op.
    # On CUDA the weights already carry the compute dtype, so autocast
    # stays off; on CPU it is the only way to get reduced precision.
    autocast_enabled = device.type == "cpu" and compute_dtype != torch.float32
    autocast_ctx = torch.autocast(
        device_type=device.type,
        dtype=compute_dtype if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    )
    log.info(
        "pipeline starting: steps=%d guidance=%.2f compute_dtype=%s autocast=%s",
        steps, guidance,
        str(compute_dtype).rsplit(".", 1)[-1],
        autocast_enabled,
    )

    face_track = _load_face_track(pipeline, config, mask_image_path, face_track_source, face_track_offset_frames)

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
            # Passed through **kwargs; the pipeline reads it when present
            # and handles None/missing gracefully.
            denoise_checkpoint_path=(
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            progress_callback=_write_progress,
            face_track=face_track,
            # Resident persona conditioning on the workers (video assistant
            # replies redraw the same footage every turn).
            persona_key=persona_key,
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


_RUN_LOCK = _threading.Lock()
_PREPARE_LOCK = _threading.Lock()


def _load_face_track(pipeline, config, mask_image_path, face_track_source, face_track_offset_frames):
    """Landmarks for the source clip a window was cut from (cached by content)."""
    if not face_track_source:
        return None
    from . import face_track as _face_track

    source = Path(face_track_source)
    if not source.exists():
        raise RuntimeError(f"face_track_source not found: {source}")
    processor = pipeline.ensure_image_processor(int(config.data.resolution), str(mask_image_path))
    model_cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/models"))
    track_started = time.perf_counter()
    track = _face_track.load_or_build(
        source,
        model_cache_dir=model_cache_dir,
        fps=25,
        extract=processor.try_extract_landmarks3,
        smooth_window=int(os.environ.get("LATENTSYNC_LANDMARK_SMOOTH_WINDOW", "5")),
        max_miss_ratio=float(os.environ.get("LATENTSYNC_MAX_MISSING_FACE_RATIO", "0.5")),
        frame_budget_bytes=int(os.environ.get("LATENTSYNC_FRAME_BUDGET_MB", "8192")) * 1024 * 1024,
    )
    log.info("face track ready in %.1fs (%d frames, %d without a face); window offset %d",
             time.perf_counter() - track_started, len(track["landmarks"]),
             int((~track["visible"]).sum()), int(face_track_offset_frames or 0))
    return {"landmarks": track["landmarks"], "visible": track["visible"],
            "offset": int(face_track_offset_frames or 0)}


def prepare(video_path, audio_path, weight_paths, face_track_source=None, face_track_offset_frames=0) -> dict:
    """Decode, warp and compute audio features for a window ahead of its
    /lipsync request. Runs outside _RUN_LOCK so it overlaps the previous
    window's denoise; the coordinator thread is mostly waiting then."""
    started = time.perf_counter()
    with _PREPARE_LOCK:
        pipeline, config, mask_image_path = _run_impl(
            video_path=Path(video_path), audio_path=Path(audio_path), output_path=Path("/nonexistent"),
            weight_paths=weight_paths, prepare_only=True,
        )
        face_track = _load_face_track(pipeline, config, mask_image_path, face_track_source, face_track_offset_frames)
        import torch

        with torch.no_grad():
            result = pipeline.prepare_ahead(str(video_path), str(audio_path), 25, face_track)
    result["total_seconds"] = round(time.perf_counter() - started, 2)
    log.info("prepared ahead: %s (%d frames) in %.1fs", Path(video_path).name, result["frames"], result["total_seconds"])
    return result


def warmup(weight_paths) -> float:
    """Build and cache everything a first request would otherwise pay for
    (pipeline ~24 s, denoise pool ~14 s, ONNX detector sessions). Returns
    the seconds spent. Called from the service startup hook."""
    started = time.perf_counter()
    with _RUN_LOCK:
        _run_impl(
            video_path=Path("/nonexistent"), audio_path=Path("/nonexistent"),
            output_path=Path("/nonexistent"), weight_paths=weight_paths, prepare_only=True,
        )
    return time.perf_counter() - started


def run(**kwargs):
    import tempfile
    import subprocess
    import json
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height:format=duration", "-of", "json",
                            str(kwargs["video_path"])], capture_output=True, check=True, timeout=30)
    info = json.loads(probe.stdout)
    video = info["streams"][0]
    size = int(video["width"]) * int(video["height"]) * 3 * 25 * float(info["format"]["duration"])
    budget = int(os.environ.get("LATENTSYNC_FRAME_BUDGET_MB", "8192")) * 1024 * 1024
    if size > budget:
        raise RuntimeError("video exceeds decoded-frame budget; use shorter clips or lower resolution")
    with _RUN_LOCK, tempfile.TemporaryDirectory(prefix="latentsync-") as directory:
        try:
            return _run_impl(**kwargs, request_temp_dir=directory)
        except BaseException:
            for pipeline in _PIPELINE_CACHE.values():
                pool = getattr(pipeline, "denoise_pool", None)
                if pool is not None:
                    pool.close()
            _PIPELINE_CACHE.clear()
            raise


def shutdown_workers():
    for pipeline in _PIPELINE_CACHE.values():
        pool = getattr(pipeline, "denoise_pool", None)
        if pool is not None:
            pool.close()
    _PIPELINE_CACHE.clear()
