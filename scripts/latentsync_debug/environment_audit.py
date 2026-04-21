#!/usr/bin/env python3
"""LatentSync environment audit — lock down what's actually running.

Prints:
  - Exact versions of every dependency LatentSync touches
  - SHA256 of every model checkpoint on disk
  - Python / CPU / MKL / OpenMP thread configuration
  - Numerical precision flags

Run this inside the lipsync-latentsync container before every
debugging session. Save the output and diff against a known-good
baseline — subtle dependency drift is a real source of "same code,
different numbers."

Usage (from repo root):
    docker compose exec -T lipsync-latentsync python /app/scripts/environment_audit.py

Or equivalently (if the script is bind-mounted at /app/scripts):
    docker compose exec -T lipsync-latentsync \\
        python /app/app/latentsync_driver/../../../scripts/environment_audit.py

No dependencies beyond what the service already has.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

# Modules we care about pinning / inspecting.
_MODULES = [
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
    "whisper",
    "imageio",
    "scenedetect",
    "kornia",
    "face_alignment",
    "intel_extension_for_pytorch",
    "scipy",
    "numpy",
    "PIL",
    "DeepCache",
]


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def module_versions() -> dict:
    out = {}
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            continue
        ver = getattr(mod, "__version__", None)
        out[name] = {"ok": True, "version": ver}
    return out


def checkpoint_hashes(root: Path) -> dict:
    """SHA256 every known checkpoint file. Lets us confirm the bytes
    on disk match what we expect — detects silent partial downloads
    and swapped files.
    """
    expected = [
        "latentsync/latentsync_unet.pt",
        "latentsync/stable_syncnet.pt",
        "latentsync/whisper/tiny.pt",
        # sd-vae-ft-mse is downloaded via HF hub into HF_HOME; the
        # actual .bin file lives under a hash-pathed cache dir. Best
        # effort here.
    ]
    out = {}
    for rel in expected:
        path = root / rel
        if not path.exists():
            out[rel] = {"ok": False, "reason": "missing"}
            continue
        size = path.stat().st_size
        out[rel] = {
            "ok": True,
            "size_bytes": size,
            "sha256": _sha256(path),
        }

    # Sweep HF cache for the SD VAE.
    hf_cache = Path(os.environ.get("HF_HOME", "/models/huggingface"))
    vae_bins = list(hf_cache.rglob("**/sd-vae-ft-mse/**/diffusion_pytorch_model*.bin")) + \
               list(hf_cache.rglob("**/sd-vae-ft-mse/**/diffusion_pytorch_model*.safetensors"))
    for bin_path in vae_bins:
        rel = f"huggingface/{bin_path.relative_to(hf_cache)}"
        out[rel] = {
            "ok": True,
            "size_bytes": bin_path.stat().st_size,
            "sha256": _sha256(bin_path),
        }

    return out


def torch_config() -> dict:
    try:
        import torch
    except ImportError:
        return {"ok": False, "error": "torch not importable"}

    info = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        # Deterministic algorithms — off by default in our stack. The
        # debug harness below flips them on. Reported here for audit.
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }

    # torch.__config__.show() is an enormous string. Include a hash
    # of it for quick change-detection without flooding the audit.
    try:
        cfg = torch.__config__.show()
        info["config_hash"] = hashlib.sha256(cfg.encode()).hexdigest()[:16]
        info["config_len"] = len(cfg)
    except Exception:
        pass

    # Capture IPEX status if present.
    try:
        import intel_extension_for_pytorch as ipex
        info["ipex_version"] = getattr(ipex, "__version__", None)
    except Exception as e:
        info["ipex_version"] = f"not-importable: {e}"

    return info


def env_flags() -> dict:
    """Selected env vars that affect numerical reproducibility."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "LD_PRELOAD",
        "LATENTSYNC_IPEX_DTYPE",
        "LATENTSYNC_ENABLE_DEEPCACHE",
        "LATENTSYNC_AFFINE_SMOOTH_WINDOW",
        "LATENTSYNC_STEPS",
        "LATENTSYNC_GUIDANCE",
        "LATENTSYNC_UNET_CONFIG",
        "LATENTSYNC_DRY_RUN",
        "HF_HOME",
        "MODEL_CACHE_DIR",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def system_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def main():
    report = {
        "system": system_info(),
        "torch": torch_config(),
        "env": env_flags(),
        "modules": module_versions(),
        "checkpoints": checkpoint_hashes(
            Path(os.environ.get("MODEL_CACHE_DIR", "/models")),
        ),
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
