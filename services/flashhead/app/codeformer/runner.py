"""Load CodeFormer, run it on cropped faces, blend back.

Wiring mirrors the upstream `inference_codeformer.py` at a minimum: lazy
singleton model load, fp32 on CPU, no GFPGAN/Real-ESRGAN fallback. The
blend weight and `fidelity_weight` (CodeFormer's `w` param) are exposed as
runtime knobs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from .codeformer_arch import CodeFormer
from .face_alignment import FACE_SIZE, align_to_canonical, paste_back

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model loader
# --------------------------------------------------------------------------- #

_model: CodeFormer | None = None
_model_lock = Lock()


def _weight_path() -> Path:
    root = Path(os.environ.get("MODEL_CACHE_DIR", "/models")) / "codeformer"
    return root / "codeformer.pth"


def _load_model(weight_path: Path, device: torch.device) -> CodeFormer:
    """Build the CodeFormer module, load the checkpoint, move to device."""
    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )
    if not weight_path.exists():
        raise FileNotFoundError(
            f"CodeFormer weights missing at {weight_path}; run "
            "`docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh`"
        )

    # Upstream ships the state under `params_ema`; some forks ship bare
    # state dicts. Handle both.
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        state = ckpt["params_ema"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        state = ckpt["params"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=False)
    net.eval()
    net.to(device)
    return net


def _get_model(device: torch.device) -> CodeFormer:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info("Loading CodeFormer from %s", _weight_path())
        _model = _load_model(_weight_path(), device)
        return _model


# --------------------------------------------------------------------------- #
# Per-frame restore
# --------------------------------------------------------------------------- #


def _fidelity_weight() -> float:
    raw = os.environ.get("MUSETALK_FACE_RESTORE_FIDELITY", "0.7")
    try:
        value = float(raw)
    except ValueError:
        log.warning("bad MUSETALK_FACE_RESTORE_FIDELITY=%r; using 0.7", raw)
        return 0.7
    return max(0.0, min(1.0, value))


def _blend_weight() -> float:
    raw = os.environ.get("MUSETALK_FACE_RESTORE_BLEND", "0.6")
    try:
        value = float(raw)
    except ValueError:
        log.warning("bad MUSETALK_FACE_RESTORE_BLEND=%r; using 0.6", raw)
        return 0.6
    return max(0.0, min(1.0, value))


def restore_frame(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
    device: torch.device,
    fidelity: float | None = None,
    blend: float | None = None,
) -> np.ndarray:
    """Run CodeFormer on the face region of `frame_bgr` and return the
    enhanced full-frame image.

    `kps` are SCRFD's 5-point keypoints in the frame's coordinate space.
    If `kps` is None or alignment fails, the original frame is returned.

    `fidelity` and `blend` override the env-driven defaults when provided.
    """
    try:
        aligned, M = align_to_canonical(frame_bgr, kps)
    except Exception as e:
        log.warning("CodeFormer alignment failed (%s); skipping this frame", e)
        return frame_bgr

    # BGR → RGB, normalize to [-1, 1] per CodeFormer convention.
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    tensor = tensor.to(device)

    model = _get_model(device)
    w = fidelity if fidelity is not None else _fidelity_weight()
    w = max(0.0, min(1.0, float(w)))
    with torch.no_grad():
        # CodeFormer returns (output, logits, lq_feat) in default mode; we
        # only need the image. `w` is the identity-vs-quality slider.
        out = model(tensor, w=w, adain=True)
        if isinstance(out, tuple):
            out = out[0]

    # [-1, 1] → [0, 255] uint8 BGR.
    out_img = (out.squeeze(0).clamp(-1, 1).add(1.0).mul(127.5)
                    .permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    b = blend if blend is not None else _blend_weight()
    b = max(0.0, min(1.0, float(b)))
    return paste_back(
        original=frame_bgr,
        restored_512=out_bgr,
        M=M,
        blend_weight=b,
    )


def preload(device: torch.device | None = None) -> bool:
    """Force-load the model so the first real request doesn't stall.

    Returns True on success, False if weights are missing or anything else
    went wrong (callers log and continue).
    """
    dev = device or torch.device("cpu")
    try:
        _get_model(dev)
        return True
    except Exception as e:
        log.warning("CodeFormer preload failed: %s", e)
        return False
