"""Occlusion-aware paste-back for LatentSync.

LatentSync pastes the whole generated face crop back wherever the landmark
track says the face is. When a hand or an object passes in front of the face
the track either keeps the last good landmarks (detection lost) or still finds
the eyes and nose (mouth covered), and the generated mouth lands on the object
(seen on 5 Oct 2026: a chocolate box held up in front of the speaker).

Two layers fix this:

* **Pixel mask.** BiSeNet face parsing on the *source* face crop labels every
  pixel (skin, nose, lips, hair, background...). Only face pixels receive the
  generated face; an occluder keeps the source pixels. Because the generated
  crop equals the source outside the mouth region, excluding pixels is harmless
  anywhere else, so the mask is dilated, feathered and averaged over three
  frames to stay stable rather than tight.
* **Frame gate.** Frames where detection failed (and a margin around them)
  are not pasted at all, with a short alpha ramp in and out, since the carried
  landmarks say nothing about where the face is.

Known residual: a bare hand over the mouth is skin to the parser and may still
get a mouth; the frame gate catches it only if detection drops as well.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# BiSeNet (CelebAMask-HQ) classes that are the face itself. Hair, hat, neck,
# cloth, ears, earrings and background are left to the source frame.
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
DEFAULT_WEIGHTS_DIR = "/models/musetalk/face-parse-bisent"


def enabled() -> bool:
    return os.environ.get("LATENTSYNC_OCCLUSION_MASK", "1") == "1"


def occlusion_alpha(visible, margin: int = 1, ramp: int = 3) -> np.ndarray:
    """Per-frame paste weight from per-frame detection success.

    Frames without a detected face, and `margin` frames either side of them,
    get 0; the weight then rises linearly to 1 over `ramp` frames so the mouth
    neither pops in nor out."""
    visible = np.asarray(visible, dtype=bool)
    n = len(visible)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    blocked = ~visible
    if margin > 0 and blocked.any():
        grown = blocked.copy()
        for shift in range(1, margin + 1):
            grown[shift:] |= blocked[:-shift]
            grown[:-shift] |= blocked[shift:]
        blocked = grown
    if not blocked.any():
        return np.ones(n, dtype=np.float32)
    # Distance (in frames) from each frame to the nearest blocked frame.
    distance = np.full(n, n, dtype=np.int64)
    last = None
    for i in range(n):
        if blocked[i]:
            last = i
        if last is not None:
            distance[i] = i - last
    last = None
    for i in range(n - 1, -1, -1):
        if blocked[i]:
            last = i
        if last is not None:
            distance[i] = min(distance[i], last - i)
    alpha = np.clip(distance / float(max(ramp, 1)), 0.0, 1.0)
    return alpha.astype(np.float32)


def face_mask_from_parsing(parsing, dilate: int = 9, feather: int = 15):
    """(N,H,W) class map -> (N,1,H,W) float mask in [0,1]: 1 on face pixels,
    dilated by `dilate` px, feathered with a `feather` px Gaussian, then
    averaged over three frames. Torch-only so it runs where the parse ran."""
    import torch
    import torch.nn.functional as F

    classes = torch.tensor(FACE_CLASSES, device=parsing.device)
    mask = torch.isin(parsing, classes).to(torch.float32).unsqueeze(1)
    if dilate > 1:
        k = dilate if dilate % 2 else dilate + 1
        mask = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    if feather > 1:
        k = feather if feather % 2 else feather + 1
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        x = torch.arange(k, device=mask.device, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g = (g / g.sum()).view(1, 1, 1, k)
        mask = F.conv2d(F.pad(mask, (k // 2, k // 2, 0, 0), mode="replicate"), g)
        mask = F.conv2d(F.pad(mask, (0, 0, k // 2, k // 2), mode="replicate"), g.transpose(2, 3))
    if len(mask) >= 3:
        padded = torch.cat([mask[:1], mask, mask[-1:]], dim=0)
        mask = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    return mask.clamp_(0.0, 1.0)


class FaceParser:
    """Lazy BiSeNet wrapper. `masks(crops)` takes (N,3,H,W) uint8 RGB face crops
    (LatentSync's canonical crops) and returns (N,1,out_h,out_w) float masks on
    the same device, or None when parsing is disabled or weights are missing."""

    def __init__(self, device, weights_dir: str | None = None):
        self.device = device
        self.weights_dir = Path(weights_dir or os.environ.get("LATENTSYNC_FACE_PARSE_DIR", DEFAULT_WEIGHTS_DIR))
        self._net = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        import torch

        from .face_parsing import BiSeNet

        weights = self.weights_dir / "79999_iter.pth"
        resnet = self.weights_dir / "resnet18-5c106cde.pth"
        if not weights.exists() or not resnet.exists():
            raise FileNotFoundError(f"face parsing weights missing under {self.weights_dir}")
        net = BiSeNet(resnet_path=str(resnet), n_classes=19)
        net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False))
        net.to(self.device).eval()
        return net

    def available(self) -> bool:
        if not enabled() or self._failed:
            return False
        if self._net is None:
            with self._lock:
                if self._net is None and not self._failed:
                    try:
                        self._net = self._load()
                        log.info("face parser ready (%s)", self.weights_dir)
                    except Exception as exc:
                        self._failed = True
                        log.warning("face parsing unavailable, occluders will not be masked: %s", exc)
        return self._net is not None

    def masks(self, crops, out_size: tuple[int, int], batch_size: int = 32):
        """`out_size` is (height, width) of the warp crop the mask is applied to."""
        if not self.available():
            return None
        import torch
        import torch.nn.functional as F

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        parsed = []
        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                x = crops[start:start + batch_size].to(self.device, non_blocking=True).float().div_(255.0)
                if x.shape[-2:] != (512, 512):
                    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
                x = (x - mean) / std
                parsed.append(self._net(x)[0].argmax(1).to(torch.uint8))
            parsing = torch.cat(parsed, dim=0)
            mask = face_mask_from_parsing(parsing)
            mask = F.interpolate(mask, size=out_size, mode="bilinear", align_corners=False)
        return mask
