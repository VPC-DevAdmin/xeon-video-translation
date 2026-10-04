"""MuseTalk VAE wrapper.

Adapted from https://github.com/TMElyralab/MuseTalk (MIT code). Device
selection explicit (CPU by default); in-memory ndarray preprocess path
preferred over file paths.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL


class VAE:
    """Wraps a Stable Diffusion 1.5 VAE for MuseTalk's 256x256 face latent space."""

    def __init__(
        self,
        model_path: str | Path,
        resized_img: int = 256,
        use_float16: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.model_path = str(model_path)
        self.vae = AutoencoderKL.from_pretrained(self.model_path)
        self.device = torch.device(device)
        self.vae.to(self.device)

        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False

        self.scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self._resized_img = resized_img
        self._mask_tensor = self.get_mask_tensor()

        self.vae.eval()

    def get_mask_tensor(self) -> torch.Tensor:
        mask = torch.zeros((self._resized_img, self._resized_img))
        mask[: self._resized_img // 2, :] = 1
        return (mask >= 0.5).float()

    def preprocess_img(self, img: np.ndarray, half_mask: bool = False) -> torch.Tensor:
        """Preprocess a single BGR ndarray frame to the VAE input shape (1,3,H,W)."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (self._resized_img, self._resized_img), interpolation=cv2.INTER_LANCZOS4
        )
        arr = rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        x = self.transform(x)
        x = x.unsqueeze(0).to(self.device)
        return x

    def encode_latents(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            init = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        return self.scaling_factor * init.sample()

    def decode_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Return NCHW uint8 BGR on the inference device."""
        with torch.no_grad():
            image = self.vae.decode((latents / self.scaling_factor).to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1).float().mul(255).round().to(torch.uint8)
        return image[:, [2, 1, 0]].contiguous()

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        return self.decode_latents_tensor(latents).permute(0, 2, 3, 1).cpu().numpy()

    def get_latents_for_unet(self, img: np.ndarray) -> torch.Tensor:
        """Returns the (masked | reference) concatenated latents MuseTalk's UNet expects."""
        ref = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref)
        ref = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref)
        return torch.cat([masked_latents, ref_latents], dim=1)

    def get_latents_for_unet_batch(self, crops_bgr_256: list[np.ndarray]) -> torch.Tensor:
        """Batched `get_latents_for_unet` for N already-256x256 BGR crops.

        Returns (N, 8, 32, 32). The per-frame version costs ~270 ms on the
        GPU box because its preprocessing runs as many tiny CPU tensor ops;
        here the uint8 crops are stacked once, moved to the device, and
        normalised there, and the masked + reference encodes run as one
        call of 2N images. ~10x faster per frame at N=32.
        """
        arr = np.stack([c[:, :, ::-1] for c in crops_bgr_256])  # BGR -> RGB, (N,H,W,3)
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)  # (N,3,H,W) in [0,1]
        mask = (self._mask_tensor > 0.5).to(self.device)
        masked = x * mask  # broadcast over (N,3,H,W)
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        both = torch.cat([masked, x], dim=0)
        both = (both - mean) / std
        lat = self.encode_latents(both)  # (2N,4,32,32)
        n = x.shape[0]
        return torch.cat([lat[:n], lat[n:]], dim=1)
