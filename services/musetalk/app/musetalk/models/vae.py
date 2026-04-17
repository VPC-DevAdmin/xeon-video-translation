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

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = (1.0 / self.scaling_factor) * latents
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        # RGB → BGR for OpenCV consumers
        return image[..., ::-1]

    def get_latents_for_unet(self, img: np.ndarray) -> torch.Tensor:
        """Returns the (masked | reference) concatenated latents MuseTalk's UNet expects."""
        ref = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref)
        ref = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref)
        return torch.cat([masked_latents, ref_latents], dim=1)
