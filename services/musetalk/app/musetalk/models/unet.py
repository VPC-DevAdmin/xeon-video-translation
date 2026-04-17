"""MuseTalk UNet wrapper.

Adapted from https://github.com/TMElyralab/MuseTalk (MIT code). CPU-forced
device selection; `map_location` honored even when CUDA is available so the
same weights file can be loaded anywhere.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 384, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        return x + pe.to(x.device)


class UNet:
    def __init__(
        self,
        unet_config: str,
        model_path: str,
        use_float16: bool = False,
        device: torch.device | str | None = None,
    ):
        with open(unet_config, "r") as f:
            cfg = json.load(f)
        self.model = UNet2DConditionModel(**cfg)
        self.pe = PositionalEncoding(d_model=384)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        if use_float16:
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
