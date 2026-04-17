"""Wav2Lip model architecture.

Vendored from https://github.com/Rudrabha/Wav2Lip (MIT-licensed code,
Rudrabha Mukhopadhyay et al., IIIT Hyderabad, 2020). The published
``wav2lip_gan.pth`` checkpoint loads directly into this definition.

Weights are **CC-BY-NC 4.0** — demo/research use only.
"""

from __future__ import annotations

import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        residual: bool = False,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 0,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv_block(x))


class Wav2Lip(nn.Module):
    """Classic Wav2Lip generator. Takes (audio_mel, face_concat) and returns
    a 3-channel 96×96 image.

    Inputs:
        audio_sequences: (B, 1, 80, 16)     — mel chunk
        face_sequences : (B, 6, 96, 96)     — [masked-lower | ref] concat on channel
    Output:
        image          : (B, 3, 96, 96)
    """

    def __init__(self):
        super().__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, 7, 1, 3)),                                      # 96
            nn.Sequential(
                Conv2d(16, 32, 3, 2, 1),                                                # 48
                Conv2d(32, 32, 3, 1, 1, residual=True),
                Conv2d(32, 32, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(32, 64, 3, 2, 1),                                                # 24
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(64, 128, 3, 2, 1),                                               # 12
                Conv2d(128, 128, 3, 1, 1, residual=True),
                Conv2d(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(128, 256, 3, 2, 1),                                              # 6
                Conv2d(256, 256, 3, 1, 1, residual=True),
                Conv2d(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(256, 512, 3, 2, 1),                                              # 3
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2d(512, 512, 3, 1, 0),                                              # 1
                Conv2d(512, 512, 1, 1, 0),
            ),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1, residual=True),
            Conv2d(32, 32, 3, 1, 1, residual=True),

            Conv2d(32, 64, 3, (3, 1), 1),
            Conv2d(64, 64, 3, 1, 1, residual=True),
            Conv2d(64, 64, 3, 1, 1, residual=True),

            Conv2d(64, 128, 3, 3, 1),
            Conv2d(128, 128, 3, 1, 1, residual=True),
            Conv2d(128, 128, 3, 1, 1, residual=True),

            Conv2d(128, 256, 3, (3, 2), 1),
            Conv2d(256, 256, 3, 1, 1, residual=True),

            Conv2d(256, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, 1, 1, 0)),
            nn.Sequential(
                Conv2dTranspose(1024, 512, 3, 1, 0),                                    # 3
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(1024, 512, 3, 2, 1, output_padding=1),                  # 6
                Conv2d(512, 512, 3, 1, 1, residual=True),
                Conv2d(512, 512, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(768, 384, 3, 2, 1, output_padding=1),                   # 12
                Conv2d(384, 384, 3, 1, 1, residual=True),
                Conv2d(384, 384, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(512, 256, 3, 2, 1, output_padding=1),                   # 24
                Conv2d(256, 256, 3, 1, 1, residual=True),
                Conv2d(256, 256, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(320, 128, 3, 2, 1, output_padding=1),                   # 48
                Conv2d(128, 128, 3, 1, 1, residual=True),
                Conv2d(128, 128, 3, 1, 1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(160, 64, 3, 2, 1, output_padding=1),                    # 96
                Conv2d(64, 64, 3, 1, 1, residual=True),
                Conv2d(64, 64, 3, 1, 1, residual=True),
            ),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, 3, 1, 1),
            nn.Conv2d(32, 3, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, audio_sequences: torch.Tensor, face_sequences: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio_encoder(audio_sequences)

        feats: list[torch.Tensor] = []
        x = face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)

        x = audio_embedding
        for block in self.face_decoder_blocks:
            x = block(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:  # shape mismatch surfaces here in practice
                raise RuntimeError(
                    f"Wav2Lip decoder shape mismatch: x={tuple(x.shape)}, "
                    f"skip={tuple(feats[-1].shape)}"
                ) from e
            feats.pop()

        return self.output_block(x)
