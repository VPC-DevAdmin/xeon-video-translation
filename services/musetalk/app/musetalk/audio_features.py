"""MuseTalk V1.5 audio feature extraction.

Adapted from https://github.com/TMElyralab/MuseTalk (MIT code). Semantically
identical to upstream's `musetalk/utils/audio_processor.py`; only the path
handling is softened (paths come from constructor args).

Contract:
    get_audio_feature(wav_path) -> (list[Tensor], int)
        list of 30s-window mel feature tensors + total audio sample count.

    get_whisper_chunk(features, device, dtype, whisper, librosa_length, fps,
                       audio_padding_length_left, audio_padding_length_right)
         -> Tensor of shape (num_video_frames, 50, 384)

The caller is responsible for loading the Whisper model (HF transformers
`WhisperModel`) and passing it into `get_whisper_chunk`. Done this way
because upstream does the same — the weights are big and loaded once.
"""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path: str | Path):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(feature_extractor_path))

    def get_audio_feature(
        self,
        wav_path: str | Path,
        weight_dtype: torch.dtype | None = None,
    ):
        wav_path = str(wav_path)
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        assert sr == 16000

        segment_length = 30 * sr
        segments = [audio[i : i + segment_length] for i in range(0, len(audio), segment_length)]

        features = []
        for seg in segments:
            feat = self.feature_extractor(seg, return_tensors="pt", sampling_rate=sr).input_features
            if weight_dtype is not None:
                feat = feat.to(dtype=weight_dtype)
            features.append(feat)
        return features, len(audio)

    def get_whisper_chunk(
        self,
        whisper_input_features: list[torch.Tensor],
        device: torch.device,
        weight_dtype: torch.dtype,
        whisper,
        librosa_length: int,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ) -> torch.Tensor:
        audio_feature_length_per_frame = 2 * (
            audio_padding_length_left + audio_padding_length_right + 1
        )

        per_segment = []
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            hidden = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            stacked = torch.stack(hidden, dim=2)  # (1, T, L+1, D)
            per_segment.append(stacked)
        whisper_feature = torch.cat(per_segment, dim=1)

        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, ...]

        padding_nums = math.ceil(whisper_idx_multiplier)
        whisper_feature = torch.cat(
            [
                torch.zeros_like(whisper_feature[:, : padding_nums * audio_padding_length_left]),
                whisper_feature,
                # Generous right-pad so the window never runs off the end.
                torch.zeros_like(whisper_feature[:, : padding_nums * 3 * audio_padding_length_right]),
            ],
            dim=1,
        )

        audio_prompts = []
        for frame_index in range(num_frames):
            audio_index = math.floor(frame_index * whisper_idx_multiplier)
            audio_clip = whisper_feature[:, audio_index : audio_index + audio_feature_length_per_frame]
            if audio_clip.shape[1] != audio_feature_length_per_frame:
                # Can happen on the very last frame — pad with zeros.
                pad_n = audio_feature_length_per_frame - audio_clip.shape[1]
                audio_clip = torch.cat(
                    [audio_clip, torch.zeros_like(audio_clip[:, :pad_n])], dim=1
                )
            audio_prompts.append(audio_clip)

        audio_prompts = torch.cat(audio_prompts, dim=0)  # (T, 10, L+1, D)
        audio_prompts = rearrange(audio_prompts, "b c h w -> b (c h) w")  # (T, 50, D)
        return audio_prompts
