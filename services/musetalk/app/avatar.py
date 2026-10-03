"""Prepared still-image MuseTalk avatars. GPU validation required on target host."""

from __future__ import annotations
import hashlib
import os
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
from .musetalk.inference import get_or_load, WeightPaths
from .musetalk.face_tracking import detect_batch
from .musetalk.blending import composite_np, face_large_crop_rgb

_prepared = OrderedDict()


@torch.inference_mode()
def render(image_path: Path, audio_path: Path, output_path: Path):
    state = get_or_load(
        WeightPaths.from_cache(Path(os.getenv("MODEL_CACHE_DIR", "/models")))
    )
    key = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if key not in _prepared:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("invalid avatar image")
        h, w = image.shape[:2]
        scale = min(1, 512 / max(h, w))
        image = cv2.resize(
            image, (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        )
        detection = detect_batch(state.aligner, [image])[0]
        if detection.face_box is None:
            raise RuntimeError("no face in avatar image")
        x1, y1, x2, y2 = detection.face_box
        y2 = min(image.shape[0], y2 + 10)
        box = (x1, y1, x2, y2)
        crop = cv2.resize(image[y1:y2, x1:x2], (256, 256))
        latent = state.vae.get_latents_for_unet_batch([crop])
        mask = state.face_parsing.parse_batch_np(
            [face_large_crop_rgb(image, box)], mode="mouth"
        )[0]
        _prepared[key] = (image, box, latent, mask)
        if len(_prepared) > 8:
            _prepared.popitem(last=False)
    _prepared.move_to_end(key)
    image, box, latent, mask = _prepared[key]
    features, samples = state.audio_processor.get_audio_feature(
        audio_path, weight_dtype=state.weight_dtype
    )
    audio = state.audio_processor.get_whisper_chunk(
        features, state.device, state.weight_dtype, state.whisper, samples, fps=25
    )
    count = len(audio)
    if not count or count > 250:
        raise RuntimeError("avatar chunks must be between 40 ms and 10 seconds")
    output = []
    x1, y1, x2, y2 = box
    for start in range(0, count, 8):
        batch = audio[start : start + 8].to(state.device, dtype=state.weight_dtype)
        condition = latent.expand(len(batch), -1, -1, -1).to(dtype=state.weight_dtype)
        with torch.autocast(
            device_type=state.device.type,
            dtype=state.weight_dtype,
            enabled=state.weight_dtype in (torch.float16, torch.bfloat16),
        ):
            prediction = state.unet.model(
                condition,
                torch.tensor([0], device=state.device),
                encoder_hidden_states=state.unet.pe(batch),
            ).sample
            faces = state.vae.decode_latents(prediction)
        for face in faces:
            composed = composite_np(
                image, cv2.resize(face, (x2 - x1, y2 - y1)), box, mask
            )
            cv2.putText(
                composed,
                "AI avatar",
                (8, image.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            output.append(composed)
    # Local shared-volume exchange avoids JPEG/base64 and additional lossy encoding.
    np.save(output_path, np.stack(output))
    return {"frames_path": str(output_path), "fps": 25}
