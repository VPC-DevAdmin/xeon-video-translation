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


def _read_image_bgr(image_path: Path):
    """Decode a portrait with Pillow (EXIF-rotated) and return BGR uint8.

    cv2.imread is avoided for file decoding: the OpenCV wheel's bundled
    libpng breaks once torch/onnxruntime have loaded their own zlib into the
    process, which made PNG portraits fail intermittently."""
    from PIL import Image, ImageOps
    import numpy as np

    try:
        with Image.open(image_path) as handle:
            rgb = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
    except Exception:
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])


@torch.inference_mode()
def render(image_path: Path, audio_path: Path, output_path: Path):
    state = get_or_load(
        WeightPaths.from_cache(Path(os.getenv("MODEL_CACHE_DIR", "/models")))
    )
    key = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if key not in _prepared:
        image = _read_image_bgr(image_path)
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
        blend = None
        if state.device.type == "cuda" and os.getenv("AVATAR_GPU_COMPOSITE", "1") == "1":
            from .musetalk.tensor_blending import prepare
            # Rasterize the disclosure once per portrait, then apply on CUDA.
            badge = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.putText(badge, "AI avatar", (8, image.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
            blend = prepare(
                torch.from_numpy(image).permute(2, 0, 1).to(state.device), box,
                torch.from_numpy(mask).to(state.device),
                disclosure=torch.from_numpy(badge).to(state.device)[None,None].float()/255,
            )
        _prepared[key] = (image, box, latent, mask, blend)
        if len(_prepared) > 8:
            _prepared.popitem(last=False)
    _prepared.move_to_end(key)
    image, box, latent, mask, blend = _prepared[key]
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
            if blend is not None:
                from .musetalk.tensor_blending import composite
                from gpu_runtime import span
                with span("avatar.composite", device=state.device, frames=len(batch)):
                    frames = composite(blend, state.vae.decode_latents_tensor(prediction))
                    output.extend(frames.cpu().numpy())
                continue
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
