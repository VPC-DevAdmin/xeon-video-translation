"""MuseTalk V1.5 inference driver.

Adapted from https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py
(MIT code). Simplified for a CPU-only single-clip path:

- No GPU fallback branches.
- One-shot inference per request (no batch of multiple videos).
- Face detection via `face-alignment` (replaces mmpose/DWPose).
- No "cycle/loop" extension when audio is longer than video — we just pad
  with the last detected frame, which is the behavior the upstream has for
  the audio > video case.

Public entry point: `run(video_path, audio_path, output_path, **kwargs)`.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

import cv2
import numpy as np
import torch

from .audio_features import AudioProcessor
from .blending import get_image
from .face_parsing import FaceParsing
from .face_tracking import build_aligner, detect_batch
from .models.unet import UNet
from .models.vae import VAE

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            timeout=30,
        )
        return float(out.decode().strip())
    except Exception as e:
        log.warning("ffprobe failed on %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- #
# Paths — relative to the service's shared /models volume.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WeightPaths:
    unet_weights: Path
    unet_config: Path
    vae_dir: Path
    whisper_dir: Path
    bisenet_weights: Path
    resnet_weights: Path

    @classmethod
    def from_cache(cls, cache_root: Path) -> "WeightPaths":
        root = cache_root / "musetalk"
        return cls(
            unet_weights=root / "musetalkV15" / "unet.pth",
            unet_config=root / "musetalkV15" / "musetalk.json",
            vae_dir=root / "sd-vae",
            whisper_dir=root / "whisper",
            bisenet_weights=root / "face-parse-bisent" / "79999_iter.pth",
            resnet_weights=root / "face-parse-bisent" / "resnet18-5c106cde.pth",
        )

    def missing(self) -> list[Path]:
        return [p for p in (self.unet_weights, self.unet_config, self.vae_dir,
                            self.whisper_dir, self.bisenet_weights,
                            self.resnet_weights) if not p.exists()]


# --------------------------------------------------------------------------- #
# Lazy singletons. MuseTalk models are heavy enough that reloading per request
# isn't defensible; keep them in memory until the worker restarts.
# --------------------------------------------------------------------------- #


@dataclass
class _Loaded:
    audio_processor: AudioProcessor
    whisper: "torch.nn.Module"
    vae: VAE
    unet: UNet
    face_parsing: FaceParsing
    aligner: object
    device: torch.device
    weight_dtype: torch.dtype


_loaded: _Loaded | None = None
_load_lock = Lock()


def _load(paths: WeightPaths) -> _Loaded:
    from transformers import WhisperModel

    device = torch.device("cpu")
    dtype = torch.float32  # int8 quantization is a later exercise

    log.info("Loading Whisper encoder from %s", paths.whisper_dir)
    whisper = WhisperModel.from_pretrained(str(paths.whisper_dir)).to(device)
    whisper.eval()

    audio_processor = AudioProcessor(paths.whisper_dir)

    log.info("Loading VAE from %s", paths.vae_dir)
    vae = VAE(paths.vae_dir, device=device)

    log.info("Loading UNet from %s", paths.unet_weights)
    unet = UNet(str(paths.unet_config), str(paths.unet_weights), device=device)

    log.info("Loading BiSeNet face parser")
    face_parsing = FaceParsing(
        model_path=paths.bisenet_weights,
        resnet_path=paths.resnet_weights,
        device=device,
    )

    log.info("Loading face-alignment landmark detector")
    aligner = build_aligner(device="cpu")

    return _Loaded(
        audio_processor=audio_processor,
        whisper=whisper,
        vae=vae,
        unet=unet,
        face_parsing=face_parsing,
        aligner=aligner,
        device=device,
        weight_dtype=dtype,
    )


def get_or_load(paths: WeightPaths) -> _Loaded:
    global _loaded
    if _loaded is not None:
        return _loaded
    with _load_lock:
        if _loaded is not None:
            return _loaded
        _loaded = _load(paths)
        return _loaded


# --------------------------------------------------------------------------- #
# Main inference path
# --------------------------------------------------------------------------- #


@dataclass
class InferenceResult:
    output_path: str
    frames_processed: int


def run(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    weight_paths: WeightPaths,
    progress: ProgressCallback | None = None,
    extra_margin: int = 10,
    batch_size: int = 4,
) -> InferenceResult:
    """Run MuseTalk V1.5 inference end-to-end.

    Parameters mirror the upstream script:
        extra_margin: pixels added to the bottom of each face crop (V1.5 default 10)
        batch_size: frames per UNet forward pass. CPU memory bound.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    missing = weight_paths.missing()
    if missing:
        raise FileNotFoundError(
            "MuseTalk weights missing: " + ", ".join(str(p) for p in missing)
        )

    state = get_or_load(weight_paths)

    # --- 1. Audio features -------------------------------------------------
    log.info("Extracting audio features")
    whisper_input_features, librosa_length = state.audio_processor.get_audio_feature(
        audio_path, weight_dtype=state.weight_dtype
    )

    # --- 2. Read frames ----------------------------------------------------
    log.info("Reading video frames")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("no frames in input video")

    # Whisper chunks — one 50×384 tensor per video frame, using whisper model.
    whisper_chunks = state.audio_processor.get_whisper_chunk(
        whisper_input_features,
        device=state.device,
        weight_dtype=state.weight_dtype,
        whisper=state.whisper,
        librosa_length=librosa_length,
        fps=int(round(fps)),
    )  # (T_audio_frames, 50, 384)

    # --- 3. Face detection + landmarks ------------------------------------
    log.info("Detecting faces in %d frames", len(frames))
    detections = detect_batch(state.aligner, frames)
    missing_count = sum(1 for d in detections if d.face_box is None)
    if missing_count == len(frames):
        raise RuntimeError(
            "no face detected in any frame. Is the subject front-facing?"
        )
    if missing_count:
        log.warning("Face missing in %d/%d frames — will leave those untouched",
                    missing_count, len(frames))

    # --- 4. Per-frame VAE latents -----------------------------------------
    log.info("Encoding face crops via VAE")
    input_latents: list[torch.Tensor | None] = []
    face_boxes: list[tuple[int, int, int, int] | None] = []
    for frame, det in zip(frames, detections):
        if det.face_box is None:
            input_latents.append(None)
            face_boxes.append(None)
            continue
        x1, y1, x2, y2 = det.face_box
        # V1.5 adds a bottom margin so the chin is fully included.
        y2 = min(y2 + extra_margin, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            input_latents.append(None)
            face_boxes.append(None)
            continue
        crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = state.vae.get_latents_for_unet(crop_256)
        input_latents.append(latents)
        face_boxes.append((x1, y1, x2, y2))

    # --- 5. Pair audio ↔ frames and run UNet -----------------------------
    n_video = len(frames)
    n_audio = whisper_chunks.shape[0]
    n = min(n_video, n_audio)
    log.info("Running UNet on %d frames (video=%d, audio_chunks=%d)", n, n_video, n_audio)

    predicted_faces: list[np.ndarray | None] = [None] * n
    timesteps = torch.tensor([0], device=state.device)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            # Gather only the frames that have latents; skipped frames pass through.
            active: list[int] = [i for i in range(start, end) if input_latents[i] is not None]
            if not active:
                continue

            latent_batch = torch.cat([input_latents[i] for i in active], dim=0).to(
                state.device, dtype=state.weight_dtype
            )
            audio_batch = whisper_chunks[torch.tensor(active)].to(
                state.device, dtype=state.weight_dtype
            )
            # Positional encoding on the audio tokens, per MuseTalk.
            audio_batch = state.unet.pe(audio_batch)

            pred_latents = state.unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_batch
            ).sample
            recon = state.vae.decode_latents(pred_latents)  # (B, 256, 256, 3) uint8 BGR

            for local, i in enumerate(active):
                predicted_faces[i] = recon[local]

            if progress is not None:
                progress(min(1.0, end / n))

    # --- 6. Paste predicted faces back with BiSeNet-aware blending --------
    log.info("Compositing predicted faces back into frames")
    output_frames: list[np.ndarray] = []
    for i in range(n):
        frame = frames[i].copy()
        face = predicted_faces[i]
        box = face_boxes[i]
        if face is None or box is None:
            output_frames.append(frame)
            continue
        x1, y1, x2, y2 = box
        face_resized = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        blended = get_image(
            image=frame,
            face=face_resized,
            face_box=(x1, y1, x2, y2),
            fp=state.face_parsing,
            mode="raw",
        )
        output_frames.append(blended)

    # --- 7. Write video + mux audio ---------------------------------------
    log.info("Writing output video")
    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    height, width = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    try:
        for f in output_frames:
            writer.write(f)
    finally:
        writer.release()

    # Mux new audio onto the silent MP4. If audio is longer than the
    # lipsynced video (common — XTTS output often runs past the source clip),
    # freeze the last frame rather than truncating speech with `-shortest`.
    audio_dur = _probe_duration(audio_path)
    video_dur = _probe_duration(tmp_video)
    pad_seconds = 0.0
    if audio_dur is not None and video_dur is not None and audio_dur > video_dur:
        pad_seconds = audio_dur - video_dur

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(audio_path),
    ]
    if pad_seconds > 0.0:
        cmd.extend([
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}",
        ])
    cmd.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac",
        # Deliberately no `-shortest`: we padded video above when needed.
        str(output_path),
    ])
    log.info(
        "musetalk mux: video=%.2fs audio=%.2fs pad=%.2fs",
        video_dur or -1.0, audio_dur or -1.0, pad_seconds,
    )
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    tmp_video.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed: {proc.stderr.decode(errors='replace')[-1000:]}"
        )

    if progress is not None:
        progress(1.0)

    return InferenceResult(output_path=str(output_path), frames_processed=n)
