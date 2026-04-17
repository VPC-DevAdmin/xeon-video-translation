"""Wav2Lip inference loop for polyglot-demo.

Adapted from the original Wav2Lip ``inference.py`` (Rudrabha Mukhopadhyay et
al., IIIT Hyderabad, 2020). Kept intentionally minimal; this is a demo path
for CPU, not a production reimplementation.

Pipeline:
  1. Load video frames via OpenCV, detect a face per frame (Haar frontal).
  2. Load translated audio, compute Wav2Lip's mel spectrogram.
  3. For each frame, pair the face crop with a 16-mel-frame audio window and
     run the model. Paste the predicted mouth region back into the frame.
  4. Write frames + audio to an intermediate MP4.

Known, deliberate limitations (documented in docs/lipsync.md):
- Haar frontal detector — profile shots fail, fast head motion drops frames.
- No smoothing/tracking across frames — detector flicker can cause judder.
- Single-face-per-frame assumption.
- Output face resolution is 96×96 upsampled back; mouth region looks softer
  than the rest of the face.

Progress reporting: caller passes a ``progress(float_in_0_1)`` callback that
we invoke after each frame batch. The orchestrator forwards these to SSE.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np

from ...config import settings
from . import audio as w2l_audio

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

MEL_STEP_SIZE = 16          # audio frames per video frame window
FACE_SIZE = 96              # Wav2Lip operates on 96x96 face crops
BATCH_SIZE = 16             # frames per forward pass
PADDING = (0, 10, 0, 0)     # top, bottom, left, right — Wav2Lip's default


_model = None
_model_lock = Lock()


def _checkpoint_path() -> Path:
    return settings.model_cache_dir / "wav2lip" / "wav2lip_gan.pth"


def _download_checkpoint(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = settings.wav2lip_checkpoint_url
    log.info("Downloading Wav2Lip checkpoint from %s", url)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        import torch

        from .wav2lip_model import Wav2Lip

        ckpt = _checkpoint_path()
        if not ckpt.exists():
            _download_checkpoint(ckpt)

        model = Wav2Lip()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        # Wav2Lip checkpoints wrap params under "state_dict" with "module."
        # prefixes from DataParallel training.
        sd = state.get("state_dict", state)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        _model = model
        return _model


# --------------------------------------------------------------------------- #
# Face detection — OpenCV Haar frontal cascade. Fast on CPU, front-facing only.
# --------------------------------------------------------------------------- #


def _haar_cascade():
    import cv2
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"failed to load Haar cascade at {path}")
    return clf


def _detect_face(frame, cascade) -> tuple[int, int, int, int] | None:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # Largest detected face wins.
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _pad_box(box, frame_shape) -> tuple[int, int, int, int]:
    import cv2  # noqa: F401
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    pt, pb, pl, pr = PADDING
    x1 = max(0, x - pl)
    y1 = max(0, y - pt)
    x2 = min(fw, x + w + pr)
    y2 = min(fh, y + h + pb)
    return x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# Main inference function
# --------------------------------------------------------------------------- #


def run(
    video_in: Path,
    audio_in: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
):
    import cv2
    import torch

    from ..lipsync import LipsyncError, LipsyncResult

    if not video_in.exists():
        raise LipsyncError(f"video input missing: {video_in}")
    if not audio_in.exists():
        raise LipsyncError(f"audio input missing: {audio_in}")

    model = _load_model()
    cascade = _haar_cascade()

    # --- 1. Load frames + detect faces (single pass, keeps memory bounded) ---
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    face_boxes: list[tuple[int, int, int, int] | None] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        face_boxes.append(_detect_face(frame, cascade))
    cap.release()

    if not frames:
        raise LipsyncError("no frames in input video")

    detected = sum(1 for b in face_boxes if b is not None)
    if detected == 0:
        raise LipsyncError(
            "no face detected in any frame. Wav2Lip requires a front-facing "
            "speaker. Consider LIPSYNC_BACKEND=none."
        )
    if detected < len(frames):
        log.warning(
            "Face missing in %d/%d frames; those will be left untouched.",
            len(frames) - detected, len(frames),
        )

    # --- 2. Audio → mel ------------------------------------------------------
    wav = w2l_audio.load_wav(str(audio_in))
    mel = w2l_audio.melspectrogram(wav)  # (80, T)
    if np.isnan(mel).any():
        raise LipsyncError("mel spectrogram contains NaN (is the audio silent?)")

    # Wav2Lip uses a fixed mapping: 16 mel frames per video frame at 25fps.
    mel_idx_multiplier = 80.0 / fps
    mel_chunks: list[np.ndarray] = []
    for i in range(len(frames)):
        start = int(i * mel_idx_multiplier)
        if start + MEL_STEP_SIZE > mel.shape[1]:
            # Pad with the last window so we have one chunk per frame.
            start = max(0, mel.shape[1] - MEL_STEP_SIZE)
        mel_chunks.append(mel[:, start : start + MEL_STEP_SIZE])

    # Trim frames/chunks to the same length; they usually already match.
    n = min(len(frames), len(mel_chunks))
    frames = frames[:n]
    face_boxes = face_boxes[:n]
    mel_chunks = mel_chunks[:n]

    # --- 3. Inference batches ------------------------------------------------
    # Write to a temporary AVI first (loss-free), then mux with ffmpeg to MP4.
    tmp_video = Path(tempfile.mkstemp(suffix=".avi")[1])
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(tmp_video),
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    try:
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                batch_frames = frames[start : start + BATCH_SIZE]
                batch_boxes = face_boxes[start : start + BATCH_SIZE]
                batch_mels = mel_chunks[start : start + BATCH_SIZE]

                # Build model inputs only for frames with a detected face.
                face_crops: list[np.ndarray] = []
                face_coords: list[tuple[int, int, int, int]] = []
                active_mels: list[np.ndarray] = []
                active_idx: list[int] = []
                for i, (frame, box, m) in enumerate(zip(batch_frames, batch_boxes, batch_mels)):
                    if box is None:
                        continue
                    x1, y1, x2, y2 = _pad_box(box, frame.shape)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
                    face_crops.append(crop_resized)
                    face_coords.append((x1, y1, x2, y2))
                    active_mels.append(m)
                    active_idx.append(i)

                if face_crops:
                    face_arr = np.asarray(face_crops).astype(np.float32) / 255.0
                    # Build the masked-lower + reference 6-channel input.
                    masked = face_arr.copy()
                    masked[:, FACE_SIZE // 2 :] = 0  # bottom half zeroed
                    model_in = np.concatenate([masked, face_arr], axis=3)
                    model_in = torch.from_numpy(model_in.transpose(0, 3, 1, 2))
                    mel_arr = np.asarray(active_mels).astype(np.float32)
                    mel_arr = mel_arr[:, np.newaxis, :, :]  # (B, 1, 80, 16)
                    mel_tensor = torch.from_numpy(mel_arr)

                    pred = model(mel_tensor, model_in)  # (B, 3, 96, 96)
                    pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(
                        np.uint8
                    )

                    for local_i, idx in enumerate(active_idx):
                        x1, y1, x2, y2 = face_coords[local_i]
                        pasted = cv2.resize(pred_np[local_i], (x2 - x1, y2 - y1))
                        batch_frames[idx][y1:y2, x1:x2] = pasted

                for frame in batch_frames:
                    writer.write(frame)

                if progress is not None:
                    progress(min(1.0, (start + len(batch_frames)) / n))
    finally:
        writer.release()

    # --- 4. Mux the audio back on. Stage 6 will re-mux with watermark, but
    # doing it here too keeps the intermediate self-contained. --------------
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(audio_in),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=900)
    os.unlink(tmp_video)
    if proc.returncode != 0:
        raise LipsyncError(
            f"ffmpeg mux after Wav2Lip failed: "
            f"{proc.stderr.decode(errors='replace')[-1000:]}"
        )

    if progress is not None:
        progress(1.0)

    return LipsyncResult(
        backend="wav2lip",
        output_path=output_path.name,
        passthrough=False,
    )
