"""MuseTalk V1.5 inference driver.

Adapted from https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py
(MIT code). Simplified for a CPU-only single-clip path:

- No GPU fallback branches.
- One-shot inference per request (no batch of multiple videos).
- Face detection via InsightFace SCRFD (replaces face-alignment SFD+FAN).
- Raw detections are cached per-input in /models/cache/face_detections/ so
  iterating on downstream params doesn't pay the detection cost twice.
- After detection we gap-fill (carry nearest bbox) and temporally smooth
  (moving average) to stabilize the blend seam.

Public entry point: `run(video_path, audio_path, output_path, **kwargs)`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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


# --------------------------------------------------------------------------- #
# Env-driven blend knobs. Resolved once at import so logs and requests are
# consistent within a service lifetime.
# --------------------------------------------------------------------------- #


def _read_blend_mode() -> str:
    # Default flipped to `jaw` after user feedback: `mouth` preserved stubble
    # but produced a visible "sticker" effect because the lips moved while
    # the chin stayed frozen — real jaw and lips are muscularly coupled.
    # `jaw` regenerates chin+cheeks+mouth together so the motion reads as
    # anatomically correct. Stubble loss in the chin region is compensated
    # for downstream by CodeFormer face restoration (future PR).
    mode = os.environ.get("MUSETALK_BLEND_MODE", "jaw").strip().lower()
    if mode not in ("raw", "jaw", "mouth", "neck"):
        log.warning("unknown MUSETALK_BLEND_MODE=%r; falling back to jaw", mode)
        mode = "jaw"
    return mode


def _read_blend_feather() -> float:
    # 0.04 is narrow enough that mouth-mode's small mask keeps an opaque
    # center (no "ghost mouth" overlay), and wide enough that jaw-mode
    # seams still fade smoothly. blending.py caps the absolute kernel at
    # 31 px, so setting this very high above ~0.08 has no extra effect on
    # large crops — intentional.
    raw = os.environ.get("MUSETALK_BLEND_FEATHER", "0.04")
    try:
        value = float(raw)
    except ValueError:
        log.warning("bad MUSETALK_BLEND_FEATHER=%r; falling back to 0.04", raw)
        return 0.04
    # Guard against absurd values that would produce invalid kernel sizes.
    if not 0.02 <= value <= 0.30:
        log.warning(
            "MUSETALK_BLEND_FEATHER=%r out of bounds [0.02, 0.30]; clamping", raw
        )
        value = max(0.02, min(0.30, value))
    return value


_blend_mode = _read_blend_mode()
_blend_feather = _read_blend_feather()


def _read_face_restore() -> str:
    """`MUSETALK_FACE_RESTORE`: codeformer | none.

    `codeformer` runs the CodeFormer face-restoration model per frame after
    the MuseTalk blend to recover high-frequency skin detail that the VAE
    round-trip loses. `none` skips restoration entirely — use this for
    speed or when debugging upstream stages.
    """
    mode = os.environ.get("MUSETALK_FACE_RESTORE", "codeformer").strip().lower()
    if mode not in ("codeformer", "none"):
        log.warning("unknown MUSETALK_FACE_RESTORE=%r; falling back to codeformer", mode)
        mode = "codeformer"
    return mode


_face_restore_mode = _read_face_restore()


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


def _fill_boxes(
    boxes: list[tuple[int, int, int, int] | None],
) -> list[tuple[int, int, int, int] | None]:
    """Fill `None` bboxes with the nearest valid one in time.

    Forward-fills first (gaps use the last valid detection we saw), then
    backward-fills leading gaps (frames before the first detection get the
    earliest detection). A clip with zero detections stays all-None — the
    caller is expected to have raised before getting here.
    """
    out: list[tuple[int, int, int, int] | None] = list(boxes)

    last: tuple[int, int, int, int] | None = None
    for i, b in enumerate(out):
        if b is not None:
            last = b
        elif last is not None:
            out[i] = last

    first: tuple[int, int, int, int] | None = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is not None:
            first = out[i]
        elif first is not None:
            out[i] = first

    return out


def _smooth_boxes(
    boxes: list[tuple[int, int, int, int] | None],
    window: int = 5,
) -> list[tuple[int, int, int, int] | None]:
    """Moving-average smoother on per-coordinate bbox sequences.

    SCRFD gives accurate-but-not-stable bboxes: the same static face often
    jitters ±5 px frame-to-frame. Downstream, that jitter moves the blend
    seam and reads as flicker between "regenerated" and "original" texture.
    Averaging across a window of `window` frames damps the jitter without
    lagging real head motion noticeably at this window size (±2 frames at
    30 fps = ±67 ms).

    None entries (which _fill_boxes should already have handled) are passed
    through unchanged.
    """
    n = len(boxes)
    out: list[tuple[int, int, int, int] | None] = list(boxes)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        nearby = [b for b in boxes[lo:hi] if b is not None]
        if not nearby:
            continue
        xs1 = sum(b[0] for b in nearby) / len(nearby)
        ys1 = sum(b[1] for b in nearby) / len(nearby)
        xs2 = sum(b[2] for b in nearby) / len(nearby)
        ys2 = sum(b[3] for b in nearby) / len(nearby)
        out[i] = (int(round(xs1)), int(round(ys1)), int(round(xs2)), int(round(ys2)))
    return out


# --------------------------------------------------------------------------- #
# Detection cache
#
# Keyed on a cheap signature of the input video (size + first 1 MB).
# Cached under MODEL_CACHE_DIR/cache/face_detections/.
# The cache stores raw SCRFD output (pre-fill, pre-smooth) so downstream
# preprocessing can change without invalidating detection work.
# --------------------------------------------------------------------------- #

_DETECTOR_VERSION = "scrfd/buffalo_l/v1"
# v2 added 5-point kps per frame for the CodeFormer alignment path. Old v1
# caches are ignored and re-detected — the invalidation is cheap because
# SCRFD is seconds, not minutes.
_CACHE_SCHEMA_VERSION = 2


def _cache_dir() -> Path:
    root = Path(os.environ.get("MODEL_CACHE_DIR", "/models")) / "cache" / "face_detections"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _video_signature(path: Path) -> str:
    """Fast-enough fingerprint. First 1 MB hash + size + detector version.

    Collisions in practice are negligible for demo use; we're not verifying
    video identity for security, just avoiding redundant compute across
    repeated runs on the same asset.
    """
    stat = path.stat()
    h = hashlib.sha1()
    h.update(_DETECTOR_VERSION.encode())
    h.update(str(stat.st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()


def _cache_path_for(video_path: Path) -> Path:
    return _cache_dir() / f"{_video_signature(video_path)}.json"


# Cache entry type: (bbox, score, kps). `kps` is stored as a python list of
# 5 [x, y] pairs when present, None otherwise. Numpy is reconstructed on
# load.
_CacheEntry = tuple[
    tuple[int, int, int, int] | None,
    float | None,
    "np.ndarray | None",
]


def _load_detection_cache(video_path: Path) -> list[_CacheEntry] | None:
    path = _cache_path_for(video_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        log.warning("face-detection cache unreadable (%s); will re-detect", e)
        return None
    if data.get("schema") != _CACHE_SCHEMA_VERSION:
        return None
    frames = data.get("frames", [])
    out: list[_CacheEntry] = []
    for entry in frames:
        bbox = entry.get("bbox")
        score = entry.get("score")
        kps_list = entry.get("kps")
        kps = None
        if kps_list is not None:
            try:
                import numpy as _np  # local to keep json path cheap
                kps = _np.asarray(kps_list, dtype=_np.float32).reshape(-1, 2)
            except Exception:
                kps = None
        bbox_tuple = None if bbox is None else tuple(int(v) for v in bbox)
        out.append((bbox_tuple, score, kps))  # type: ignore[arg-type]
    return out


def _save_detection_cache(
    video_path: Path,
    detections: list[_CacheEntry],
) -> None:
    path = _cache_path_for(video_path)
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "detector": _DETECTOR_VERSION,
        "video": str(video_path),
        "frames": [
            {
                "index": i,
                "bbox": list(bbox) if bbox else None,
                "score": score,
                "kps": kps.tolist() if kps is not None else None,
            }
            for i, (bbox, score, kps) in enumerate(detections)
        ],
    }
    try:
        path.write_text(json.dumps(payload))
    except Exception as e:
        log.warning("failed to write face-detection cache: %s", e)


# --------------------------------------------------------------------------- #
# Intel Extension for PyTorch — kernel-level acceleration on Xeon.
# --------------------------------------------------------------------------- #


def _ipex_dtype() -> "torch.dtype":
    """Resolve the IPEX compute dtype from env.

    fp32 is the safe default — pure kernel acceleration, no numerical drift.
    bf16 is opt-in because the VAE and UNet haven't been validated end-to-end
    at lower precision and may produce subtle output changes (mouth texture,
    color shift). Enable with `MUSETALK_IPEX_DTYPE=bf16`.
    """
    choice = os.environ.get("MUSETALK_IPEX_DTYPE", "fp32").lower()
    if choice in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _ipex_optimize(model: "torch.nn.Module", name: str) -> "torch.nn.Module":
    """Wrap `model` with ipex.optimize() if IPEX is installed.

    Silently falls through on any failure — we never want perf plumbing
    to hard-fail a request. Caught broadly because IPEX import can raise
    native-loader errors (e.g. executable-stack markers) that aren't
    strictly ImportError.
    """
    try:
        import intel_extension_for_pytorch as ipex
    except Exception as e:
        log.warning("IPEX import failed (%s); skipping %s optimization", e, name)
        return model

    dtype = _ipex_dtype()
    try:
        optimized = ipex.optimize(model.eval(), dtype=dtype, inplace=False)
    except Exception as e:
        log.warning("IPEX optimize(%s) failed (%s); using vanilla PyTorch", name, e)
        return model

    log.info("IPEX optimized %s (dtype=%s)", name, str(dtype).rsplit(".", 1)[-1])
    return optimized


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
    # `weight_dtype` drives tensor casting in the AudioProcessor + UNet path.
    # IPEX's optimize() can still run fp32 kernels underneath while our own
    # tensors stay in this dtype — they're independent knobs.
    dtype = _ipex_dtype()

    log.info("Loading Whisper encoder from %s", paths.whisper_dir)
    whisper = WhisperModel.from_pretrained(str(paths.whisper_dir)).to(device)
    whisper.eval()
    whisper = _ipex_optimize(whisper, name="whisper")

    audio_processor = AudioProcessor(paths.whisper_dir)

    log.info("Loading VAE from %s", paths.vae_dir)
    vae = VAE(paths.vae_dir, device=device)
    vae.vae = _ipex_optimize(vae.vae, name="sd-vae")

    log.info("Loading UNet from %s", paths.unet_weights)
    unet = UNet(str(paths.unet_config), str(paths.unet_weights), device=device)
    unet.model = _ipex_optimize(unet.model, name="musetalk-unet")

    log.info("Loading BiSeNet face parser")
    face_parsing = FaceParsing(
        model_path=paths.bisenet_weights,
        resnet_path=paths.resnet_weights,
        device=device,
    )
    # BiSeNet is tiny and called per-frame at blend time — the per-call
    # overhead of optimize() isn't worth paying for the modest speedup.
    # Leaving it vanilla.

    log.info("Loading SCRFD face detector")
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

    # --- 3. Face detection + stabilization --------------------------------
    cached = _load_detection_cache(video_path)
    if cached is not None and len(cached) == len(frames):
        log.info("Face detections loaded from cache (%d frames)", len(frames))
        raw_boxes = [bbox for bbox, _, _ in cached]
        raw_kps = [kps for _, _, kps in cached]
    else:
        log.info("Detecting faces in %d frames (SCRFD)", len(frames))
        detections = detect_batch(state.aligner, frames)
        raw_boxes = [d.face_box for d in detections]
        raw_kps = [d.kps for d in detections]
        try:
            _save_detection_cache(
                video_path,
                [(d.face_box, d.score, d.kps) for d in detections],
            )
        except Exception as e:
            log.warning("could not persist detection cache: %s", e)

    missing_count = sum(1 for b in raw_boxes if b is None)
    if missing_count == len(frames):
        raise RuntimeError(
            "no face detected in any frame. Is the subject front-facing?"
        )
    if missing_count:
        # Fill the gaps with the nearest valid bbox (forward first, then
        # backward for leading gaps) so every frame gets inference. Without
        # this, frames where detection dropped render as passthrough and
        # flip visibly against MuseTalk-regenerated neighbors.
        log.info(
            "Face detection gaps: %d/%d frames — carrying nearest bbox forward",
            missing_count, len(frames),
        )
    face_boxes_filled = _fill_boxes(raw_boxes)
    # 5-frame moving average on bbox coordinates — kills SCRFD's natural
    # ±few-pixel jitter so the blend seam doesn't wobble between frames.
    face_boxes_filled = _smooth_boxes(face_boxes_filled, window=5)

    # --- 4. Per-frame VAE latents -----------------------------------------
    log.info("Encoding face crops via VAE")
    input_latents: list[torch.Tensor | None] = []
    face_boxes: list[tuple[int, int, int, int] | None] = []
    for frame, box in zip(frames, face_boxes_filled):
        if box is None:
            input_latents.append(None)
            face_boxes.append(None)
            continue
        x1, y1, x2, y2 = box
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

    # When running under bf16, wrap the forward in CPU autocast so the mixed
    # math happens safely (BatchNorm/LayerNorm stays fp32 via autocast's
    # allowlist). fp32 path is unchanged — autocast becomes a no-op below.
    autocast_enabled = state.weight_dtype == torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast_enabled)
    )

    with torch.no_grad(), autocast_ctx:
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
            # Blend mode / feather are env-tunable.
            #   MUSETALK_BLEND_MODE  = raw | jaw | mouth | neck  (default: jaw)
            #   MUSETALK_BLEND_FEATHER = kernel ratio           (default: 0.04)
            mode=_blend_mode,
            feather_ratio=_blend_feather,
        )
        output_frames.append(blended)

    # --- 6b. Optional face restoration (CodeFormer) -----------------------
    # Applied after the MuseTalk blend so the restored skin detail covers
    # both the original area (unchanged by MuseTalk but degraded by the
    # underlying VAE round-trip on nearby frames) and the regenerated
    # lower face. Per-frame cost is ~3-8 s on the Xeon; total adds 2-7 min
    # on a ~50-frame clip. Skipped cleanly if weights are missing.
    if _face_restore_mode == "codeformer":
        try:
            from ._codeformer import restore_frame as _cf_restore
            log.info("CodeFormer face restoration: %d frames", n)
            for i in range(n):
                kps = raw_kps[i] if i < len(raw_kps) else None
                if kps is None:
                    continue
                try:
                    output_frames[i] = _cf_restore(
                        output_frames[i], kps=kps, device=state.device,
                    )
                except Exception as e:
                    log.warning("CodeFormer skipped on frame %d: %s", i, e)
                if progress is not None:
                    progress(min(1.0, (i + 1) / n))
        except Exception as e:
            log.warning(
                "CodeFormer unavailable (%s); shipping un-restored frames", e,
            )

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
