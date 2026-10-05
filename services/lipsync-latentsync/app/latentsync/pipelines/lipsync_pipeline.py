# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import queue
import threading
import time
import json

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, write_video_with_audio, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _decompose_similarity(matrix: torch.Tensor) -> tuple:
    """Decompose a 2×3 similarity-transform matrix into (tx, ty, theta, scale).

    LatentSync's face aligner produces similarity transforms
    (rotation + translation + uniform scale, no shear), so 4 scalars
    fully characterize each matrix. Smoothing these four components
    independently is the right way to temporally stabilize a sequence
    of affine matrices — averaging the raw 2×3 entries mixes rotation
    and scale noise together, producing compound artifacts visible as
    "zoom + rotate + drift" in the output.

    Shape convention: accepts (2, 3) or (1, 2, 3); always returns
    scalars.
    """
    m = matrix.squeeze(0) if matrix.dim() == 3 else matrix
    a, _b_ignored, tx = m[0, 0], m[0, 1], m[0, 2]
    c, _d_ignored, ty = m[1, 0], m[1, 1], m[1, 2]
    # Under a pure similarity transform: d = a and b = -c. We pull
    # scale from the norm of the first column; that's robust to any
    # small numerical drift from the un-used symmetry entries.
    scale = torch.sqrt(a * a + c * c)
    theta = torch.atan2(c, a)
    return tx, ty, theta, scale


def _compose_similarity(
    tx: torch.Tensor, ty: torch.Tensor,
    theta: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """Compose (tx, ty, theta, scale) into a (1, 2, 3) similarity matrix."""
    ct = torch.cos(theta) * scale
    st = torch.sin(theta) * scale
    row0 = torch.stack([ct, -st, tx])
    row1 = torch.stack([st, ct, ty])
    return torch.stack([row0, row1]).unsqueeze(0)  # (1, 2, 3)


def _smooth_affine_sequence(
    affine_matrices: list,
    boxes: list,
    window: int,
) -> tuple[list, list]:
    """CPU patch: temporal smoothing for affine matrices + bboxes in the
    similarity-parameter space (tx, ty, theta, scale).

    The face aligner's per-frame matrices jitter with landmark noise;
    applied directly in restore_img, that jitter becomes a visible
    face-bouncing in the output: translation drift, small rotations,
    and zoom in/out all at once. Averaging the raw 2×3 matrix entries
    mixes these components — a small rotation change bleeds into scale
    and translation estimates. The fix is to decompose each matrix
    into its 4 similarity parameters (tx, ty, rotation, scale), smooth
    each component independently, then recompose.

    Per-component smoothing details:

    - **tx / ty**: plain Savitzky-Golay over the window. SG preserves
      local trends better than a flat moving average — when the subject
      actually turns their head, the smoothed output tracks the real
      motion rather than lagging behind it. For static content SG is
      indistinguishable from a moving average.

    - **rotation**: smoothed in unit-vector space (cos/sin separately,
      recombined via atan2) to respect the ±π wraparound. In practice
      face-alignment rotations are always small, so the wraparound
      never bites; doing it right is free.

    - **scale**: smoothed in log-space. Scale is multiplicative: 1.02
      and 1/1.02 ≈ 0.98 are equidistant from 1.0 geometrically, but
      linear-averaging them gives 1.0 only by coincidence at small
      amplitudes. Log-averaging is correct-by-construction.

    - **bboxes**: smoothed component-wise via SG on each of the four
      coordinates.

    Boundary handling uses scipy's `mode='interp'` — a polynomial is
    fit to the first/last window and evaluated at the boundary points.
    Cleaner than shrinking the window at edges (which leaves visible
    residual jitter on the first and last few frames) and simpler than
    a hand-rolled exponential taper.

    The UNet inputs are unchanged — only the warp-back alignment is
    stabilized. For small jitter amplitudes, the mismatch between
    "what the UNet saw" (per-frame affine) and "where we warp it back"
    (smoothed affine) is imperceptible; the jitter elimination is not.
    """
    if window <= 1 or len(affine_matrices) < 2:
        return affine_matrices, boxes

    # scipy is a pipeline dependency (pyproject.toml pins scipy>=1.11).
    # Import here rather than at module top so any failure lands as a
    # clear per-call error rather than blocking pipeline import.
    from scipy.signal import savgol_filter

    n = len(affine_matrices)

    # Decompose each matrix into its similarity parameters. Use numpy
    # throughout since savgol_filter is a numpy function and the
    # per-frame tensors are small anyway.
    txs = np.empty(n, dtype=np.float64)
    tys = np.empty(n, dtype=np.float64)
    thetas = np.empty(n, dtype=np.float64)
    scales = np.empty(n, dtype=np.float64)
    for i, m in enumerate(affine_matrices):
        tx, ty, theta, scale = _decompose_similarity(m)
        txs[i] = float(tx)
        tys[i] = float(ty)
        thetas[i] = float(theta)
        scales[i] = float(scale)

    # Rotation via unit-vector decomposition (handles ±π wraparound).
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Scale in log space — multiplicative quantity, symmetric treatment
    # of x and 1/x around 1.0.
    log_scales = np.log(np.clip(scales, 1e-6, None))

    box_stack = np.array(boxes, dtype=np.float64)  # (n, 4)

    # Effective window: must be odd, must be <= signal length, and
    # > polyorder. For short clips we shrink toward 3 (the minimum SG
    # window for polyorder=2) and fall through to unsmoothed output if
    # even that doesn't fit.
    polyorder = 2
    eff_window = min(int(window), n)
    if eff_window % 2 == 0:
        eff_window -= 1
    if eff_window <= polyorder:
        # Too few frames to do a meaningful SG — return inputs unchanged.
        return affine_matrices, boxes

    def _sg(signal: np.ndarray) -> np.ndarray:
        # mode='interp' fits a polynomial to first/last window and
        # evaluates at boundary points — handles edges smoothly.
        return savgol_filter(signal, eff_window, polyorder, mode="interp")

    txs_s = _sg(txs)
    tys_s = _sg(tys)
    cos_s = _sg(cos_t)
    sin_s = _sg(sin_t)
    log_scales_s = _sg(log_scales)

    # Recombine into smoothed matrices, matching the original dtype and
    # device of each input matrix so downstream consumers see no change.
    smoothed_mats = []
    for i in range(n):
        theta_s = float(np.arctan2(sin_s[i], cos_s[i]))
        scale_s = float(np.exp(log_scales_s[i]))

        m_dtype = affine_matrices[i].dtype
        m_device = affine_matrices[i].device
        smoothed = _compose_similarity(
            torch.tensor(txs_s[i], dtype=m_dtype),
            torch.tensor(tys_s[i], dtype=m_dtype),
            torch.tensor(theta_s, dtype=m_dtype),
            torch.tensor(scale_s, dtype=m_dtype),
        ).to(device=m_device)
        smoothed_mats.append(smoothed)

    # Smooth each of the four bbox coordinates independently.
    smoothed_box_cols = np.stack(
        [_sg(box_stack[:, c]) for c in range(4)], axis=1,
    )  # (n, 4)
    smoothed_boxes = [tuple(int(v) for v in row) for row in smoothed_box_cols]

    return smoothed_mats, smoothed_boxes


def _smooth_landmarks_sequence(
    landmarks_per_frame: list,
    window: int,
) -> list:
    """CPU patch: temporal smoothing on the 3-point landmark array used
    to derive each frame's affine-to-canonical warp.

    Operates upstream of `transformation_from_points` — the SVD-based
    affine estimator that, through `(s2/s1) * R` amplification, turns
    sub-pixel landmark noise (0.1–0.5 px, common on lossy H.264 input)
    into pixel-scale translation and rotation drift in the per-frame
    affine matrix. Our existing `_smooth_affine_sequence` runs *after*
    that amplification; smoothing landmarks *before* the SVD prevents
    the noise from ever entering the affine.

    Each of the 3 landmarks (left eye, right eye, nose) has its x and
    y coordinate smoothed independently across the clip with a
    Savitzky-Golay filter (polyorder=2). That preserves real motion
    (the subject turning their head) while damping per-frame detection
    noise.

    Edge handling uses scipy's `mode='interp'` — a polynomial is fit
    to the first/last window and evaluated at the boundary, so the
    first and last few frames aren't left as untreated.

    Returns the smoothed sequence in the same Python list-of-(3,2)-array
    form as the input so the caller can slot it in without reshaping.
    """
    if window <= 1 or len(landmarks_per_frame) < 2:
        return landmarks_per_frame

    from scipy.signal import savgol_filter

    arr = np.stack(landmarks_per_frame, axis=0)  # (N, 3, 2)
    n = arr.shape[0]

    polyorder = 2
    eff_window = min(int(window), n)
    if eff_window % 2 == 0:
        eff_window -= 1
    if eff_window <= polyorder:
        return landmarks_per_frame

    # Smooth each of the 6 coordinate tracks (3 points × 2 dims) along
    # the time axis. axis=0 applies SG per-column of the flattened (N, 6).
    flat = arr.reshape(n, -1)                     # (N, 6)
    flat_s = savgol_filter(flat, eff_window, polyorder, axis=0, mode="interp")
    smoothed = flat_s.reshape(n, 3, 2)

    return [smoothed[i] for i in range(n)]


def _fill_missing_landmarks(
    per_frame_landmarks: list,
) -> tuple[list, list[int]]:
    """Gap-fill None entries in a per-frame landmark sequence.

    Input is the raw output of `try_extract_landmarks3` called across
    every frame of the clip — a list where each entry is either a
    (3, 2) array (successful detection) or None (no face detected).

    Strategy:
      1. Forward-pass: carry the last successful detection forward
         into each None slot.
      2. Any leading Nones (frames before the first successful
         detection) are filled with the first-good detection,
         backward-extending the clip.
      3. All-None sequence → return (None, [all indices]); caller
         must bail because we have no landmarks at all.

    The returned list has no None entries (unless the clip had zero
    successful detections). Downstream smoothing and affine
    computation see a continuous trajectory — the result of carrying
    landmarks forward is usually a small visual stall at the filled
    frame rather than a full-run failure.

    Returns (filled_sequence, missing_indices). `missing_indices`
    is a list of frame indices that were None in the input, for
    logging / diagnostics.
    """
    missing_indices = [i for i, l in enumerate(per_frame_landmarks) if l is None]
    if not missing_indices:
        return per_frame_landmarks, missing_indices

    # Forward fill.
    last_good = None
    filled: list = []
    for lmk in per_frame_landmarks:
        if lmk is not None:
            last_good = lmk
        filled.append(last_good)

    # Find the first successful detection for backward-fill of any
    # leading Nones. If there are none, signal with an empty list so
    # the caller can raise a clear "no faces anywhere" error.
    first_good = next((l for l in filled if l is not None), None)
    if first_good is None:
        return [], missing_indices

    filled = [first_good if l is None else l for l in filled]
    return filled, missing_indices


def chunk_timeline_report(chunk_times: dict, num_frames: int, fps: float, t0_wall: float,
                          total_frames: int, persona_key=None) -> dict:
    """Streaming margin of a chunked render, from per-chunk timestamps.

    For playout at ``fps`` that starts ``H`` seconds after the render began,
    chunk ``i`` is due at ``H + i * num_frames / fps``. The smallest ``H`` with
    no stall is ``max_i(restored_i - i * num_frames / fps)``. Also reports the
    first chunk's latency and the aggregate restored-frame rate."""
    rows = []
    for index in sorted(chunk_times):
        row = {"chunk": index, **chunk_times[index]}
        rows.append(row)
    restored = [(r["chunk"], r["restored"]) for r in rows if "restored" in r]
    report = {"event": "latentsync_chunks", "t0_wall": round(t0_wall, 3), "num_frames": num_frames,
              "fps": fps, "frames": total_frames, "persona_key": persona_key, "chunks": rows}
    if restored:
        per_chunk = num_frames / float(fps)
        report["first_chunk_restored_seconds"] = restored[0][1]
        report["last_chunk_restored_seconds"] = restored[-1][1]
        report["min_head_start_seconds"] = round(max(t - i * per_chunk for i, t in restored), 3)
        span = restored[-1][1] - min(r.get("submit", restored[0][1]) for r in rows)
        report["restored_fps_aggregate"] = round(total_frames / span, 2) if span > 0 else None
        cached = [r.get("persona_cached") for r in rows if "persona_cached" in r]
        if cached:
            report["persona_cache_hits"] = int(sum(1 for c in cached if c))
    return report


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray, landmarks=None, visible=None):
        # CPU patch — two-pass landmark-smoothed affine.
        #
        # Upstream did detection + affine in a single per-frame call,
        # which meant sub-pixel landmark noise (0.1–0.5 px from libx264
        # re-encodes and similar) got amplified through the SVD-derived
        # `(s2/s1) * R` scale in `transformation_from_points` into
        # pixel-scale drift in the affine matrix — and visible
        # face-bouncing in the output. See the full bisection in
        # scripts/latentsync_debug/DEBUG_PLAN.md.
        #
        # We now do two passes:
        #   1) Extract per-frame 3-point landmarks across the whole clip.
        #   2) Temporally smooth each landmark's (x, y) trajectory with
        #      a Savitzky-Golay filter (see _smooth_landmarks_sequence).
        #   3) Compute the per-frame affine from the smoothed landmarks.
        #
        # Smoothing landmarks *before* the SVD prevents the noise from
        # entering the affine at all. Our existing affine-matrix smoother
        # (`_smooth_affine_sequence`) still runs afterwards for belt-and-
        # suspenders, but is largely redundant with noise-free inputs.
        #
        # Window default 5 (0.2 s at 25 fps) is a conservative floor —
        # large enough to damp high-frequency detection noise, small
        # enough not to lag head turns. Set LATENTSYNC_LANDMARK_SMOOTH_WINDOW
        # to 1 (or 0) to disable; larger values (7, 9) smooth more aggressively.
        landmark_window = int(
            os.environ.get("LATENTSYNC_LANDMARK_SMOOTH_WINDOW", "5"),
        )

        # Resilience knob. A single frame without a detected face
        # (eyes closed, motion blur, partial occlusion, cut-to-insert)
        # used to kill the whole clip with "Face not detected". Now
        # we carry last-good landmarks forward through gaps and only
        # bail when the fraction of missing frames exceeds this
        # threshold — at which point the clip likely has no real
        # face content and we want to fail fast rather than produce
        # nonsense output. Default 0.5 = up to half the frames can
        # legitimately have no detection. Set lower for stricter,
        # higher for more tolerant. 1.0 means "never fail here"
        # (the all-None case still errors because there's literally
        # no geometry to work with).
        max_miss_ratio = float(
            os.environ.get("LATENTSYNC_MAX_MISSING_FACE_RATIO", "0.5"),
        )

        detect_started = time.perf_counter()
        if landmarks is not None:
            # Shared face track (latentsync_driver.face_track): detection and
            # whole-clip smoothing already happened once for the source, so
            # this window only needs the warp pass below.
            if len(landmarks) != len(video_frames):
                raise RuntimeError(
                    f"face track has {len(landmarks)} entries for {len(video_frames)} frames"
                )
            per_frame_landmarks = [np.asarray(l, dtype=np.float32) for l in landmarks]
            landmark_window = 1
            if visible is not None and len(visible) != len(video_frames):
                raise RuntimeError(
                    f"face visibility has {len(visible)} entries for {len(video_frames)} frames"
                )
            print(f"Using shared face track for {len(video_frames)} frames (detection skipped)")
        else:
            # Pass 1: extract landmarks per frame (this is what detection
            # actually costs; tqdm shows the expected 1 s/frame wall clock).
            # Use the non-raising try_extract so a single bad frame doesn't
            # fail the whole run. Gaps are filled after the pass completes.
            print(f"Extracting landmarks from {len(video_frames)} frames...")
            per_frame_landmarks = []
            confidences = []
            for frame in tqdm.tqdm(video_frames, desc="detect"):
                landmarks3, score = self.image_processor.try_extract_with_score(frame)
                per_frame_landmarks.append(landmarks3)
                confidences.append(score)
            if visible is None:
                visible = np.asarray(confidences, dtype=np.float32)

        total_frames = len(per_frame_landmarks)
        per_frame_landmarks, missing_indices = _fill_missing_landmarks(
            per_frame_landmarks,
        )
        if visible is None:
            visible = np.ones(total_frames, dtype=np.float32)
        # Detection confidence per frame, 0.0 where the landmarks are carried
        # over from a neighbour: the paste-back must not trust those frames
        # (latentsync_driver.face_parse).
        visible = np.asarray(visible, dtype=np.float32).copy()
        visible[missing_indices] = 0.0
        if missing_indices:
            miss_ratio = len(missing_indices) / total_frames
            # Log the first few indices so the user can pinpoint bad
            # frames quickly; a dump of all of them when hundreds are
            # missing is noisy.
            preview = missing_indices[:10]
            ellipsis = " ..." if len(missing_indices) > 10 else ""
            print(
                f"Face detection: {len(missing_indices)}/{total_frames} frame(s) "
                f"had no detectable face ({miss_ratio:.1%}). "
                f"Carrying last-good landmarks through gaps. "
                f"Missing indices: {preview}{ellipsis}"
            )
            if not per_frame_landmarks:
                # All-None: nothing to carry forward. Fail cleanly.
                raise RuntimeError(
                    f"no face detected in any of {total_frames} frames. "
                    f"Clip has no usable face content; check the input or "
                    f"trim to a face-visible range."
                )
            if miss_ratio > max_miss_ratio:
                raise RuntimeError(
                    f"{len(missing_indices)}/{total_frames} frames "
                    f"({miss_ratio:.1%}) lack a detected face, exceeding "
                    f"LATENTSYNC_MAX_MISSING_FACE_RATIO={max_miss_ratio}. "
                    f"Either raise the threshold or trim the clip to a "
                    f"face-visible range. First missing indices: {preview}"
                )

        # Temporal smoothing.
        if landmark_window > 1 and len(per_frame_landmarks) > 1:
            per_frame_landmarks = _smooth_landmarks_sequence(
                per_frame_landmarks, window=landmark_window,
            )
            print(
                f"Smoothed {len(per_frame_landmarks)} landmark trajectories "
                f"(window={landmark_window}) — damps sub-pixel noise before the SVD."
            )

        # Pass 2: compute affine + canonical face crop per frame using
        # the smoothed landmarks.
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        warp_started = time.perf_counter()
        for frame, landmarks3 in tqdm.tqdm(
            zip(video_frames, per_frame_landmarks),
            total=len(video_frames),
            desc="warp",
        ):
            face, box, affine_matrix = self.image_processor.affine_transform(
                frame, landmarks3=landmarks3,
            )
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        print(json.dumps({"event": "latentsync_face_prep", "frames": len(video_frames),
                          "detect_seconds": round(warp_started - detect_started, 2),
                          "warp_seconds": round(time.perf_counter() - warp_started, 2),
                          "shared_track": landmarks is not None,
                          "occluded_frames": int((visible <= 0).sum())}), flush=True)
        return faces, boxes, affine_matrices, visible

    def restore_video(
        self,
        faces: torch.Tensor,
        video_frames: np.ndarray,
        boxes: list,
        affine_matrices: list,
        progress_callback=None,  # CPU patch: per-frame progress reporting
        face_masks=None,  # (N,1,fh,fw) source-face masks, see latentsync_driver.face_parse
        alpha=None,  # (N,) per-frame paste weight; 0 keeps the source frame
    ):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        total = len(faces)
        print(f"Restoring {total} faces...")
        restorer = self.image_processor.restorer
        device = self._execution_device
        # Frames travel host->device and device->host once per batch rather
        # than once per frame; the per-frame math is unchanged.
        batch = max(1, int(os.environ.get("LATENTSYNC_RESTORE_BATCH", "16")))
        debug = os.environ.get("LATENTSYNC_DEBUG_DUMP", "0") == "1"
        for start in range(0, total, batch):
            stop = min(total, start + batch)
            frames_t = torch.from_numpy(np.ascontiguousarray(video_frames[start:stop])).to(
                device=restorer.device, dtype=restorer.dtype, non_blocking=True
            ).permute(0, 3, 1, 2)
            composited = []
            for index in range(start, stop):
                x1, y1, x2, y2 = boxes[index]
                height = int(y2 - y1)
                width = int(x2 - x1)
                face = torchvision.transforms.functional.resize(
                    faces[index].to(device), size=(height, width),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
                )
                weight = 1.0 if alpha is None else float(alpha[index])
                if weight <= 0.0:
                    composited.append(frames_t[index - start])  # face not located: source frame as is
                    continue
                mask = None
                if face_masks is not None:
                    mask = face_masks[index].to(device=restorer.device, dtype=restorer.dtype, non_blocking=True).unsqueeze(0)
                composited.append(
                    restorer.restore_on_device(
                        frames_t[index - start], face, affine_matrices[index], debug=debug,
                        face_mask=mask, alpha=weight,
                    )
                )
            out = torch.stack(composited).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            out_frames.append(out.cpu().numpy())
            # Progress over the restore phase; __call__ maps it into 0.90–0.98.
            if progress_callback is not None:
                try:
                    progress_callback(stop / total)
                except Exception:
                    pass
        return np.concatenate(out_frames, axis=0)

    def _smooth_affines(self, affine_matrices, boxes):
        """Temporal smoothing of the per-frame affine + boxes (with the
        optional debug dumps). Runs before denoise so chunks can be restored
        as soon as they are denoised."""
        # CPU patch: temporal smoothing on affine matrices + bboxes.
        # Landmark detection jitters a few pixels frame-to-frame (normal
        # for InsightFace on real video), and that jitter propagates
        # through the affine warp-back into visible face-bouncing:
        # translation drift, small rotations, and zoom in/out. This step
        # decomposes each affine into its 4 similarity parameters
        # (tx, ty, rotation, scale), smooths each independently with a
        # centered moving average over LATENTSYNC_AFFINE_SMOOTH_WINDOW
        # frames (default 9), then recomposes. Bboxes are smoothed
        # component-wise over the same window.
        #
        # Default raised from 5 → 9 because real-world jitter has
        # components slower than the 5-frame (0.2 s) budget — zoom and
        # rotation noise compounds visibly over longer windows. 9 frames
        # (0.36 s at 25 fps) catches those without lagging intentional
        # head motion.
        #
        # Set LATENTSYNC_AFFINE_SMOOTH_WINDOW=1 (or 0) to disable.
        _smooth_window = int(os.environ.get("LATENTSYNC_AFFINE_SMOOTH_WINDOW", "9"))

        # CPU patch — diagnostic dump of affine matrices + boxes, before
        # and after smoothing. Gated by env var so it costs nothing in
        # production. Used to verify whether affine matrices are
        # bit-identical across frames on static input (which should
        # follow from deterministic face detection) or drift by a small
        # epsilon that would explain pipeline-introduced jitter.
        # See scripts/latentsync_debug/DEBUG_PLAN.md Step 3.
        if os.environ.get("LATENTSYNC_DUMP_AFFINES", "0") == "1":
            try:
                import numpy as np
                dump_dir = os.environ.get(
                    "LATENTSYNC_DUMP_AFFINES_DIR", "/jobs/affine_debug",
                )
                os.makedirs(dump_dir, exist_ok=True)
                _to_np = lambda m: m if isinstance(m, np.ndarray) else m.cpu().numpy()
                pre = np.stack([_to_np(m) for m in affine_matrices], axis=0)
                pre_boxes = np.array(boxes)
                np.save(os.path.join(dump_dir, "affines_pre_smooth.npy"), pre)
                np.save(os.path.join(dump_dir, "boxes_pre_smooth.npy"), pre_boxes)
                print(
                    f"LATENTSYNC_DUMP_AFFINES=1: wrote pre-smooth affines "
                    f"({pre.shape}) + boxes ({pre_boxes.shape}) to {dump_dir}"
                )
            except Exception as _e:
                print(f"affine dump (pre-smooth) failed: {_e}")

        if _smooth_window > 1 and len(affine_matrices) > 1:
            affine_matrices, boxes = _smooth_affine_sequence(
                affine_matrices, boxes, window=_smooth_window,
            )
            print(
                f"Smoothed {len(affine_matrices)} affine matrices + boxes "
                f"(window={_smooth_window}) to reduce face-jitter."
            )

        if os.environ.get("LATENTSYNC_DUMP_AFFINES", "0") == "1":
            try:
                import numpy as np
                dump_dir = os.environ.get(
                    "LATENTSYNC_DUMP_AFFINES_DIR", "/jobs/affine_debug",
                )
                _to_np = lambda m: m if isinstance(m, np.ndarray) else m.cpu().numpy()
                post = np.stack([_to_np(m) for m in affine_matrices], axis=0)
                post_boxes = np.array(boxes)
                np.save(os.path.join(dump_dir, "affines_post_smooth.npy"), post)
                np.save(os.path.join(dump_dir, "boxes_post_smooth.npy"), post_boxes)
                print(
                    f"LATENTSYNC_DUMP_AFFINES=1: wrote post-smooth affines "
                    f"({post.shape}) to {dump_dir}"
                )
            except Exception as _e:
                print(f"affine dump (post-smooth) failed: {_e}")
        return affine_matrices, boxes

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray, landmarks=None, visible=None):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices, visible = self.affine_transform_video(video_frames, landmarks, visible)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_visible = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_visible.append(visible)
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_visible.append(visible[::-1])

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            visible = np.concatenate(loop_visible, axis=0)[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            if landmarks is not None:
                landmarks = landmarks[: len(video_frames)]
            if visible is not None:
                visible = visible[: len(video_frames)]
            faces, boxes, affine_matrices, visible = self.affine_transform_video(video_frames, landmarks, visible)

        return video_frames, faces, boxes, affine_matrices, visible

    def prepare_inputs(self, video_path: str, audio_path: str, video_fps: int, window_landmarks=None) -> dict:
        """Everything a window needs before denoising: audio features, decoded
        frames, warped faces, smoothed affines. Independent of the UNet, so it
        can run for window N+1 while window N denoises (see prepare_ahead)."""
        started = time.perf_counter()
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)
        window_visible = None
        if window_landmarks is not None:
            from latentsync_driver.face_track import slice_for_window

            offset = int(window_landmarks.get("offset", 0))
            if window_landmarks.get("visible") is not None:
                window_visible = slice_for_window(
                    np.asarray(window_landmarks["visible"], dtype=np.float32), offset, len(video_frames)
                )
            window_landmarks = slice_for_window(window_landmarks["landmarks"], offset, len(video_frames))
        video_frames, faces, boxes, affine_matrices, visible = self.loop_video(
            whisper_chunks, video_frames, window_landmarks, window_visible
        )
        affine_matrices, boxes = self._smooth_affines(affine_matrices, boxes)
        # Occlusion handling (latentsync_driver.face_parse): a per-pixel face
        # mask from the source crops and a per-frame weight from detection.
        from latentsync_driver import face_parse

        parse_started = time.perf_counter()
        face_masks, covered = self.face_parser().masks(
            faces, out_size=self.image_processor.restorer.face_size[::-1], visible=visible,
            frames=video_frames, affines=affine_matrices,
        )
        gate = np.asarray(visible, dtype=np.float32).copy()
        covered_frames = 0
        if face_masks is not None:
            face_masks = face_masks.to(torch.float16).cpu()
            hidden = covered.numpy() > face_parse.MOUTH_COVERED
            covered_frames = int(hidden.sum())
            gate[hidden] = 0.0  # an occluder over the mouth: treat like a lost face
        alpha = face_parse.occlusion_alpha(gate, margin=2, ramp=3)
        print(json.dumps({"event": "latentsync_occlusion", "frames": len(faces),
                          "parsed": face_masks is not None, "gated_frames": int((alpha < 1).sum()),
                          "no_face_frames": int((np.asarray(visible) <= 0).sum()), "covered_mouth_frames": covered_frames,
                          "gated": face_parse.ranges(np.flatnonzero(alpha <= 0.0)),
                          "seconds": round(time.perf_counter() - parse_started, 2)}), flush=True)
        return {
            "whisper_chunks": whisper_chunks, "audio_samples": audio_samples, "video_frames": video_frames,
            "faces": faces, "boxes": boxes, "affine_matrices": affine_matrices,
            "face_masks": face_masks, "alpha": alpha,
            "seconds": time.perf_counter() - started,
        }

    def face_parser(self):
        parser = getattr(self, "_face_parser", None)
        if parser is None:
            from latentsync_driver.face_parse import FaceParser

            parser = self._face_parser = FaceParser(self._execution_device)
        return parser

    def prepare_ahead(self, video_path: str, audio_path: str, video_fps: int, face_track=None) -> dict:
        """Compute and cache the inputs for a window that will be requested
        next. Called from the /lipsync/prepare endpoint while the previous
        window denoises. At most two windows are kept."""
        prepared = getattr(self, "_prepared", None)
        if prepared is None:
            prepared = self._prepared = {}
        key = (str(video_path), str(audio_path))
        inputs = self.prepare_inputs(str(video_path), str(audio_path), video_fps, face_track)
        inputs["mtime"] = (os.path.getmtime(video_path), os.path.getmtime(audio_path))
        prepared[key] = inputs
        while len(prepared) > 2:
            prepared.pop(next(iter(prepared)))
        return {"frames": int(len(inputs["video_frames"])), "seconds": round(inputs["seconds"], 2)}

    def _take_prepared(self, video_path: str, audio_path: str):
        prepared = getattr(self, "_prepared", None) or {}
        inputs = prepared.pop((str(video_path), str(audio_path)), None)
        if inputs is None:
            return None
        try:
            if inputs["mtime"] != (os.path.getmtime(video_path), os.path.getmtime(audio_path)):
                return None  # files were rewritten after preparation
        except OSError:
            return None
        return inputs

    def ensure_image_processor(self, height: int, mask_image_path: str):
        """Build (once) the ImageProcessor used for detection and warping.

        Its FaceDetector holds two onnxruntime CUDA sessions that took ~43 s
        to create on the XE7740, so it is cached across requests. The face
        track builder needs it before __call__ runs.
        """
        device = self._execution_device
        ip_key = (int(height), str(device), str(mask_image_path))
        if getattr(self, "_image_processor_key", None) != ip_key:
            mask_image = load_fixed_mask(height, mask_image_path)
            self.image_processor = ImageProcessor(height, device=str(device), mask_image=mask_image)
            self._image_processor_key = ip_key
        return self.image_processor

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        # GPU patch: reuse the ImageProcessor across requests (see
        # ensure_image_processor). Device comes from self._execution_device
        # so CPU paths do not hit ImageProcessor's internal .to("cuda").
        self.ensure_image_processor(height, mask_image_path)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # CPU patch: real-progress reporting. When the caller supplies a
        # `progress_callback`, we call it at phase boundaries and within
        # the denoise/restore loops. The driver wires this through to a
        # JSON file on the shared /jobs volume that the backend's
        # orchestrator polls. Replaces the backend's time-based-elapsed-
        # divided-by-estimated-ETA heuristic which consistently hit 98%
        # two minutes into a run and plateaued there for hours.
        #
        # Budget: face_detect 0–0.25, denoise 0.25–0.90, restore 0.90–0.98,
        # mux 0.98–1.00. Tuned to observed wall-clock proportions at
        # 1080p / bf16 / DeepCache (denoise dominates by a wide margin).
        progress_callback = kwargs.get("progress_callback")
        def _emit_progress(phase: str, pct: float) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(phase, float(max(0.0, min(1.0, pct))))
            except Exception:
                # Progress reporting must never break the pipeline.
                pass

        _emit_progress("face_detect", 0.01)

        # Resume support (CPU patch): try to load a post-denoise checkpoint.
        # Saves the expensive half of the pipeline (face detection +
        # denoising loop, typically 20+ minutes) on retries. Keyed by a
        # content hash computed in the driver — same inputs hit the cache
        # automatically. See docs/lipsync.md for the cache layout.
        denoise_checkpoint_path = kwargs.get("denoise_checkpoint_path")
        synced_video_frames_tensor = None
        video_frames = None
        boxes = None
        affine_matrices = None
        audio_samples = None

        if denoise_checkpoint_path and os.path.exists(denoise_checkpoint_path):
            try:
                print(f"Loading denoise checkpoint: {denoise_checkpoint_path}")
                cache = torch.load(
                    denoise_checkpoint_path, weights_only=False, map_location="cpu",
                )
                synced_video_frames_tensor = cache["synced_video_frames"]
                video_frames = cache["video_frames"]
                boxes = cache["boxes"]
                affine_matrices = cache["affine_matrices"]
                audio_samples = cache["audio_samples"]
                print(
                    f"Resume: skipping denoise "
                    f"({synced_video_frames_tensor.shape[0]} frames from cache)",
                )
                # Cache hit — skip face_detect + denoise budget entirely.
                _emit_progress("denoise", 0.90)
            except Exception as e:
                print(f"Checkpoint load failed ({e}); running full pipeline")
                synced_video_frames_tensor = None

        restored_frames = None
        face_masks, alpha = None, None
        if synced_video_frames_tensor is not None:
            affine_matrices, boxes = self._smooth_affines(affine_matrices, boxes)

        if synced_video_frames_tensor is None:
            stage_started = time.perf_counter()
            profile = {}
            # Per-chunk timeline (seconds since this call began, plus the wall
            # clock origin) so a streaming consumer's playout margin can be
            # computed offline: when each chunk was submitted, when its pixels
            # came back, and when it was pasted into the full frames.
            call_t0 = stage_started
            call_t0_wall = time.time()
            chunk_times: dict = {}
            persona_key = kwargs.get("persona_key")
            profile["persona_key"] = persona_key
            # Window inputs: decoded frames, warped faces (via the shared face
            # track when supplied), audio features. Prepared ahead by
            # /lipsync/prepare while the previous window denoised, otherwise
            # computed here.
            inputs = self._take_prepared(video_path, audio_path)
            profile["prepared_ahead"] = inputs is not None
            if inputs is None:
                inputs = self.prepare_inputs(video_path, audio_path, video_fps, kwargs.get("face_track"))
            whisper_chunks = inputs["whisper_chunks"]
            audio_samples = inputs["audio_samples"]
            video_frames = inputs["video_frames"]
            faces, boxes, affine_matrices = inputs["faces"], inputs["boxes"], inputs["affine_matrices"]
            face_masks, alpha = inputs.get("face_masks"), inputs.get("alpha")
            profile["decode_audio_face_seconds"] = time.perf_counter() - stage_started
            profile["conditioning_seconds"] = 0.0
            profile["wait_for_worker_seconds"] = 0.0
            profile["collect_and_paste_seconds"] = 0.0
            # Face detection + affine transform is done at this point.
            _emit_progress("face_detect", 0.25)

            synced_video_frames = []

            num_channels_latents = self.vae.config.latent_channels

            # Prepare latent variables
            all_latents = self.prepare_latents(
                len(whisper_chunks),
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )

            # CPU patch — UNet bypass for jitter bisection. When this env
            # var is set, we skip the denoise loop + VAE decode entirely
            # and pass the reference face crop straight through to the
            # restore/composite stage. Isolates geometric pipeline jitter
            # (affine warp + soft-mask compositing) from anything the
            # UNet / VAE numerical path introduces. See
            # scripts/latentsync_debug/DEBUG_PLAN.md Step 2b.
            bypass_unet = os.environ.get("LATENTSYNC_BYPASS_UNET", "0") == "1"
            if bypass_unet:
                print(
                    "LATENTSYNC_BYPASS_UNET=1: skipping denoise + VAE decode; "
                    "reference face crops will be passed through unchanged."
                )

            num_inferences = math.ceil(len(whisper_chunks) / num_frames)

            # GPU patch — sharded denoise. `unet_replicas` (set by the
            # driver) holds the UNet on the main device plus one copy per
            # additional visible GPU. The 16-frame chunks are independent
            # given the shared noise sample, so:
            #   A. main thread, main device: conditioning for every chunk
            #      (VAE encodes of masks / reference), as before;
            #   B. one thread per replica runs the 20-step UNet loop for
            #      its round-robin share of chunks and ships the latents
            #      back to the main device;
            #   C. main thread decodes + pastes in chunk order.
            # With one replica this is the original sequential loop, just
            # split into the same three phases. The scheduler is stateless
            # per step, `torch.no_grad()` is thread-local so each worker
            # enters it itself, and DeepCache (which patches `self.unet`
            # with per-call cache state) is disabled by the driver when
            # more than one replica is in play.
            replicas = list(getattr(self, "unet_replicas", None) or [self.unet])
            replica_devices = [next(u.parameters()).device for u in replicas]
            # Process-per-GPU pool (latentsync_driver/shard_workers.py) takes
            # precedence over in-process thread replicas when the driver
            # attached one. Each worker owns a UNet + VAE and returns decoded
            # pixels; this process only conditions and pastes.
            pool = getattr(self, "denoise_pool", None)
            use_pool = pool is not None and not bypass_unet
            if use_pool:
                print(f"Denoise sharded across {pool.size} worker processes on cuda:{pool.devices}")
                pool.begin_job(num_inference_steps, guidance_scale, do_classifier_free_guidance, eta)
            elif len(replicas) > 1:
                print(f"Denoise sharded across {len(replicas)} UNet replicas on {replica_devices}")

            # --- Workers (one per replica) start first and consume chunks
            # as Phase A produces them, so conditioning, denoising and the
            # decode below overlap instead of running back to back.
            todo_count = [0]
            denoised: dict = {}
            denoise_lock = threading.Lock()
            done_count = [0]
            ready: dict = {}  # chunk index -> threading.Event
            work_queues = [queue.Queue() for _ in replicas]
            failures: list = []
            stopping = threading.Event()

            vae_replicas = list(getattr(self, "vae_replicas", None) or [])

            def _denoise_chunk(unet, dev, k, cond):
                latents, mask_latents, masked_image_latents, ref_latents, audio_embeds, _, _ = cond
                mv = lambda x: None if x is None else x.to(dev)
                lat, ml, mil, rl, ae = mv(latents), mv(mask_latents), mv(masked_image_latents), mv(ref_latents), mv(audio_embeds)
                with torch.no_grad():
                    for t in timesteps:
                        unet_input = torch.cat([lat] * 2) if do_classifier_free_guidance else lat
                        unet_input = self.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, ml, mil, rl], dim=1)
                        noise_pred = unet(unet_input, t.to(dev), encoder_hidden_states=ae).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                        lat = self.scheduler.step(noise_pred, t, lat, **extra_step_kwargs).prev_sample
                    if k < len(vae_replicas) and vae_replicas[k] is not None:
                        # Decode here, on this device, so the main GPU is
                        # not left doing all 78 decodes after the workers
                        # finish. Returns pixels; Phase C only pastes.
                        vae = vae_replicas[k]
                        z = lat / vae.config.scaling_factor + vae.config.shift_factor
                        z = rearrange(z, "b c f h w -> (b f) c h w")
                        return ("pixels", vae.decode(z).sample.to(device))
                return ("latents", lat.to(device))

            def _worker(k: int):
                unet, dev = replicas[k], replica_devices[k]
                q = work_queues[k]
                while True:
                    item = q.get()
                    if item is None:
                        return
                    i, cond = item
                    if stopping.is_set():
                        ready[i].set()
                        continue
                    try:
                        out = _denoise_chunk(unet, dev, k, cond)
                    except Exception as e:  # surface on the main thread
                        with denoise_lock:
                            failures.append((i, e))
                        ready[i].set()
                        continue
                    with denoise_lock:
                        denoised[i] = out
                        done_count[0] += 1
                        _emit_progress("denoise", 0.35 + 0.50 * (done_count[0] / max(1, todo_count[0] or num_inferences)))
                    ready[i].set()

            threads = []
            if not use_pool:
                threads = [threading.Thread(target=_worker, args=(k,), daemon=True) for k in range(len(replicas))]
                for th in threads:
                    th.start()

            # Restore (paste faces back into the 1080p frames) runs on a side
            # thread per chunk as soon as the chunk is denoised, instead of as
            # a serial stage after the last chunk. Disabled when a denoise
            # checkpoint must be saved (that path needs the pre-restore tensor).
            from concurrent.futures import ThreadPoolExecutor
            overlap_restore = (
                not denoise_checkpoint_path
                and os.environ.get("LATENTSYNC_OVERLAP_RESTORE", "1") == "1"
            )
            restore_pool = ThreadPoolExecutor(max_workers=1) if overlap_restore else None
            restore_futures: dict = {}
            restore_seconds = [0.0]

            def restore_chunk(index, decoded):
                a = index * num_frames
                b = min(len(video_frames), a + len(decoded))
                started_at = time.perf_counter()
                out = self.restore_video(
                    decoded[: b - a], video_frames[a:b], boxes[a:b], affine_matrices[a:b],
                    face_masks=None if face_masks is None else face_masks[a:b],
                    alpha=None if alpha is None else alpha[a:b],
                )
                restore_seconds[0] += time.perf_counter() - started_at
                chunk_times.setdefault(index, {})["restored"] = round(time.perf_counter() - call_t0, 3)
                return out

            def collect_chunk(index):
                collect_started = time.perf_counter()
                _, _, _, _, _, ref_pixels, chunk_masks = chunk_cond[index]
                if use_pool:
                    decoded = pool.result(index).to(device, dtype=weight_dtype)
                    done_count[0] += 1
                    entry = chunk_times.setdefault(index, {})
                    entry["result"] = round(time.perf_counter() - call_t0, 3)
                    timing = pool.timings.get(index)
                    if timing:
                        entry["worker_received"] = round(timing["received"] - call_t0_wall, 3)
                        entry["worker_done"] = round(timing["done"] - call_t0_wall, 3)
                        entry["device"] = timing.get("device")
                        entry["persona_cached"] = timing.get("persona_cached", False)
                else:
                    ready[index].wait()
                    if failures:
                        raise failures[0][1]
                    kind, payload = denoised.pop(index)
                    decoded = payload if kind == "pixels" else self.decode_latents(payload)
                profile["wait_for_worker_seconds"] += time.perf_counter() - collect_started
                paste_started = time.perf_counter()
                decoded = self.paste_surrounding_pixels_back(
                    decoded, ref_pixels, 1 - chunk_masks, device, weight_dtype)
                if restore_pool is not None:
                    restore_futures[index] = restore_pool.submit(restore_chunk, index, decoded)
                else:
                    synced_video_frames.append(decoded.cpu())
                chunk_cond[index] = None
                profile["collect_and_paste_seconds"] += time.perf_counter() - paste_started
                _emit_progress("denoise", 0.35 + 0.50 * done_count[0] / num_inferences)

            try:
                collected = 0
                # --- Phase A: conditioning per chunk (main device) -------------
                chunk_cond: list = []
                decoded_by_chunk: dict = {}
                next_worker = 0
                prefetch = max(1, min(4, int(os.environ.get("LATENTSYNC_PREFETCH_PER_WORKER", "1"))))
                for i in tqdm.tqdm(range(num_inferences), desc="Preparing chunks..."):
                    conditioning_started = time.perf_counter()
                    if self.unet.add_audio_layer:
                        audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                        audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                        if do_classifier_free_guidance:
                            null_audio_embeds = torch.zeros_like(audio_embeds)
                            audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
                    else:
                        audio_embeds = None
                    inference_faces = faces[i * num_frames : (i + 1) * num_frames]
                    latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                    ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                        inference_faces, affine_transform=False
                    )

                    if bypass_unet:
                        # Short-circuit: use the reference face crop as the
                        # "generated" output. Skips denoise+VAE entirely.
                        decoded_latents = ref_pixel_values.to(dtype=weight_dtype)
                        decoded_latents = self.paste_surrounding_pixels_back(
                            decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                        )
                        decoded_by_chunk[i] = decoded_latents.cpu()
                        chunk_cond.append(None)
                        continue

                    worker_conditioning = use_pool and os.environ.get(
                        "LATENTSYNC_WORKER_CONDITIONING", "1"
                    ) == "1"
                    if worker_conditioning:
                        # The worker encodes masks/masked/reference crops with
                        # its own VAE (shard_workers.py). Profiling showed the
                        # coordinator spending ~50% of its time in the VAE
                        # encode and the sync it forced; the pixel crops are
                        # ~19 MB per chunk over /dev/shm, which is cheap.
                        cond = (latents, None, None, None, audio_embeds, ref_pixel_values, masks)
                        chunk_cond.append(cond)
                        todo_count[0] += 1
                        chunk_times.setdefault(i, {})["submit"] = round(time.perf_counter() - call_t0, 3)
                        audio_cpu = None if audio_embeds is None else audio_embeds.detach().to("cpu")
                        if persona_key:
                            # Resident persona: the worker keeps this chunk's
                            # footage latents after the first reply; the crops
                            # still travel (cheap over /dev/shm) so a cold
                            # worker can build them.
                            pool.submit(i, ("persona", persona_key, i, latents.detach().to("cpu"), audio_cpu, *(
                                x.detach().to("cpu") for x in (masks, masked_pixel_values, ref_pixel_values)
                            )))
                        else:
                            pool.submit(i, ("pixels", *(
                                x.detach().to("cpu") for x in (latents, masks, masked_pixel_values, ref_pixel_values)
                            ), audio_cpu))
                    else:
                        # 7. Prepare mask latent variables
                        mask_latents, masked_image_latents = self.prepare_mask_latents(
                            masks,
                            masked_pixel_values,
                            height,
                            width,
                            weight_dtype,
                            device,
                            generator,
                            do_classifier_free_guidance,
                        )
                        # 8. Prepare image latents
                        ref_latents = self.prepare_image_latents(
                            ref_pixel_values,
                            device,
                            weight_dtype,
                            generator,
                            do_classifier_free_guidance,
                        )
                        cond = (latents, mask_latents, masked_image_latents, ref_latents,
                                audio_embeds, ref_pixel_values, masks)
                        chunk_cond.append(cond)
                        todo_count[0] += 1
                    if worker_conditioning:
                        pass
                    elif use_pool:
                        pool.submit(i, tuple(
                            None if x is None else x.detach().to("cpu")
                            for x in (latents, mask_latents, masked_image_latents, ref_latents, audio_embeds)
                        ))
                    else:
                        ready[i] = threading.Event()
                        work_queues[next_worker].put((i, cond))
                        next_worker = (next_worker + 1) % len(replicas)
                    profile["conditioning_seconds"] += time.perf_counter() - conditioning_started
                    # Bounded prefetch overlaps coordinator work with workers.
                    if i - collected + 1 >= prefetch * max(1, pool.size if use_pool else len(replicas)):
                        collect_chunk(collected)
                        collected += 1
                    _emit_progress("denoise", 0.25 + 0.10 * ((i + 1) / num_inferences))

                if not use_pool:
                    for q in work_queues:
                        q.put(None)

                # --- Phase C: decode + paste, in order, as results land --------
                for i in tqdm.tqdm(range(collected, num_inferences), desc="Decoding chunks..."):
                    if chunk_cond[i] is None:
                        if restore_pool is not None:
                            restore_futures[i] = restore_pool.submit(
                                restore_chunk, i, decoded_by_chunk[i].to(device, dtype=weight_dtype)
                            )
                        else:
                            synced_video_frames.append(decoded_by_chunk[i])
                        continue
                    collect_chunk(i)

                for th in threads:
                    th.join()
                for dev in replica_devices[1:]:
                    torch.cuda.synchronize(dev)

            finally:
                # A failed request must not leave threads touching the scheduler
                # or models after the service releases its inference lock.
                stopping.set()
                for q in work_queues:
                    q.put(None)
                for th in threads:
                    th.join()

            # Always printed: perf_counter sums only, no stream syncs. This is
            # the stage breakdown operators need to pick the next optimization.
            profile.update(event="latentsync_stages", frames=len(whisper_chunks), steps=num_inference_steps, workers=pool.size if use_pool else len(replicas), prefetch=prefetch)
            profile = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in profile.items()}
            print(json.dumps(profile), flush=True)

            if restore_pool is not None:
                # Chunks already restored on the side thread; join in order.
                restored_frames = np.concatenate(
                    [restore_futures[i].result() for i in range(num_inferences)], axis=0
                )
                restore_pool.shutdown(wait=True)
                print(json.dumps({"event": "latentsync_restore_overlapped", "frames": int(len(restored_frames)),
                                  "restore_seconds_on_side_thread": round(restore_seconds[0], 2)}), flush=True)
                synced_video_frames_tensor = None
            else:
                # Consolidate all chunk outputs into one tensor, then save for
                # resume. Cache write is best-effort: if disk is full or the
                # path is unwritable we log and continue — the live run still
                # completes; only a future retry would miss the cache.
                synced_video_frames_tensor = torch.cat(synced_video_frames)
            print(json.dumps(chunk_timeline_report(chunk_times, num_frames, video_fps, call_t0_wall,
                                                   len(whisper_chunks), persona_key)), flush=True)
            if denoise_checkpoint_path and synced_video_frames_tensor is not None:
                try:
                    print(f"Saving denoise checkpoint: {denoise_checkpoint_path}")
                    os.makedirs(os.path.dirname(denoise_checkpoint_path), exist_ok=True)
                    torch.save(
                        {
                            "synced_video_frames": synced_video_frames_tensor,
                            "video_frames": video_frames,
                            "boxes": boxes,
                            "affine_matrices": affine_matrices,
                            "audio_samples": audio_samples,
                        },
                        denoise_checkpoint_path,
                    )
                except Exception as e:
                    print(f"Checkpoint save failed ({e}); continuing anyway")

        _emit_progress("restore", 0.90)
        restore_started = time.perf_counter()
        if restored_frames is not None:
            synced_video_frames = restored_frames
        else:
            synced_video_frames = self.restore_video(
                synced_video_frames_tensor, video_frames, boxes, affine_matrices,
                face_masks=face_masks, alpha=alpha,
                progress_callback=(
                    lambda frame_pct: _emit_progress(
                        "restore", 0.90 + 0.08 * frame_pct,
                    )
                ),
            )

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        _emit_progress("mux", 0.98)
        # GPU patch: one encode+mux pass (NVENC when available) instead of
        # imageio libx264 crf 13 followed by a second libx264 crf 18 pass.
        write_started = time.perf_counter()
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)
        write_video_with_audio(
            video_out_path, synced_video_frames, fps=video_fps,
            audio_wav_path=os.path.join(temp_dir, "audio.wav"),
        )
        print(json.dumps({"event": "latentsync_finish", "frames": int(synced_video_frames.shape[0]),
                          "restore_seconds": round(write_started - restore_started, 2),
                          "write_seconds": round(time.perf_counter() - write_started, 2)}), flush=True)
        _emit_progress("done", 1.0)
