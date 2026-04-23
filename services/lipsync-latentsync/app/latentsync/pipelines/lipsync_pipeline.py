# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

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
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
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

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(
        self,
        faces: torch.Tensor,
        video_frames: np.ndarray,
        boxes: list,
        affine_matrices: list,
        progress_callback=None,  # CPU patch: per-frame progress reporting
    ):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        total = len(faces)
        print(f"Restoring {total} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
            # Per-frame progress. Caller gets a 0.0–1.0 fraction over
            # the restore phase; __call__ maps it into the pipeline's
            # 0.90–0.98 budget.
            if progress_callback is not None:
                try:
                    progress_callback((index + 1) / total)
                except Exception:
                    pass
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

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
        mask_image = load_fixed_mask(height, mask_image_path)
        # CPU patch: upstream hard-coded device="cuda" here even though
        # `device` (self._execution_device) is already defined on the
        # line above. Use it so CPU execution paths don't crash in
        # ImageProcessor's internal .to("cuda") calls.
        self.image_processor = ImageProcessor(height, device=str(device), mask_image=mask_image)
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
                    denoise_checkpoint_path, weights_only=False, map_location=device,
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

        if synced_video_frames_tensor is None:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            audio_samples = read_audio(audio_path)
            video_frames = read_video(video_path, use_decord=False)

            video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)
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
            for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
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
                    # "generated" output. Skips denoise+VAE entirely. The
                    # shape of ref_pixel_values matches what decode_latents
                    # would return (both are in pixel space after VAE-round-trip
                    # normalization), so the downstream paste + restore
                    # pipeline treats it identically.
                    decoded_latents = ref_pixel_values.to(dtype=weight_dtype)
                    decoded_latents = self.paste_surrounding_pixels_back(
                        decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                    )
                    synced_video_frames.append(decoded_latents)
                    # Emit the same progress signal we would have from the
                    # denoise loop so the backend's progress bar still moves.
                    _total = num_inferences
                    _emit_progress("denoise", 0.25 + 0.65 * ((i + 1) / _total))
                    continue

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

                # 9. Denoising loop
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        unet_input = self.scheduler.scale_model_input(unet_input, t)

                        # concat latents, mask, masked_image_latents in the channel dimension
                        unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                        # predict the noise residual
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents)

                        # Real-progress emit (every step). Progress within
                        # the denoise budget (0.25 -> 0.90) linearly tracks
                        # total-steps-done / total-steps. "Total" here =
                        # num_inferences * num_inference_steps.
                        _total_steps = num_inferences * num_inference_steps
                        _done_steps = i * num_inference_steps + (j + 1)
                        _emit_progress(
                            "denoise",
                            0.25 + 0.65 * (_done_steps / _total_steps),
                        )

                # Recover the pixel values
                decoded_latents = self.decode_latents(latents)
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)

            # Consolidate all chunk outputs into one tensor, then save for
            # resume. Cache write is best-effort: if disk is full or the
            # path is unwritable we log and continue — the live run still
            # completes; only a future retry would miss the cache.
            synced_video_frames_tensor = torch.cat(synced_video_frames)
            if denoise_checkpoint_path:
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

        _emit_progress("restore", 0.90)
        synced_video_frames = self.restore_video(
            synced_video_frames_tensor, video_frames, boxes, affine_matrices,
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
        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
        _emit_progress("done", 1.0)
