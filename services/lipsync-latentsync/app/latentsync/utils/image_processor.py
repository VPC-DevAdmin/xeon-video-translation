# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        # CPU patch: upstream refused face detection on CPU because
        # their FaceDetector defaulted to CUDA-only providers. We
        # patched FaceDetector to use onnxruntime's CPU provider when
        # device=="cpu", so this no-op-on-cpu guard is now actively
        # wrong for us. Always initialize.
        self.face_detector = FaceDetector(device=device)

    def _landmarks3_from_detection(self, landmark_2d_106) -> np.ndarray:
        """Project the 106-point detection output to our 3 anchor points
        (left-eye / right-eye / nose centers) in the shape our affine
        warp expects.
        """
        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)
        return np.array([pt_left_eye, pt_right_eye, pt_nose], dtype=np.float64)

    def extract_landmarks3(self, image) -> np.ndarray:
        """Return the 3 anchor landmarks (left-eye / right-eye / nose)
        this pipeline uses for its affine-to-canonical warp.

        Raises RuntimeError on detection failure. Use `try_extract_landmarks3`
        for a no-raise variant that returns None — appropriate when
        the caller wants to gap-fill across frames.
        """
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")
        return self._landmarks3_from_detection(landmark_2d_106)

    def try_extract_landmarks3(self, image) -> np.ndarray | None:
        """Non-raising variant of `extract_landmarks3`.

        Returns the 3-anchor array on detection success, or None when
        no face was found on this frame. Used by the video-pass code
        (`lipsync_pipeline.affine_transform_video`) to carry landmarks
        forward across frames with no detected face, rather than
        failing the whole clip on a single bad frame (eyes closed,
        motion blur, partial occlusion, a cut to a non-face insert).
        """
        try:
            bbox, landmark_2d_106 = self.face_detector(image)
        except Exception:
            # Detector raised (ONNX runtime glitch, unexpected input
            # shape, etc.). Treat like a missed detection; the caller
            # handles the None-gap.
            return None
        if bbox is None:
            return None
        return self._landmarks3_from_detection(landmark_2d_106)

    def affine_transform(
        self,
        image: torch.Tensor,
        landmarks3: np.ndarray | None = None,
    ) -> np.ndarray:
        # CPU patch — allow callers to pass pre-extracted, temporally
        # smoothed landmarks. When `landmarks3` is None (upstream's
        # default behavior), we detect + extract per-frame as before.
        # When provided, we skip detection and use the supplied points
        # directly — enabling the two-pass landmark-smoothing flow
        # in lipsync_pipeline.affine_transform_video().
        #
        # Upstream applied `np.round` to quantize to integer pixels.
        # That's a per-frame noise amplifier when raw landmark values
        # jitter across fractional pixel positions (e.g. 37.4, 37.5,
        # 37.4 → 37, 38, 37), so we skip rounding in the pre-computed
        # path; the smoother's job is already to hand us stable
        # sub-pixel values.
        if landmarks3 is None:
            landmarks3 = np.round(self.extract_landmarks3(image))

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)

        # Debug dump: the canonical face crop as it goes into the UNet.
        # Should look like an upright, centered face. If it doesn't, the
        # affine transform is wrong (either bad landmarks or bad matrix).
        try:
            from . import _debug
            _debug.dump("02_canonical_face_crop", face)
        except Exception:
            pass

        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
