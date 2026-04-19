"""BiSeNet face parsing — adapted from MuseTalk (MIT).

Paths are constructor args (no hard-coded `./models/...`). Device forced to
CPU for this build.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .model import BiSeNet


class FaceParsing:
    def __init__(
        self,
        model_path: str | Path,
        resnet_path: str | Path,
        device: torch.device | str = "cpu",
        left_cheek_width: int = 80,
        right_cheek_width: int = 80,
    ):
        self.device = torch.device(device)
        self.net = self._model_init(str(model_path), str(resnet_path))
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Jaw-region mask kernels (from upstream; keeps chin-correction behavior).
        cone_height = 21
        tail_height = 12
        total = cone_height + tail_height
        kernel = np.zeros((total, total), dtype=np.uint8)
        center_x = total // 2
        for row in range(cone_height):
            if row < cone_height // 2:
                continue
            width = int(2 * (row - cone_height // 2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1
        base_width = int(kernel[cone_height - 1].sum()) if cone_height > 0 else 1
        for row in range(cone_height, total):
            start = max(0, int(center_x - (base_width // 2)))
            end = min(total, int(center_x + (base_width // 2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel
        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
        self.cheek_mask = self._create_cheek_mask(left_cheek_width, right_cheek_width)

    @staticmethod
    def _create_cheek_mask(left: int, right: int) -> np.ndarray:
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left, 512), 255, -1)
        cv2.rectangle(mask, (center + right, 0), (512, 512), 255, -1)
        return mask

    def _model_init(self, model_path: str, resnet_path: str) -> BiSeNet:
        net = BiSeNet(resnet_path=resnet_path)
        net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        net.to(self.device)
        net.eval()
        return net

    def __call__(self, image, size: tuple[int, int] = (512, 512), mode: str = "raw") -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        width, height = image.size
        with torch.no_grad():
            resized = image.resize(size, Image.BILINEAR)
            img = self.preprocess(resized).unsqueeze(0).to(self.device)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            if mode == "neck":
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[parsing != 255] = 0
            elif mode == "jaw":
                face_region = (np.isin(parsing, [1]) * 255).astype(np.uint8)
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(
                    face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask)
                )
                parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
                parsing[np.isin(parsing, [11, 12, 13])] = 255
                parsing[parsing != 255] = 0
            elif mode == "mouth":
                # Tightest option: only upper lip, lower lip, and mouth/teeth.
                # Intentionally excludes any skin (class 1) so stubble around
                # the mouth survives.
                #
                # The naïve version of this mask produces a "ghost mouth" at
                # composite time: the downstream Gaussian feather has a
                # kernel wider than the mask itself, so even the center of
                # the lips gets blended at ~50% alpha with the original.
                # Dilating by ~7 px here gives the mask an opaque core large
                # enough that a modest feather doesn't erode the center.
                # The dilated ring bleeds slightly into the skin immediately
                # surrounding the lips — acceptable for the ghost fix; the
                # bulk of the beard/cheek stubble is still outside this.
                mask = (np.isin(parsing, [11, 12, 13]) * 255).astype(np.uint8)
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.dilate(mask, dilate_kernel, iterations=1)
                parsing[:] = 0
                parsing[mask == 255] = 255
            else:
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[parsing != 255] = 0

        return Image.fromarray(parsing.astype(np.uint8))
