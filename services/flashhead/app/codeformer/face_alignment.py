"""5-point face alignment — crop/warp to CodeFormer's 512×512 canonical.

CodeFormer is trained on faces aligned by a similarity transform that maps
five landmarks (eye centers, nose tip, mouth corners) to fixed positions
inside a 512×512 frame. Feeding it un-aligned faces produces noticeably
worse reconstruction. We get the five points from InsightFace's SCRFD
(face.kps is exactly the same five-point ordering).

Reference constants are from sczhou/CodeFormer/facelib/utils/face_utils.py.
"""

from __future__ import annotations

import numpy as np
import cv2

# Canonical 5-point positions inside a 512×512 face tile.
# Order: left-eye, right-eye, nose tip, left mouth corner, right mouth corner.
_CANONICAL_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

FACE_SIZE = 512


def align_to_canonical(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp `frame_bgr` so `kps` map onto the canonical 512×512 positions.

    Returns (aligned_512, M) where `M` is the 2×3 affine matrix used for
    the forward warp. Pass `M` to `paste_back()` to composite the CodeFormer
    output into the original frame.
    """
    if kps is None:
        raise ValueError("kps required for CodeFormer alignment")
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)

    # Similarity transform (rotation + uniform scale + translation).
    # cv2.estimateAffinePartial2D is the library helper for this.
    M, _ = cv2.estimateAffinePartial2D(
        src, _CANONICAL_512, method=cv2.LMEDS
    )
    if M is None:
        raise ValueError("could not compute alignment affine")

    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def paste_back(
    original: np.ndarray,
    restored_512: np.ndarray,
    M: np.ndarray,
    blend_weight: float = 0.6,
    feather_px: int = 11,
) -> np.ndarray:
    """Inverse-warp `restored_512` back into `original` using affine `M`.

    `blend_weight` in [0, 1] is how much of the CodeFormer output to blend
    on top of the original. Combined with a Gaussian-feathered elliptical
    mask so the seam is invisible.
    """
    h, w = original.shape[:2]
    inv_M = cv2.invertAffineTransform(M)

    warped = cv2.warpAffine(
        restored_512,
        inv_M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Elliptical alpha mask covering most of the 512×512 tile, then warped
    # to match `warped` orientation. Feathered so the boundary fades.
    mask_512 = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    cv2.ellipse(
        mask_512,
        center=(FACE_SIZE // 2, int(FACE_SIZE * 0.55)),
        axes=(int(FACE_SIZE * 0.42), int(FACE_SIZE * 0.55)),
        angle=0, startAngle=0, endAngle=360,
        color=255, thickness=-1,
    )
    if feather_px > 0:
        k = max(3, feather_px * 2 + 1)
        mask_512 = cv2.GaussianBlur(mask_512, (k, k), 0)

    warped_mask = cv2.warpAffine(
        mask_512,
        inv_M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32) / 255.0

    alpha = (warped_mask * float(blend_weight))[:, :, None]
    composite = (original.astype(np.float32) * (1.0 - alpha) +
                 warped.astype(np.float32) * alpha)
    return composite.clip(0, 255).astype(np.uint8)
