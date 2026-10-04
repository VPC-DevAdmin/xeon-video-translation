"""Device-resident still-portrait blending; RGB/BGR stays explicit.

The portrait and mask upload once per cached avatar. Generated faces stay on
CUDA through resize, feathering, composition and the disclosure overlay. Only
completed frames cross to the current host-memory WebRTC transport.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class PreparedBlend:
    image: torch.Tensor  # 1,3,H,W, BGR float32 0..255
    alpha: torch.Tensor  # 1,1,H,W
    box: tuple[int, int, int, int]
    disclosure: torch.Tensor | None = None


def prepare(image, box, mask, *, feather_ratio=0.05, disclosure=None):
    """Inputs are tensors already on the target device; mask is 2D 0..255."""
    if image.ndim != 3 or image.shape[0] != 3 or mask.ndim != 2:
        raise ValueError("expected CHW BGR image and 2D mask")
    x, y, x1, y1 = map(int, box)
    h, w = image.shape[-2:]
    if not (0 <= x < x1 <= w and 0 <= y < y1 <= h):
        raise ValueError("face box must be inside the portrait")
    xc, yc = (x + x1) // 2, (y + y1) // 2
    s = int(max(x1 - x, y1 - y) // 2 * 1.5)
    if s < 1:
        raise ValueError("face crop too small")
    xs, ys, xe, ye = xc - s, yc - s, xc + s, yc + s
    size = 2 * s
    m = F.interpolate(
        mask[None, None].float(), (size, size), mode="bilinear", align_corners=False
    ).round()
    restricted = torch.zeros_like(m)
    restricted[:, :, y - ys : y1 - ys, x - xs : x1 - xs] = m[
        :, :, y - ys : y1 - ys, x - xs : x1 - xs
    ]
    restricted[:, :, : size // 2] = 0
    k = max(3, min(31, int(feather_ratio * size // 2) * 2 + 1))
    # OpenCV's sigma=0 small kernels are exact binomial coefficients.
    small = {
        3: [1, 2, 1],
        5: [1, 4, 6, 4, 1],
        7: [2, 7, 14, 18, 14, 7, 2],
        9: [4, 13, 30, 51, 60, 51, 30, 13, 4],
    }
    if k in small:
        g = torch.tensor(small[k], device=image.device, dtype=torch.float32)
    else:
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        z = torch.arange(k, device=image.device, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-z.square() / (2 * sigma * sigma))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    # reflect padding requires dimension > pad; valid portraits satisfy this.
    if size <= k // 2:
        raise ValueError("face crop too small for feather kernel")
    a = F.conv2d(F.pad(restricted, (k // 2,) * 4, mode="reflect"), kernel).round() / 255
    alpha = torch.zeros((1, 1, h, w), device=image.device)
    sx, sy, ex, ey = max(0, xs), max(0, ys), min(w, xe), min(h, ye)
    alpha[:, :, sy:ey, sx:ex] = a[:, :, sy - ys : ey - ys, sx - xs : ex - xs]
    return PreparedBlend(image[None].float(), alpha, (x, y, x1, y1), disclosure)


def composite(prepared, faces):
    """NCHW BGR faces -> NHWC uint8 frames on the same device."""
    if faces.ndim != 4 or faces.shape[1] != 3 or faces.device != prepared.image.device:
        raise ValueError("faces must be NCHW BGR on the portrait device")
    x, y, x1, y1 = prepared.box
    resized = F.interpolate(
        faces.float(), (y1 - y, x1 - x), mode="bilinear", align_corners=False
    )
    canvas = prepared.image.expand(len(faces), -1, -1, -1).clone()
    canvas[:, :, y:y1, x:x1] = resized.round().clamp(0, 255)
    result = canvas * prepared.alpha + prepared.image * (1 - prepared.alpha)
    if prepared.disclosure is not None:
        result = result * (1 - prepared.disclosure) + 255 * prepared.disclosure
    return result.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
