from PIL import Image
import numpy as np
import cv2
import copy


# Absolute cap on the Gaussian feather kernel. Larger kernels look smoother
# on big masks but over-blur small ones (mouth-only mask, narrow faces),
# hollowing out the opaque center and producing ghost overlays.
# 31 px = ~15 px feather radius — wide enough to hide hard seams, narrow
# enough that a ~30 px mask still reads as fully opaque at its center.
MAX_FEATHER_KERNEL = 31


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def face_large_crop(image, face_box, expand=1.5):
    """The expanded face crop `get_image` feeds to the parser, as PIL RGB.

    Exposed so callers can run BiSeNet on many crops in one batch
    (`FaceParsing.parse_batch`) and hand the masks back via
    `get_image(..., mask_image=...)`.
    """
    body = Image.fromarray(image[:, :, ::-1])
    crop_box, _ = get_crop_box(face_box, expand)
    return body.crop(crop_box)


def _crop_padded(image_bgr: np.ndarray, crop_box) -> np.ndarray:
    """`image[y_s:y_e, x_s:x_e]` with zero padding where the box leaves the
    frame — the same geometry PIL's `crop` produces for out-of-range boxes."""
    x_s, y_s, x_e, y_e = crop_box
    h, w = image_bgr.shape[:2]
    out = np.zeros((y_e - y_s, x_e - x_s, 3), dtype=image_bgr.dtype)
    sx0, sy0 = max(0, x_s), max(0, y_s)
    sx1, sy1 = min(w, x_e), min(h, y_e)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y_s:sy1 - y_s, sx0 - x_s:sx1 - x_s] = image_bgr[sy0:sy1, sx0:sx1]
    return out


def face_large_crop_rgb(image_bgr: np.ndarray, face_box, expand=1.5) -> np.ndarray:
    """numpy RGB version of `face_large_crop` for `FaceParsing.parse_batch_np`."""
    crop_box, _ = get_crop_box(face_box, expand)
    return np.ascontiguousarray(_crop_padded(image_bgr, crop_box)[:, :, ::-1])


def composite_np(
    image_bgr: np.ndarray,
    face_bgr: np.ndarray,
    face_box,
    mask512: np.ndarray,
    upper_boundary_ratio: float = 0.5,
    expand: float = 1.5,
    feather_ratio: float | None = None,
) -> np.ndarray:
    """`get_image` with a precomputed parse mask, done in numpy/cv2 on the
    crop region only.

    Same result as `get_image(..., mask_image=...)`: the face is pasted
    into its box inside the expanded crop, the parse mask is restricted
    to the face box, its top `upper_boundary_ratio` is blanked, it is
    Gaussian-feathered, and the crop is alpha-blended back. Unlike the
    PIL path this never converts the full frame, so it is ~10x cheaper
    per 1080p frame and releases the GIL for most of its work.
    """
    x, y, x1, y1 = face_box
    crop_box, _ = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    cw, ch = x_e - x_s, y_e - y_s

    face_large = _crop_padded(image_bgr, crop_box)
    fx0, fy0 = x - x_s, y - y_s
    fx1, fy1 = x1 - x_s, y1 - y_s
    face_large[fy0:fy1, fx0:fx1] = face_bgr[: fy1 - fy0, : fx1 - fx0]

    mask = cv2.resize(mask512, (cw, ch), interpolation=cv2.INTER_LINEAR)
    restricted = np.zeros((ch, cw), dtype=np.uint8)
    restricted[fy0:fy1, fx0:fx1] = mask[fy0:fy1, fx0:fx1]
    top = int(ch * upper_boundary_ratio)
    restricted[:top, :] = 0

    ratio = 0.05 if feather_ratio is None else feather_ratio
    computed = int(ratio * cw // 2) * 2 + 1
    k = max(3, min(MAX_FEATHER_KERNEL, computed))
    alpha = cv2.GaussianBlur(restricted, (k, k), 0).astype(np.float32)[:, :, None] / 255.0

    out = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    sx0, sy0 = max(0, x_s), max(0, y_s)
    sx1, sy1 = min(w, x_e), min(h, y_e)
    if sx1 > sx0 and sy1 > sy0:
        cx0, cy0 = sx0 - x_s, sy0 - y_s
        cx1, cy1 = cx0 + (sx1 - sx0), cy0 + (sy1 - sy0)
        a = alpha[cy0:cy1, cx0:cx1]
        region = out[sy0:sy1, sx0:sx1].astype(np.float32)
        fl = face_large[cy0:cy1, cx0:cx1].astype(np.float32)
        out[sy0:sy1, sx0:sx1] = (fl * a + region * (1.0 - a)).round().clip(0, 255).astype(np.uint8)
    return out


def get_image(
    image,
    face,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    mode="raw",
    fp=None,
    feather_ratio: float | None = None,
    mask_image=None,
):
    """Composite a predicted face back onto the original frame.

    Args:
        image: Full frame (numpy BGR).
        face: Predicted face crop (numpy BGR).
        face_box: (x1, y1, x2, y2) on the original frame.
        upper_boundary_ratio: Top fraction of the mask to *blank* — keeps
            the forehead/eyes/nose untouched. 0.5 = keep top half of the
            face-region mask from being active.
        expand: Crop-box expansion factor vs the face_box.
        mode: `face_seg` mode (raw | jaw | mouth | neck).
        fp: a `FaceParsing` instance.
        feather_ratio: Gaussian-blur kernel size as a fraction of the crop
            width. None → upstream default of 0.05 (~13 px on a 256-wide
            crop). Higher values (0.08–0.12) produce softer mask edges,
            hiding small frame-to-frame prediction differences at the
            expense of letting more "original" pixels through at the
            seam. Pair with `mode="mouth"` for the tightest, softest
            composite.
    """
    body = Image.fromarray(image[:, :, ::-1])
    face = Image.fromarray(face[:, :, ::-1])

    x, y, x1, y1 = face_box
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    # Mask from the face-parsing network, unless the caller precomputed it
    # (batched parse). Either way it is resized to the crop.
    if mask_image is None:
        mask_image = face_seg(face_large, mode=mode, fp=fp)
    else:
        mask_image = mask_image.resize(ori_shape)

    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new("L", ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # Blank the top `upper_boundary_ratio` of the mask so the forehead/eyes
    # keep their original pixels.
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new("L", ori_shape, 0)
    modified_mask_image.paste(
        mask_image.crop((0, top_boundary, width, height)),
        (0, top_boundary),
    )

    # Gaussian-feather the boundary. Odd-length kernel required by OpenCV.
    #
    # The ratio is applied to the crop width, but we cap the absolute size
    # at MAX_FEATHER_KERNEL so small masks (mouth mode, narrow faces) don't
    # get eroded to a ghost overlay. On a 700-px crop the uncapped 0.08
    # ratio lands at 57 px, far wider than a typical lips+teeth mask — the
    # center of the mouth ended up 50% transparent, which read as a ghost
    # mouth on top of the original.
    ratio = 0.05 if feather_ratio is None else feather_ratio
    computed = int(ratio * ori_shape[0] // 2) * 2 + 1
    blur_kernel_size = max(3, min(MAX_FEATHER_KERNEL, computed))
    mask_array = cv2.GaussianBlur(
        np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0
    )
    mask_image = Image.fromarray(mask_array)

    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)

    return body[:, :, ::-1]


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
