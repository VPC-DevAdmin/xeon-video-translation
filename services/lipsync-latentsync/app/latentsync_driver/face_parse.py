"""Occlusion-aware paste-back for LatentSync.

LatentSync pastes the whole generated face crop back wherever the landmark
track says the face is. When a hand or an object passes in front of the face
the track either keeps the last good landmarks (detection lost) or still finds
the eyes and nose (mouth covered), and the generated mouth lands on the object
(seen on 5 Oct 2026: a chocolate box held up in front of the speaker).

Two layers fix this:

* **Pixel mask.** BiSeNet face parsing on the *source* face crop labels every
  pixel (skin, nose, lips, hair, background...). Only face pixels receive the
  generated face; an occluder keeps the source pixels. Because the generated
  crop equals the source outside the mouth region, excluding pixels is harmless
  anywhere else, so the mask is dilated, feathered and averaged over three
  frames to stay stable rather than tight.
* **Frame gate.** Frames where detection failed (and a margin around them)
  are not pasted at all, with a short alpha ramp in and out, since the carried
  landmarks say nothing about where the face is.

* **Temporal occluder mask.** A hand, or a pale box, is skin to the parser.
  In the aligned crop the face itself barely moves, so each frame is compared
  with the median of nearby frames *before* it and *after* it: a pixel that
  differs from both is something passing in front of the face (a pose change
  persists into the future, a moving mouth is a compact blob that never
  reaches the crop border). Detected at 128 px, kept only for large blobs
  that touch the border, dilated, and spread over neighbouring frames.
* **Confidence fade.** A face half covered is still detected, but with a much
  lower score; the paste fades between `CONF_LOW` and `CONF_HIGH`.
* **Hand mask.** A finger resting on the lips for half a second is skin to the
  parser and part of the temporal median, so hands are found directly with
  MediaPipe Hands on the source frame and drawn into the crop through the same
  affine as the paste (palm polygon plus finger bones at finger width).
* **Closed mouth in silence.** LatentSync is given the current frame as its
  reference, so under silence it copies that frame's mouth: after the
  translated speech ends the speaker keeps mouthing the original language.
  Silent frames (from the translated audio) get the clip's most closed-mouth
  frame as the UNet reference instead; the masked frame still gives the pose
  and the paste-back still uses the real frame.
* **Covered-mouth gate.** A pale, blurred object over the mouth is only partly
  caught by the temporal mask (a half-painted mouth would look worse than
  either extreme), so when the occluder mask covers more than `MOUTH_COVERED`
  of the lower-face band the whole frame is treated as covered and the source
  mouth is shown for those few frames.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# BiSeNet (CelebAMask-HQ) classes that are the face itself. Hair, hat, neck,
# cloth, ears, earrings and background are left to the source frame.
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
DEFAULT_WEIGHTS_DIR = "/models/musetalk/face-parse-bisent"


def enabled() -> bool:
    return os.environ.get("LATENTSYNC_OCCLUSION_MASK", "1") == "1"


# Below CONF_LOW the face is treated as covered; clean handheld frames sit at
# 0.75-0.9 on buffalo_l, half-covered ones at 0.5-0.72.
CONF_LOW, CONF_HIGH = 0.55, 0.72
# Lower-face band of the canonical 512 crop (where LatentSync changes pixels)
# and the fraction of it an occluder may cover before the frame is skipped.
MOUTH_BAND = (slice(300, 430), slice(130, 390))
MOUTH_COVERED = 0.30  # clean frames on handheld footage reach ~0.1, boxes/hands over the mouth 0.4+


def occlusion_alpha(visible, margin: int = 1, ramp: int = 3) -> np.ndarray:
    """Per-frame paste weight from per-frame detection.

    `visible` is a bool array (face detected) or the detector's confidence per
    frame (0.0 = no face). Frames without a detected face, and `margin` frames
    either side of them, get 0; the weight then rises linearly to 1 over
    `ramp` frames so the mouth neither pops in nor out. With confidences, the
    weight is also faded between CONF_LOW and CONF_HIGH (minimum over the
    frame and its neighbours), since a half-covered face still detects, but
    poorly."""
    values = np.asarray(visible)
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if values.dtype == bool:
        visible = values
        confidence = None
    else:
        confidence = values.astype(np.float32)
        visible = confidence > 0
    blocked = ~visible
    if margin > 0 and blocked.any():
        grown = blocked.copy()
        for shift in range(1, margin + 1):
            grown[shift:] |= blocked[:-shift]
            grown[:-shift] |= blocked[shift:]
        blocked = grown
    if not blocked.any():
        return _confidence_fade(np.ones(n, dtype=np.float32), confidence)
    # Distance (in frames) from each frame to the nearest blocked frame.
    distance = np.full(n, n, dtype=np.int64)
    last = None
    for i in range(n):
        if blocked[i]:
            last = i
        if last is not None:
            distance[i] = i - last
    last = None
    for i in range(n - 1, -1, -1):
        if blocked[i]:
            last = i
        if last is not None:
            distance[i] = min(distance[i], last - i)
    alpha = np.clip(distance / float(max(ramp, 1)), 0.0, 1.0).astype(np.float32)
    return _confidence_fade(alpha, confidence)


def _confidence_fade(alpha: np.ndarray, confidence) -> np.ndarray:
    """Binary: a partial weight would dissolve two different mouths into a
    smudge (seen at a gate exit on 5 Oct 2026), so a frame is either pasted
    or not. Below the midpoint of CONF_LOW..CONF_HIGH it is not."""
    if confidence is None:
        return alpha
    fade = (confidence >= (CONF_LOW + CONF_HIGH) / 2).astype(np.float32)
    fade = np.where(confidence > 0, fade, 1.0).astype(np.float32)  # gaps are already 0 in alpha
    if len(fade) > 1:  # minimum over the frame and its neighbours
        padded = np.concatenate([fade[:1], fade, fade[-1:]])
        fade = np.minimum(np.minimum(padded[:-2], padded[1:-1]), padded[2:])
    return np.minimum(alpha, fade).astype(np.float32)


def feather(mask, size: int = 15):
    """(N,1,H,W) float mask -> Gaussian-blurred copy (separable, replicate pad)."""
    import torch
    import torch.nn.functional as F

    k = size if size % 2 else size + 1
    if k <= 1:
        return mask
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, k)
    out = F.conv2d(F.pad(mask, (k // 2, k // 2, 0, 0), mode="replicate"), g)
    return F.conv2d(F.pad(out, (0, 0, k // 2, k // 2), mode="replicate"), g.transpose(2, 3)).clamp(0.0, 1.0)


def face_mask_from_parsing(parsing, dilate: int = 9, feather: int = 15):
    """(N,H,W) class map -> (N,1,H,W) float mask in [0,1]: 1 on face pixels,
    dilated by `dilate` px, feathered with a `feather` px Gaussian, then
    averaged over three frames. Torch-only so it runs where the parse ran."""
    import torch
    import torch.nn.functional as F

    classes = torch.tensor(FACE_CLASSES, device=parsing.device)
    mask = torch.isin(parsing, classes).to(torch.float32).unsqueeze(1)
    if dilate > 1:
        k = dilate if dilate % 2 else dilate + 1
        mask = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    if feather > 1:
        k = feather if feather % 2 else feather + 1
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        x = torch.arange(k, device=mask.device, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g = (g / g.sum()).view(1, 1, 1, k)
        mask = F.conv2d(F.pad(mask, (k // 2, k // 2, 0, 0), mode="replicate"), g)
        mask = F.conv2d(F.pad(mask, (0, 0, k // 2, k // 2), mode="replicate"), g.transpose(2, 3))
    if len(mask) >= 3:
        padded = torch.cat([mask[:1], mask, mask[-1:]], dim=0)
        mask = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    return mask.clamp_(0.0, 1.0)


def occluder_masks(crops, visible, size: int = 128, window: int = 30, threshold: float = 22.0,
                   min_area: float = 0.02, dilate: int = 3, min_reference: int = 5):
    """(N,3,H,W) uint8 aligned crops + per-frame visibility -> (N,1,size,size)
    float occluder mask (1 = something in front of the face).

    Each frame is compared with the median of up to `window` visible frames
    before it and after it (brightness-matched); the deviation is the smaller
    of the two, so only what differs from both past and future counts. Blobs
    that cover at least `min_area` of the crop and touch its border are kept,
    dilated by `dilate` px, and spread to the neighbouring frames."""
    import cv2
    import torch
    import torch.nn.functional as F

    n = len(crops)
    visible = np.asarray(visible)
    visible = visible if visible.dtype == bool else visible > 0
    out = torch.zeros((n, 1, size, size), dtype=torch.float32)
    if n < min_reference + 1 or int(visible.sum()) < min_reference:
        return out
    device = crops.device
    small = F.interpolate(crops.float(), size=(size, size), mode="area")
    small = F.avg_pool2d(small, 3, stride=1, padding=1)
    lum = small.mean(1)  # (N,size,size)
    centre = (slice(size // 4, size * 3 // 4), slice(size // 4, size * 3 // 4))
    visible_idx = np.flatnonzero(visible)

    def reference(lo, hi):
        idx = visible_idx[(visible_idx >= lo) & (visible_idx < hi)]
        if len(idx) < min_reference:
            return None
        ref = small[torch.as_tensor(idx, device=device)].median(dim=0).values
        return ref, ref.mean(0)[centre[0], centre[1]]

    def deviation(i, ref):
        if ref is None:
            return None
        pixels, ref_lum = ref
        # Brightness match on the median pixel ratio over the centre of the
        # crop, using only pixels that roughly agree with the reference so a
        # large occluder cannot drag the gain and flag the whole face.
        current = lum[i][centre[0], centre[1]].clamp(min=1.0)
        agree = (current - ref_lum).abs() < 2.0 * threshold
        ratio = (ref_lum / current)[agree] if int(agree.sum()) >= 16 else (ref_lum / current)
        gain = ratio.flatten().median().clamp(0.7, 1.4)
        return (small[i] * gain - pixels).abs().max(0).values

    kernel = np.ones((dilate, dilate), np.uint8) if dilate > 1 else None
    masks = np.zeros((n, size, size), np.uint8)
    for i in range(n):
        past = deviation(i, reference(i - window, i))
        future = deviation(i, reference(i + 1, i + 1 + window))
        if past is None and future is None:
            continue
        dev = past if future is None else future if past is None else torch.minimum(past, future)
        m = (dev > threshold).to(torch.uint8).cpu().numpy()
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for j in range(1, count):
            x, y, w, h, area = stats[j]
            # the 3x3 blur pulls a blob one pixel off the border
            touches = x <= 1 or y <= 1 or x + w >= size - 1 or y + h >= size - 1
            if area >= min_area * size * size and touches:
                keep[labels == j] = 1
        masks[i] = cv2.dilate(keep, kernel) if kernel is not None else keep
    spread = masks.copy()
    spread[1:] |= masks[:-1]
    spread[:-1] |= masks[1:]
    return torch.from_numpy(spread.astype(np.float32)).unsqueeze(1)


def ranges(indices) -> str:
    """Compact "3-7, 12, 20-22" form of a sorted index list, for log lines."""
    indices = [int(i) for i in indices]
    if not indices:
        return ""
    out, start, prev = [], indices[0], indices[0]
    for i in indices[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
        if i is not None:
            start = prev = i
    return ", ".join(out)


# MediaPipe hand landmark topology (21 points): palm corners and finger bones.
HAND_PALM = (0, 1, 2, 5, 9, 13, 17)
HAND_BONES = ((1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
              (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 1), (0, 5), (0, 17))


def hand_mask_from_landmarks(points, size: int = 512, grow: float = 0.36, min_width: int = 8, max_width: int = 40,
                             pad: int = 3):
    """(21,2) landmark pixels in crop space -> (size,size) uint8 mask: the palm
    polygon plus every finger bone drawn `grow` x palm width thick (a finger is
    about a third of the palm width; capped, since a hand near the camera is
    larger than the face), grown by `pad` px. Hand-shaped, unlike a convex
    hull, so the mouth beside a finger is still pasted. The mask is used hard:
    LatentSync repaints the finger inside its region, so the cut has to lie on
    the finger's own edge, where a hard edge is invisible; a wide or feathered
    mask blends two different mouths at the corner."""
    import cv2

    pts = np.asarray(points, dtype=np.float32)
    mask = np.zeros((size, size), np.uint8)
    palm_width = float(np.linalg.norm(pts[5] - pts[17]))
    width = int(min(max_width, max(min_width, round(palm_width * grow))))
    palm = cv2.convexHull(pts[list(HAND_PALM)].astype(np.int32))
    cv2.fillConvexPoly(mask, palm, 1)
    for a, b in HAND_BONES:
        cv2.line(mask, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), 1, width)
    if pad > 0:
        k = 2 * pad + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def _affine_2x3(matrix) -> np.ndarray:
    try:
        import torch

        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(matrix, dtype=np.float64).reshape(2, 3)


def hand_masks(frames, affines, face_size: tuple[int, int], size: int = 512, detect_width: int = 540,
               min_confidence: float = 0.4):
    """(N,H,W,3) uint8 RGB source frames + per-frame paste affines (source px ->
    face_size crop) -> (N,1,size,size) float masks of hands in crop space, or
    None when MediaPipe is unavailable. Detections are spread to the neighbour
    frames to bridge single-frame misses."""
    try:
        import cv2
        import mediapipe as mp
        import torch
    except ImportError as exc:
        log.warning("hand masks unavailable: %s", exc)
        return None
    n = len(frames)
    if n == 0:
        return None
    height, width = frames[0].shape[:2]
    scale = detect_width / float(width)
    small_size = (detect_width, max(1, int(round(height * scale))))
    crop_w, crop_h = face_size
    sx, sy = size / float(crop_w), size / float(crop_h)
    masks = np.zeros((n, size, size), np.uint8)
    found = 0
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                                  min_detection_confidence=min_confidence,
                                  min_tracking_confidence=min_confidence) as hands:
        for i in range(n):
            result = hands.process(cv2.resize(np.ascontiguousarray(frames[i]), small_size, interpolation=cv2.INTER_AREA))
            if not result.multi_hand_landmarks:
                continue
            affine = _affine_2x3(affines[i])
            for hand in result.multi_hand_landmarks:
                pts = np.array([[l.x * width, l.y * height, 1.0] for l in hand.landmark])  # source px
                crop_pts = pts @ affine.T  # (21,2) in face_size crop
                crop_pts[:, 0] *= sx
                crop_pts[:, 1] *= sy
                if crop_pts[:, 0].max() < -size or crop_pts[:, 0].min() > 2 * size or \
                        crop_pts[:, 1].max() < -size or crop_pts[:, 1].min() > 2 * size:
                    continue  # nowhere near the face
                masks[i] |= hand_mask_from_landmarks(crop_pts, size)
            found += int(masks[i].any())
    spread = masks.copy()
    spread[1:] |= masks[:-1]
    spread[:-1] |= masks[1:]
    log.info("hand masks: hands near the face on %d of %d frames", found, n)
    return torch.from_numpy(spread.astype(np.float32)).unsqueeze(1)


def silent_frames(audio, sample_rate: int, fps: float, frames: int, threshold: float = 0.01,
                  min_run: int = 10, lead: int = 3) -> np.ndarray:
    """(frames,) bool: frames inside a silent run of at least `min_run` frames
    (RMS over the frame and its neighbours below `threshold`, audio in [-1,1]),
    minus the first `lead` frames of each run so the mouth closes naturally
    rather than snapping. Audio shorter than the video counts as silent."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    per = sample_rate / float(fps)
    rms = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        lo, hi = int((i - 1) * per), int((i + 2) * per)
        seg = audio[max(0, lo):max(0, hi)]
        rms[i] = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
    quiet = rms < threshold
    out = np.zeros(frames, dtype=bool)
    i = 0
    while i < frames:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < frames and quiet[j]:
            j += 1
        if j - i >= min_run:
            out[i + lead:j] = True
        i = j
    return out


def pick_closed_mouth(mouth_open, alpha, covered=None):
    """Index of the frame to use as the silent-mouth reference: the smallest
    mouth-interior fraction among frames whose face is fully visible (alpha 1,
    nothing over the mouth). None when no frame qualifies."""
    mouth_open = np.asarray(mouth_open, dtype=np.float32)
    ok = np.asarray(alpha) >= 1.0
    if covered is not None:
        ok &= np.asarray(covered) <= 0.0
    if not ok.any():
        return None
    candidates = np.flatnonzero(ok)
    return int(candidates[np.argmin(mouth_open[candidates])])


class FaceParser:
    """Lazy BiSeNet wrapper. `masks(crops)` takes (N,3,H,W) uint8 RGB face crops
    (LatentSync's canonical crops) and returns (N,1,out_h,out_w) float masks on
    the same device, or None when parsing is disabled or weights are missing."""

    def __init__(self, device, weights_dir: str | None = None):
        self.device = device
        self.weights_dir = Path(weights_dir or os.environ.get("LATENTSYNC_FACE_PARSE_DIR", DEFAULT_WEIGHTS_DIR))
        self._net = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        import torch

        from .face_parsing import BiSeNet

        weights = self.weights_dir / "79999_iter.pth"
        resnet = self.weights_dir / "resnet18-5c106cde.pth"
        if not weights.exists() or not resnet.exists():
            raise FileNotFoundError(f"face parsing weights missing under {self.weights_dir}")
        net = BiSeNet(resnet_path=str(resnet), n_classes=19)
        net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False))
        net.to(self.device).eval()
        return net

    def available(self) -> bool:
        if not enabled() or self._failed:
            return False
        if self._net is None:
            with self._lock:
                if self._net is None and not self._failed:
                    try:
                        self._net = self._load()
                        log.info("face parser ready (%s)", self.weights_dir)
                    except Exception as exc:
                        self._failed = True
                        log.warning("face parsing unavailable, occluders will not be masked: %s", exc)
        return self._net is not None

    def masks(self, crops, out_size: tuple[int, int], batch_size: int = 32, visible=None,
              frames=None, affines=None):
        """Returns `(masks, covered, mouth_open)`: (N,1,h,w) paste masks for
        `out_size` = (height, width) of the warp crop, the per-frame fraction of
        the lower-face band hidden by an occluder (zeros without `visible`), and
        the per-frame fraction of the band parsed as mouth interior. With
        `visible` (per-frame detection / confidence) the temporal occluder
        mask is subtracted as well; with `frames` (source RGB) and `affines`
        the MediaPipe hand mask too. `(None, None, None)` when parsing is unavailable."""
        if not self.available():
            return None, None, None
        import torch
        import torch.nn.functional as F

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        parsed = []
        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                x = crops[start:start + batch_size].to(self.device, non_blocking=True).float().div_(255.0)
                if x.shape[-2:] != (512, 512):
                    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
                x = (x - mean) / std
                parsed.append(self._net(x)[0].argmax(1).to(torch.uint8))
            parsing = torch.cat(parsed, dim=0)
            mask = face_mask_from_parsing(parsing)
            band = parsing[:, MOUTH_BAND[0], MOUTH_BAND[1]]
            mouth_open = ((band == 13).float().sum((1, 2)) / float(band.shape[1] * band.shape[2])).cpu()
            covered = torch.zeros(len(mask), dtype=torch.float32)
            occluders = None  # hard, for coverage
            cut = None        # what actually cuts the paste mask
            if visible is not None and os.environ.get("LATENTSYNC_OCCLUDER_MASK", "1") == "1":
                temporal = occluder_masks(crops.to(self.device, non_blocking=True), visible).to(mask.device)
                temporal = F.interpolate(temporal, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                occluders = temporal
                # The temporal mask is blocky (128 px); a small feather hides the blocks.
                cut = feather(temporal, 7)
            if frames is not None and affines is not None and os.environ.get("LATENTSYNC_HAND_MASK", "1") == "1":
                hands = hand_masks(frames, affines, face_size=(out_size[1], out_size[0]), size=mask.shape[-1])
                if hands is not None:
                    hands = hands.to(mask.device)
                    occluders = hands if occluders is None else torch.maximum(occluders, hands)
                    cut = hands if cut is None else torch.maximum(cut, hands)  # hand edges stay hard
            if occluders is not None:
                band_face = (mask > 0.5).float()[..., MOUTH_BAND[0], MOUTH_BAND[1]]
                band_occ = (occluders > 0.5).float()[..., MOUTH_BAND[0], MOUTH_BAND[1]]
                covered = ((band_face * band_occ).sum((1, 2, 3)) / band_face.sum((1, 2, 3)).clamp(min=1.0)).cpu()
                mask = mask * (1.0 - cut)
            mask = F.interpolate(mask, size=out_size, mode="bilinear", align_corners=False)
        return mask, covered, mouth_open
