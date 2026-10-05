"""Head-pose and gaze edit of a portrait with LivePortrait, done once per persona.

FlashHead animates a portrait from audio only and has no pose control, so the
"looking down at a tablet" look of the assistant's working phase comes from a
second portrait: the same person with the head pitched down and turned a little
and the eyes lowered. LivePortrait's single-image retargeting does that edit in
about 0.1 s on the GPU; FlashHead then animates the posed portrait like any other.

LIVEPORTRAIT_SOURCE   checkout of github.com/KwaiVGI/LivePortrait with pretrained_weights/
LIVEPORTRAIT_SHIM     directory holding a gradio.py stand-in (the pipeline imports gradio
                      only for its Error/Info helpers; installing gradio would move the
                      service's FastAPI and pydantic versions)
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

SOURCE = Path(os.environ.get("LIVEPORTRAIT_SOURCE", "/experiment/LivePortrait"))
SHIM = Path(os.environ.get("LIVEPORTRAIT_SHIM", "/experiment/liveportrait_shim"))
_pipeline = None
_lock = threading.Lock()


def available() -> bool:
    return (SOURCE / "src").is_dir() and (SOURCE / "pretrained_weights" / "liveportrait" / "base_models").is_dir()


def _load():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        try:
            import gradio  # noqa: F401
        except ImportError:
            if SHIM.is_dir() and str(SHIM) not in sys.path:
                sys.path.insert(0, str(SHIM))
        from src.config.argument_config import ArgumentConfig
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.gradio_pipeline import GradioPipeline

        args = ArgumentConfig()
        inference = InferenceConfig(**{k: v for k, v in args.__dict__.items() if hasattr(InferenceConfig, k)})
        crop = CropConfig(**{k: v for k, v in args.__dict__.items() if hasattr(CropConfig, k)})
        _pipeline = GradioPipeline(inference, crop, args)
        return _pipeline


def pose_portrait(image_bgr: np.ndarray, pitch: float, yaw: float, roll: float = 0.0,
                  eyes_x: float = 0.0, eyes_y: float = 0.0, scale: float = 2.3) -> np.ndarray:
    """The portrait with its head rotated by (pitch, yaw, roll) degrees and the gaze moved
    by (eyes_x, eyes_y); positive pitch looks down, negative eyes_y lowers the gaze. Full-frame result, the face
    region pasted back into the original image."""
    return pose_sequence(image_bgr, pitch, yaw, roll, eyes_x, eyes_y, steps=1, scale=scale)[-1]


def pose_sequence(image_bgr: np.ndarray, pitch: float, yaw: float, roll: float = 0.0, eyes_x: float = 0.0,
                  eyes_y: float = 0.0, steps: int = 1, scale: float = 2.3) -> list[np.ndarray]:
    """`steps` frames turning the head from the portrait's pose to the target: an eased
    path, the eyes leading the head a little, as a person glancing down at a tablet does.
    One frame is the posed portrait itself."""
    import cv2

    pipeline = _load()
    frames = []
    with tempfile.TemporaryDirectory(prefix="pose-") as directory:
        path = str(Path(directory) / "portrait.png")
        cv2.imwrite(path, image_bgr)
        eye_ratio, lip_ratio = pipeline.init_retargeting_image(scale, 0, 0, path)
        for index in range(1, max(1, int(steps)) + 1):
            t = index / max(1, int(steps))
            head = t * t * (3 - 2 * t)                              # smoothstep
            eyes = min(1.0, t * 1.4) ** 2 * (3 - 2 * min(1.0, t * 1.4))  # the gaze arrives first
            _, blended = pipeline.execute_image_retargeting(
                eye_ratio, lip_ratio, float(pitch) * head, float(yaw) * head, float(roll) * head, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(eyes_x) * eyes, float(eyes_y) * eyes, path, scale, True, True)
            frames.append(cv2.cvtColor(np.asarray(blended), cv2.COLOR_RGB2BGR))
    return frames


def reading_loop(image_bgr: np.ndarray, pitch: float, yaw: float, roll: float = 0.0, eyes_x: float = 0.0,
                 eyes_y: float = 0.0, frames: int = 150, fps: int = 25, scale: float = 2.3) -> list[np.ndarray]:
    """A periodic loop of the posed portrait reading: the eyes scan across the lines, the
    head drifts slightly, and the eyes blink twice. Every motion is a whole number of
    cycles over the loop, so the last frame leads straight back to the first; frame 0 is
    the posed portrait at rest (the pose the turn footage arrives at)."""
    import math
    import cv2

    pipeline = _load()
    seconds = frames / fps
    out = []
    with tempfile.TemporaryDirectory(prefix="read-") as directory:
        path = str(Path(directory) / "portrait.png")
        cv2.imwrite(path, image_bgr)
        eye_ratio, lip_ratio = pipeline.init_retargeting_image(scale, 0, 0, path)
        for i in range(frames):
            t = i / fps
            scan = math.sin(2 * math.pi * t / (seconds / 3))                    # three sweeps across the page
            ex = eyes_x + 6.0 * scan
            ey = eyes_y + 1.5 * math.sin(2 * math.pi * t / (seconds / 2))       # down a line, back up
            p = pitch + 1.2 * math.sin(2 * math.pi * t / seconds)
            y = yaw + 1.0 * math.sin(2 * math.pi * t / (seconds / 2) + 1.0)
            blink = 0.0
            for at in (seconds * 0.35, seconds * 0.78):                           # two blinks, each 4 frames
                phase = (t - at) * fps
                if 0 <= phase < 4:
                    blink = 1.0 if 1 <= phase < 3 else 0.6
            ratio = eye_ratio * (1.0 - 0.9 * blink)
            _, blended = pipeline.execute_image_retargeting(
                ratio, lip_ratio, float(p), float(y), float(roll), 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(ex), float(ey), path, scale, True, True)
            out.append(cv2.cvtColor(np.asarray(blended), cv2.COLOR_RGB2BGR))
    return out


THINK_RETURN_AT = 0.70      # fraction of the thinking loop after which the face is back at the camera


def thinking_loop(image_bgr: np.ndarray, pitch: float, yaw: float, roll: float = 0.0, eyes_x: float = 0.0,
                  eyes_y: float = 0.0, frames: int = 150, fps: int = 25, scale: float = 2.3) -> list[np.ndarray]:
    """A periodic loop of the portrait thinking: the gaze and head ease away to the
    (pitch, yaw, roll, eyes) offset, wander there, come back to the camera with a small
    nod, and rest at the camera until the loop restarts. Frame 0 and the last frame are
    the portrait at rest, so the loop joins the rest pose with a cut. The face is back
    at the camera from THINK_RETURN_AT of the loop on (where a spoken beat fits)."""
    import math
    import cv2

    pipeline = _load()
    T = frames / fps
    leave_end, hold_end, back_end = 0.1 * T, 0.6 * T, THINK_RETURN_AT * T
    nod_start, nod_end = back_end + 0.05 * T, back_end + 0.17 * T

    def smooth(x):
        x = min(1.0, max(0.0, x))
        return x * x * (3 - 2 * x)

    out = []
    with tempfile.TemporaryDirectory(prefix="think-") as directory:
        path = str(Path(directory) / "portrait.png")
        cv2.imwrite(path, image_bgr)
        eye_ratio, lip_ratio = pipeline.init_retargeting_image(scale, 0, 0, path)
        for i in range(frames):
            t = i / fps
            if t < leave_end:
                a = smooth(t / leave_end)
            elif t < hold_end:
                a = 1.0
            elif t < back_end:
                a = 1.0 - smooth((t - hold_end) / (back_end - hold_end))
            else:
                a = 0.0
            wander = 1.0 if leave_end <= t < hold_end else 0.0
            p = pitch * a + wander * 0.8 * math.sin(2 * math.pi * (t - leave_end) / 2.3)
            y = yaw * a + wander * 1.5 * math.sin(2 * math.pi * (t - leave_end) / 3.1)
            r = roll * a
            ex = eyes_x * a + wander * 2.5 * math.sin(2 * math.pi * t / 1.4)
            ey = eyes_y * a + wander * 1.0 * math.sin(2 * math.pi * t / 1.9 + 0.7)
            if nod_start <= t < nod_end:
                p += 2.5 * math.sin(math.pi * (t - nod_start) / (nod_end - nod_start))   # a small nod, down and back
            blink = 0.0
            for at in (0.33 * T, 0.9 * T):
                phase = (t - at) * fps
                if 0 <= phase < 4:
                    blink = 1.0 if 1 <= phase < 3 else 0.6
            ratio = eye_ratio * (1.0 - 0.9 * blink)
            _, blended = pipeline.execute_image_retargeting(
                ratio, lip_ratio, float(p), float(y), float(r), 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(ex), float(ey), path, scale, True, True)
            out.append(cv2.cvtColor(np.asarray(blended), cv2.COLOR_RGB2BGR))
    return out
