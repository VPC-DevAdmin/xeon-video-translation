"""Vendored CodeFormer face-restoration path.

See codeformer_arch.py and vqgan_arch.py for attribution.

Public surface:
    restore_frame(frame_bgr, kps, device) -> frame_bgr
    preload(device) -> bool
"""

from .runner import preload, restore_frame

__all__ = ["preload", "restore_frame"]
