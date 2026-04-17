"""Compatibility shims for torch 2.6+.

Imported for side effects by the service's package `__init__`. Do not rely
on anything defined here outside this module.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def _force_weights_only_false() -> None:
    """Override torch.load's default so it still loads legacy pickled .pth files.

    Torch 2.6 flipped ``torch.load(weights_only=...)`` default from False to
    True. Several of our weights predate that era — BiSeNet 79999_iter.pth
    (2019), the ResNet18 ImageNet checkpoint, face-alignment's internally
    fetched SFD detector — and fail with:

        Cannot use ``weights_only=True`` with files saved in the legacy .tar
        format.

    Every weight this service loads is either downloaded by
    ``download_models.sh`` (from HF, pytorch.org, or an upstream HF mirror)
    or fetched by face-alignment from its own host. In other words, the
    source is trusted. Flipping the default off is safe for this demo.
    """
    if getattr(torch.load, "__polyglot_patched__", False):
        return

    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    patched.__polyglot_patched__ = True  # type: ignore[attr-defined]
    torch.load = patched  # type: ignore[assignment]
    log.info("torch.load patched: weights_only defaults to False")


_force_weights_only_false()
