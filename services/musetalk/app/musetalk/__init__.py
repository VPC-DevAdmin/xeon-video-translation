"""Vendored + adapted MuseTalk inference code.

The bulk of this package is copied from https://github.com/TMElyralab/MuseTalk
(MIT-licensed code) with the following deliberate deviations:

- `musetalk/whisper/` — replaced with `audio_features.py` that runs Whisper
  via HuggingFace `transformers` and taps per-layer encoder hidden states.
  Avoids vendoring the whole openai/whisper fork.
- `musetalk/utils/preprocessing.py` — replaced with `face_tracking.py` that
  uses `face-alignment` (SFD + FAN) for landmarks, not mmpose/DWPose.
- Model paths / CUDA checks softened for CPU-only operation.

Weights are under separate licenses (CC-BY-NC 4.0 on the MuseTalk UNet) —
see docs/lipsync.md and docs/ethics.md.
"""

# Side-effect import: patches torch.load to tolerate legacy `.pth` files.
# Must run before any model loading inside this package.
from . import _compat  # noqa: F401
