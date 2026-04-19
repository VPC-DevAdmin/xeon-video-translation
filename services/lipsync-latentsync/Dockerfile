# LatentSync lipsync microservice — CPU-only, dep-isolated.
#
# Layered deliberately (largest-cached → most-frequently-edited):
#   1. apt deps (ffmpeg, build tools for a couple of source wheels)
#   2. torch 2.5.1 CPU wheels (heaviest single layer; cached across PRs)
#   3. project deps from pyproject.toml (diffusers, mediapipe, ...)
#   4. application code (cheapest to rebuild)
#
# Staging:
#   1a (shipped):  FastAPI scaffold only; no torch; /lipsync returns 501.
#   1b (this PR):  Full ML stack + real download_models.sh. /ready turns
#                  green after `make rebuild`; /weights turns green after
#                  `make models-latentsync`. /lipsync still returns 501.
#   1c (next):     Vendor inference, flip INFERENCE_IMPLEMENTED = True.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      curl \
      git \
      # build-essential: several wheels on this stack (python_speech_features,
      # face-alignment, sometimes decord) compile small C extensions from
      # source when no pre-built wheel matches Python 3.11 + CPU.
      build-essential \
      # mediapipe links against libGL for some visualization paths; keep
      # the minimal GL stubs even though we never open a display.
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision pinned to match LatentSync upstream (requirements.txt
# has `torch==2.5.1` / `torchvision==0.20.1`). Upstream was developed
# against CUDA builds; we install the CPU wheels from the PyTorch index.
# Nothing in the inference code should depend on the CUDA symbol set —
# if it does, that's a PR-LS-1c bug to chase down, not something to
# paper over here.
#
# setuptools 81 dropped pkg_resources; insightface + onnx + various HF
# paths still import it. Pin to <81 like the other services.
RUN pip install --upgrade pip wheel \
    && pip install "setuptools>=68,<81" \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       "torch==2.5.1" \
       "torchvision==0.20.1"

COPY pyproject.toml ./
RUN pip install -e .

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
