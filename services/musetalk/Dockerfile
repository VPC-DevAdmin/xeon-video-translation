# MuseTalk lipsync microservice — CPU-only, dep-isolated.
#
# Layered deliberately:
#   1. apt deps (ffmpeg, build tools for a couple of source wheels)
#   2. torch CPU wheels (heaviest single layer — cached across PRs)
#   3. project deps from pyproject.toml
#   4. application code (cheapest to rebuild)
#
# PR 1b adds the ML stack + preprocessing code. PR 1c flips inference on.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchaudio/torchcodec on the same 2.6 line as the main backend. The
# Intel OpenMP that ships with these wheels is what actually does the heavy
# lifting on Xeon; nothing more clever is needed for this milestone.
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.6,<2.7" \
       "torchaudio>=2.6,<2.7" \
    && pip install "torchcodec>=0.3,<0.4"

COPY pyproject.toml ./
RUN pip install -e .

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
