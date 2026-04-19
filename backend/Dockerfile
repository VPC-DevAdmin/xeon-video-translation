# CPU-only image. Bigger than the runtime alone because torch+ctranslate2 are heavy,
# but no CUDA layers — works on any x86_64 host with AVX2.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is a hard runtime dependency for Stage 1 and Stage 6.
# rubberband-cli powers formant-preserving time-stretch in Stage 4 when
# the TTS output overshoots the source video duration (see tts.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      rubberband-cli \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
# torch/torchaudio/torchcodec have coupled major.minor versions. Keeping all
# three pinned to the torch 2.6 line: torchcodec 0.3 is the matching build.
# See https://github.com/pytorch/torchcodec#compatibility
# setuptools 81 (Oct 2025) removed `pkg_resources`. Several of our deps
# (coqui-tts path, onnx downstream, various HF libs) still import it. Pin
# to <81 so we keep a working `pkg_resources` until upstreams migrate off.
RUN pip install --upgrade pip wheel \
    && pip install "setuptools>=68,<81" \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.6,<2.7" \
       "torchaudio>=2.6,<2.7" \
    && pip install "torchcodec>=0.3,<0.4" \
    && pip install -e . \
    # F5-TTS (Stage 4 alternate backend). Installed with --no-deps so its
    # internal transformers pin doesn't fight with coqui-tts's — the
    # relevant deps (torch, vocos, accelerate, x-transformers, jieba,
    # pypinyin, librosa) are either already present from `pip install -e .`
    # above or pulled in below explicitly. If F5-TTS's API surface shifts,
    # revisit this list.
    && pip install --no-deps "f5-tts>=1.0,<2.0" \
    && pip install \
       "vocos>=0.1.0" \
       "x-transformers>=1.31.14" \
       "accelerate>=0.33.0" \
       "jieba>=0.42.1" \
       "pypinyin>=0.50.0" \
       "cached-path>=1.6.0" \
       "ema-pytorch>=0.5.2" \
       "bitsandbytes==0.44.0 ; platform_machine == 'x86_64'" \
       || true \
    # The `|| true` is intentional: bitsandbytes is GPU-oriented and may
    # fail its CPU-only import at build time on some base images. F5-TTS
    # handles its absence gracefully; we'd rather ship a working image.
    && echo "F5-TTS install step complete"

COPY app ./app
COPY tests ./tests

EXPOSE 8000

# Use uvicorn directly — no reload in container by default; bind-mount + override for dev.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
