# CPU-only image. Bigger than the runtime alone because torch+ctranslate2 are heavy,
# but no CUDA layers — works on any x86_64 host with AVX2.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is a hard runtime dependency for Stage 1 and Stage 6.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       "torch>=2.6" \
       "torchaudio>=2.6" \
    && pip install -e .

COPY app ./app
COPY tests ./tests

EXPOSE 8000

# Use uvicorn directly — no reload in container by default; bind-mount + override for dev.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
