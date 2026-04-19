# LatentSync lipsync microservice — CPU-only, dep-isolated.
#
# PR-LS-1a (this PR): scaffold only. No torch, no diffusers, no model code.
# The service boots, /health + /ready work, /lipsync returns 501. This
# gives us an image that rebuilds in seconds so the next PR can layer in
# the heavier ML stack without invalidating the network wiring.
#
# Staged exactly like lipsync-musetalk:
#   1a (this PR): apt deps + FastAPI scaffold + structured 501
#   1b (next):   torch CPU wheels + diffusers/transformers pins
#                + scripts/download_models.sh for LatentSync + SD 1.5 VAE
#   1c (later):  vendor inference, first runnable end-to-end frame
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is listed even though 1a doesn't use it — keeps the Compose wiring
# consistent with lipsync-musetalk and saves a rebuild layer later.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip wheel \
    && pip install "setuptools>=68,<81" \
    && pip install -e .

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
