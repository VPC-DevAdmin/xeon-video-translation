# MuseTalk lipsync microservice — CPU-only, dep-isolated.
#
# PR 1a ships only the FastAPI scaffold. PR 1b will add torch/diffusers/etc.
# Keeping this layered deliberately so each PR's rebuild cost is bounded.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg will be needed for PR 1c (muxing the inference output). Install now
# so the 1c rebuild is cheap.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
