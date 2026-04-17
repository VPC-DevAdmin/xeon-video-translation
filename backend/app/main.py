"""FastAPI entrypoint for the polyglot-demo backend."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import jobs as jobs_api
from .api import stream as stream_api
from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)

app = FastAPI(
    title="polyglot-demo",
    version="0.1.0",
    description="Open-source video translation demo (CPU-only build).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_api.router)
app.include_router(stream_api.router)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": "polyglot-demo",
        "version": app.version,
        "docs": "/docs",
        "milestones_implemented": ["M1", "M2"],
    }


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "whisper_model": settings.whisper_model,
        "translate_backend": settings.translate_backend,
        "watermark_enabled": settings.enable_watermark,
    }
