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

_warmup_state: dict = {"status": "disabled" if not settings.warmup_models else "pending"}


def _warmup() -> None:
    """Load the per-stage model singletons so the first job starts hot."""
    import time

    from .pipeline import transcribe, translate, tts

    log = logging.getLogger("warmup")
    steps = [
        ("whisper", transcribe._get_model),
        ("nllb", translate._get_nllb_pipeline) if settings.translate_backend == "nllb" else None,
        ("xtts", tts._get_xtts) if settings.tts_backend == "xtts" else ("f5tts", tts._get_f5tts),
    ]
    for step in steps:
        if step is None:
            continue
        name, fn = step
        t0 = time.perf_counter()
        try:
            fn()
            _warmup_state[name] = round(time.perf_counter() - t0, 1)
            log.info("warmup: %s loaded in %.1fs", name, _warmup_state[name])
        except Exception as e:  # keep serving; the stage will retry lazily
            _warmup_state[name] = f"failed: {e}"
            log.warning("warmup: %s failed: %s", name, e)
    _warmup_state["status"] = "done"


@app.on_event("startup")
async def _startup() -> None:
    if settings.warmup_models:
        import asyncio

        _warmup_state["status"] = "running"
        asyncio.get_running_loop().run_in_executor(None, _warmup)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": "polyglot-demo",
        "version": app.version,
        "docs": "/docs",
        "milestones_implemented": ["M1", "M2", "M3", "M4"],
    }


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "whisper_model": settings.whisper_model,
        "translate_backend": settings.translate_backend,
        "watermark_enabled": settings.enable_watermark,
        "device": settings.resolved_device,
        "warmup": _warmup_state,
    }
