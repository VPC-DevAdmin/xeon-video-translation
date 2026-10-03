"""FastAPI entrypoint for the polyglot-demo backend."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import speech as speech_api, diagnostics as diagnostics_api
from .api import studio as studio_api
from .api import avatar as avatar_api
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
    description="GPU video translation and live voice avatar demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def identity(request: Request, call_next):
    from .security import authenticate, principal
    from fastapi.responses import JSONResponse

    if request.url.path == "/health":
        return await call_next(request)
    try:
        owner = authenticate(request.headers)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    token = principal.set(owner)
    try:
        return await call_next(request)
    finally:
        principal.reset(token)


@app.get("/auth/whoami")
async def whoami():
    from .security import principal

    return {"owner": principal.get()}


app.include_router(jobs_api.router)
app.include_router(studio_api.router)
app.include_router(speech_api.router)
app.include_router(diagnostics_api.router)
app.include_router(avatar_api.router)
app.include_router(stream_api.router)

_maintenance_task = None
_warmup_task = None
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
    global _warmup_task, _maintenance_task
    from .pipeline.orchestrator import recover_jobs, speech_lock, blocking_call

    if settings.recover_jobs:
        from .operations import acquire_dispatcher, expire_jobs, reconcile_usage

        acquire_dispatcher()
        try:
            await blocking_call(reconcile_usage)
            await blocking_call(expire_jobs)
            await recover_jobs()
        except BaseException:
            from .operations import release_dispatcher

            release_dispatcher()
            raise
        import asyncio

        async def maintain():
            while True:
                await asyncio.sleep(60)
                await blocking_call(lambda: reconcile_usage(active_only=True))
                await blocking_call(expire_jobs)

        _maintenance_task = asyncio.create_task(maintain())
    if settings.warmup_models:
        import asyncio

        _warmup_state["status"] = "running"

        async def warmup():
            async with speech_lock():
                await blocking_call(_warmup)

        _warmup_task = asyncio.create_task(warmup())


@app.on_event("shutdown")
async def _shutdown():
    global _warmup_task, _maintenance_task
    import asyncio
    from .pipeline import orchestrator

    orchestrator._shutting_down = True

    tasks = [
        t
        for t in (*orchestrator._tasks.values(), _warmup_task, _maintenance_task)
        if t and not t.done()
    ]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _warmup_task = _maintenance_task = None
    for queue in orchestrator._queues.values():
        queue.put_nowait({"event": "stream_end", "data": {}})
    orchestrator._queues.clear()
    orchestrator._jobs.clear()
    from .operations import release_dispatcher

    release_dispatcher()


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
