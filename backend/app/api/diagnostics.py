import importlib.util
import shutil
import subprocess
from fastapi import APIRouter
from .. import operations, state_store
from ..security import principal
from ..config import settings
from ..pipeline.orchestrator import blocking_call

router = APIRouter(tags=["operations"])


@router.get("/metrics")
async def metrics():
    return {
        "stages": await blocking_call(lambda: state_store.metrics(principal.get())),
        "usage": await blocking_call(lambda: operations.usage(principal.get())),
        "resources": await blocking_call(operations.resource_sample),
    }


@router.get("/ready")
async def ready():
    def inspect():
        ffmpeg = shutil.which("ffmpeg")
        filters = ""
        if ffmpeg:
            filters = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=10
            ).stdout
        checks = {
            "ffmpeg": bool(ffmpeg),
            "watermark_filter": not settings.enable_watermark or "drawtext" in filters,
            "asr_package": importlib.util.find_spec("faster_whisper") is not None,
            "tts_package": importlib.util.find_spec("TTS") is not None,
            "disk": shutil.disk_usage(settings.job_artifacts_dir).free
            >= settings.min_free_disk_mb * 1024**2,
        }
        from .. import llm
        if settings.warmup_models:
            from ..main import _warmup_state
            checks["model_warmup"] = _warmup_state.get("status") == "done"
        if (llm.configured() or settings.translate_backend == "llm"
                or settings.quality_translate_backend == "llm"):
            checks["llm_model"] = llm.ready(ttl=60)
        from gpu_runtime import capabilities
        runtime = capabilities(settings.resolved_device)
        checks["gpu_runtime"] = runtime["ready"]
        return {
            "runtime": runtime,
            "ready": all(checks.values()),
            "checks": checks,
            "provenance": operations.provenance(),
            "gpu_validation": "pending",
            "optional_audio_quality": settings.audio_quality_url,
            "optimization_flags": {"windowed_lipsync": settings.windowed_lipsync},
        }

    return await blocking_call(inspect)
