from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Models
    whisper_model: str = "base"
    whisper_compute_type: Literal["int8", "int8_float32", "float32"] = "int8"

    translate_backend: Literal["nllb", "ollama"] = "nllb"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Limits
    max_video_duration_seconds: int = 60
    max_video_size_mb: int = 100
    max_concurrent_jobs: int = 1

    # Paths (resolved to absolute on init)
    model_cache_dir: Path = Path("./models")
    job_artifacts_dir: Path = Path("./jobs")

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3030,http://localhost:3000"

    # Lipsync
    # Backends:
    #   none        — skip lipsync; mux dubs the new audio over the original video
    #   wav2lip     — Wav2Lip (2020); ~30-60s for a 3s clip on a 16-core Xeon
    #   musetalk    — stubbed in this PR (see docs/lipsync.md)
    #   latentsync  — stubbed in this PR (see docs/lipsync.md)
    lipsync_backend: Literal["none", "wav2lip", "musetalk", "latentsync"] = "none"
    # GitHub release mirror of the Wav2Lip checkpoint (CC-BY-NC 4.0 weights).
    # Release assets are immutable, so this URL is stable. If it ever 404s,
    # see docs/lipsync.md for alternate HuggingFace mirrors.
    wav2lip_checkpoint_url: str = (
        "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
    )
    # MuseTalk lipsync microservice. Must be reachable from inside the backend
    # container — the docker-compose service name is the default.
    musetalk_service_url: str = "http://lipsync-musetalk:8000"
    # How long to wait for MuseTalk to finish a single request. Generous:
    # CPU inference for a 30s clip can run into tens of minutes.
    musetalk_timeout_seconds: int = 1800
    # LatentSync lipsync microservice. PR-LS-1a ships the scaffold only:
    # the service returns 501 until PR-LS-1c lands. The timeout is sized
    # for the eventual inference path — LatentSync on CPU is a batch
    # workflow (~10 min per second of source video), not a live one.
    latentsync_service_url: str = "http://lipsync-latentsync:8000"
    latentsync_timeout_seconds: int = 14400  # 4 hours — batch job territory
    # Watermark text drawn on the output video. Respect responsible-use guidance.
    watermark_text: str = "AI-translated"

    # TTS
    # Backends:
    #   xtts    — Coqui XTTS-v2 (default). 16 languages, CPML-licensed weights.
    #   f5tts   — F5-TTS. Strong on EN/ZH; other languages supported via
    #             community fine-tunes (see docs/models.md for the honest
    #             language support matrix — experimental outside EN/ZH).
    tts_backend: Literal["xtts", "f5tts"] = "xtts"
    # F5-TTS checkpoint to use. The default multilingual base supports EN/ZH
    # out of the box; community fine-tunes for other languages can be pointed
    # at here once we pre-download them in scripts/download_models.sh.
    f5tts_model: str = "F5-TTS_v1"

    # Pre-stabilization (Stage 1.5). Optional pass that runs ffmpeg's
    # vidstab (or deshake fallback) on the source video before the rest
    # of the pipeline sees it. Primary benefit: the lipsync stage gets
    # a stable source, so landmark detection → affine warp → face
    # compositing all carry less per-frame jitter. Best for clips with
    # handheld shake but a mostly-stationary subject; tradeoff is
    # some smoothing of intentional motion (head turns) and brief
    # warping artifacts on very fast movement. Default off — most
    # source videos are fine without it, and it adds ~2× source
    # duration to wall-clock time.
    enable_video_stabilization: bool = False
    # vidstab smoothing window (value*2+1 frames of centered moving
    # average on detected motion vectors). 10 is a balanced default —
    # eliminates handheld jitter without noticeable lag on head turns.
    # 20+ gives a "locked-off tripod" feel at the cost of visible lag
    # on real subject motion. 5 or lower is barely smoother than raw.
    stabilize_smoothing: int = 10
    # vidstab detection sensitivity (1-10). Higher = more aggressive
    # motion search, catches more shake but also more false positives
    # that can register as shake where none exists. 5 is the upstream
    # default and works well for typical phone-held footage.
    stabilize_shakiness: int = 5

    # Feature flags
    enable_watermark: bool = True
    enable_c2pa: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.job_artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.model_cache_dir = settings.model_cache_dir.resolve()
settings.job_artifacts_dir = settings.job_artifacts_dir.resolve()
settings.ensure_dirs()
