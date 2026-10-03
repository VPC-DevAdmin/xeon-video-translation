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

    # Compute device for the in-process models (whisper, NLLB, TTS, Wav2Lip).
    #   cpu   — the original Xeon build. Default so the CPU images keep working.
    #   cuda  — one GPU; pick which with CUDA_VISIBLE_DEVICES on the container.
    #   auto  — cuda if torch can see one, else cpu.
    # The lipsync microservices have their own DEVICE env (see
    # docker-compose.gpu.yml). See docs/gpu/README.md for the GPU track.
    device: Literal["cpu", "cuda", "auto"] = "cpu"

    # Models
    whisper_model: str = "base"
    # float16 is the GPU default; int8 variants are the CPU defaults.
    whisper_compute_type: Literal["int8", "int8_float32", "float16", "float32"] = "int8"

    translate_backend: Literal["nllb", "ollama"] = "nllb"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Local ownership, scheduling and optional quality integrations.
    internal_api_key: str = ""
    auth_tokens_json: str = "{}"
    max_user_jobs: int = Field(8, ge=1)
    max_user_storage_mb: int = Field(10240, ge=1)
    min_free_disk_mb: int = Field(256, ge=0)
    retention_days: int = Field(0, ge=0)
    window_seconds: float = Field(8.0, ge=1, le=30)
    window_overlap_seconds: float = Field(0.4, ge=0, le=2)
    windowed_lipsync: bool = False
    musetalk_frame_budget_mb: int = Field(4096, ge=64)
    latentsync_frame_budget_mb: int = Field(8192, ge=64)
    latentsync_realtime_factor: float = Field(5.0, gt=0)
    rewrite_overruns: bool = False
    tts_fit_retries: int = Field(2, ge=0, le=3)
    enable_diarization: bool = False
    enable_alignment: bool = False
    enable_background_audio: bool = False
    background_gain: float = Field(0.35, ge=0, le=1)
    audio_quality_url: str = "http://audio-quality:8000"
    model_revision: str | None = None

    # Limits
    max_video_duration_seconds: int = 60
    max_video_size_mb: int = 100
    max_concurrent_jobs: int = Field(1, ge=1)
    recover_jobs: bool = True
    max_pending_jobs: int = Field(32, ge=1)
    quality_translate_backend: Literal["nllb", "ollama"] = "nllb"
    tts_segment_retries: int = Field(1, ge=0, le=3)
    tts_max_speed: float = Field(1.15, ge=1.0, le=1.3)
    tts_timing_tolerance: float = Field(0.15, ge=0.0, le=1.0)

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
    #   musetalk    — microservice; minutes per clip on CPU (see docs/lipsync.md)
    #   latentsync  — microservice; hours per clip on CPU (see docs/lipsync.md)
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
    # LatentSync is a batch job. At fp32 defaults (PR #67) a 30-second
    # clip already takes ~5 hours, so any fixed ceiling becomes a landmine
    # that kills legitimate runs. Default to None = no client-side
    # ceiling; let the service finish or crash on its own (we still have
    # per-stage progress reporting via the shared JSON file). Override
    # with a positive int only if you want urlopen to bail after N seconds.
    latentsync_timeout_seconds: int | None = None
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
    # f5-tts >= 1.1 renamed the config from "F5-TTS_v1" to "F5TTS_v1_Base"
    # (see f5_tts/configs/). The name must match a yaml in that directory.
    f5tts_model: str = "F5TTS_v1_Base"

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

    # Post-stabilization — stabilize the lipsync output (lipsynced.mp4)
    # before mux. Catches any jitter introduced by the lipsync pipeline
    # itself (affine residual, VAE precision, etc.) that pre-stab can't
    # address because pre-stab runs before the pipeline. Independent of
    # enable_video_stabilization; the two can stack or be used alone.
    enable_output_stabilization: bool = False

    # Final-mux video encoder. libx264 is the portable default. h264_nvenc
    # offloads the watermark/pad re-encode to the GPU's hardware encoder
    # (the mux took ~10 s of CPU for a 30 s clip on the XE7740); it needs
    # the container to have the `video` driver capability, which the GPU
    # compose overlay sets. Falls back to libx264 if ffmpeg rejects it.
    video_encoder: Literal["libx264", "h264_nvenc"] = "libx264"

    # Load whisper, NLLB and the default TTS backend at startup instead of
    # on the first job. Costs ~60 s of startup on the GPU box and ~40 GB of
    # VRAM held idle; saves the same minute on the first request, which
    # matters for a demo and for honest per-stage timings. Off on CPU.
    warmup_models: bool = False

    # Feature flags
    enable_watermark: bool = True
    enable_c2pa: bool = False

    @property
    def resolved_device(self) -> str:
        """`device` with `auto` resolved against the torch runtime."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

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
