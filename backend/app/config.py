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
    # CUDA only: faster-whisper's BatchedInferencePipeline decodes VAD-split
    # chunks as one batch instead of sequentially. 16 fits large-v3 fp16 in
    # a few GB; raise on 96 GB cards if transcribe shows up in profiles.
    whisper_batch_size: int = Field(16, ge=1, le=64)

    # `llm` is the GPU default: an OpenAI-compatible chat server (vLLM on the
    # XE7740). `nllb` is the self-contained seq2seq path kept for CPU hosts.
    translate_backend: Literal["nllb", "llm"] = "nllb"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    # OpenAI-compatible endpoint, e.g. http://host.docker.internal:2080/v1.
    # Empty disables every LLM feature (llm translation, rewrites, avatar).
    llm_base_url: str = ""
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm_api_key_file: str = ""
    llm_timeout_seconds: int = Field(120, ge=5)
    # Batch lane: translation and overrun rewrites.
    llm_max_concurrent: int = Field(2, ge=1, le=32)
    llm_max_pending: int = Field(8, ge=0, le=128)
    # Interactive lane: avatar reply streams. Separate so a couple of live
    # sessions cannot starve translation jobs into queue timeouts.
    llm_interactive_max_concurrent: int = Field(2, ge=1, le=32)
    llm_interactive_max_pending: int = Field(4, ge=0, le=128)
    llm_queue_timeout_seconds: float = Field(10, gt=0, le=120)
    llm_max_input_chars: int = Field(96000, ge=1024)
    llm_max_output_tokens: int = Field(4096, ge=160)
    llm_max_response_bytes: int = Field(262144, ge=4096)

    # Local ownership, scheduling and optional quality integrations.
    internal_api_key: str = ""
    auth_tokens_json: str = "{}"
    max_user_jobs: int = Field(8, ge=1)
    max_user_storage_mb: int = Field(10240, ge=1)
    min_free_disk_mb: int = Field(256, ge=0)
    retention_days: int = Field(0, ge=0)
    window_seconds: float = Field(8.0, ge=1, le=30)
    window_overlap_seconds: float = Field(0.4, ge=0, le=2)
    # Streaming translation (mode "stream"): render windows per speech span and
    # the pause that separates two spans; frames outside spans pass through.
    stream_window_seconds: float = Field(8.0, ge=2, le=30)
    # How far a span's last line may run into the next span (conversational overlap).
    stream_tail_overlap_seconds: float = Field(0.6, ge=0.0, le=2.0)
    stream_span_gap_seconds: float = Field(0.6, ge=0, le=5)
    stream_span_pad_seconds: float = Field(0.25, ge=0, le=2)
    windowed_lipsync: bool = False
    musetalk_frame_budget_mb: int = Field(4096, ge=64)
    latentsync_frame_budget_mb: int = Field(8192, ge=64)
    latentsync_realtime_factor: float = Field(5.0, gt=0)
    rewrite_overruns: bool = False
    tts_fit_retries: int = Field(2, ge=0, le=3)
    enable_diarization: bool = False
    # Per-speaker voices and faces (pipeline/speakers.py): runs after
    # transcription for LatentSync jobs unless the job sets speakers=one.
    speaker_analysis: bool = True
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
    quality_translate_backend: Literal["nllb", "llm"] = "nllb"
    tts_segment_retries: int = Field(1, ge=0, le=3)
    tts_max_speed: float = Field(1.15, ge=1.0, le=1.3)
    # Ceiling when no shorter faithful translation exists: stretch this far
    # rather than fail the job. Speech is never discarded either way.
    tts_max_speed_hard: float = Field(1.3, ge=1.0, le=1.5)
    # Slots shorter than this (conversational turns) may go up to the short ceiling.
    tts_short_slot_seconds: float = Field(3.0, ge=0.0, le=10.0)
    tts_max_speed_short: float = Field(1.5, ge=1.0, le=1.8)
    # XTTS native speed for retries of a take that overran its slot.
    tts_retry_speed: float = Field(1.2, ge=1.0, le=1.5)
    # A line the timeline plan speeds up by at least this factor is re-spoken
    # by XTTS at that speed (up to `tts_native_speed_takes` takes) before any
    # waveform stretch; the stretch then only covers what is left.
    tts_native_speed_min: float = Field(1.05, ge=1.0, le=2.0)
    tts_native_speed_takes: int = Field(2, ge=0, le=4)
    # How far a line may start after its source onset when it absorbs the
    # overrun of the line before it (the lip sync follows the audio).
    tts_max_drift_seconds: float = Field(1.5, ge=0.0, le=5.0)
    # Rather than failing a long job, a span that cannot fit within the
    # ceilings may go this fast; such lines are marked last_resort in timing.
    tts_max_speed_last_resort: float = Field(1.7, ge=1.0, le=2.0)
    # Extra same-text takes when one overruns the preferred speed (XTTS
    # length variance; the shortest verified take is kept), each ~3 s on the box.
    tts_overrun_retries: int = Field(4, ge=0, le=8)
    # Pauses inside a synthesized segment are capped at this (0 disables).
    tts_max_pause_seconds: float = Field(0.35, ge=0.0, le=2.0)
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
    # Optional independent fast renderer. Empty shares the batch service.
    latentsync_fast_service_url: str = ""
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
    # compose overlay sets. GPU encode failures are reported without software fallback.
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
