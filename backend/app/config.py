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
