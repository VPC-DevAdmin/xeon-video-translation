from pydantic import BaseModel, Field, ConfigDict, field_validator


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    glossary: dict[str, str] = Field(default_factory=dict, max_length=100)
    speaker_voices: dict[str, str] = Field(default_factory=dict, max_length=20)
    voice: str | None = Field(None, max_length=100)
    diarization: bool = False
    alignment: bool = False
    background_audio: bool = False
    background_gain: float = Field(0.35, ge=0, le=1)
    rewrite_overruns: bool = False
    windowed_lipsync: bool = False

    @field_validator("glossary", "speaker_voices")
    @classmethod
    def bounded_strings(cls, mapping):
        if any(
            not key.strip() or not value.strip() or max(len(key), len(value)) > 200
            for key, value in mapping.items()
        ):
            raise ValueError("empty or excessive mapping entry")
        return mapping


def from_request(value):
    from .config import settings

    parsed = JobOptions.model_validate_json(value)
    defaults = {
        "alignment": settings.enable_alignment,
        "diarization": settings.enable_diarization,
        "background_audio": settings.enable_background_audio,
        "background_gain": settings.background_gain,
        "windowed_lipsync": settings.windowed_lipsync,
        "rewrite_overruns": settings.rewrite_overruns,
    }
    return JobOptions.model_validate(
        {**defaults, **parsed.model_dump(exclude_unset=True)}
    ).model_dump()
