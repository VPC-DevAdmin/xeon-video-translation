"""Stage 2 — transcription via faster-whisper (CPU int8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


_model: "WhisperModel | None" = None
_model_lock = Lock()


def _get_model() -> "WhisperModel":
    """Lazy-load the faster-whisper model. Thread-safe singleton."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.model_cache_dir / "faster-whisper"),
        )
        return _model


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    text: str
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "text": self.text,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in self.segments],
        }


def transcribe(
    audio_path: Path,
    output_path: Path,
    language: str | None = None,
) -> TranscriptionResult:
    """Run faster-whisper on `audio_path`, write JSON to `output_path`, return result.

    `language` is a 2-letter ISO code (e.g. "en"); if None, auto-detect.
    """
    model = _get_model()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    segments = [Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in segments_iter]
    full_text = " ".join(s.text for s in segments).strip()

    result = TranscriptionResult(
        language=info.language,
        language_probability=float(info.language_probability),
        duration=float(info.duration),
        text=full_text,
        segments=segments,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return result
