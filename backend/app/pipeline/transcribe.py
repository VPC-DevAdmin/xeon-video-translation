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
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    text: str
    segments: list[Segment]
    # Time (in seconds from clip start) at which the first recognized word
    # begins. Used downstream to align TTS speech-start to the source's
    # pre-speech silence. `None` if the clip has no detected speech at all.
    first_speech_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "text": self.text,
            "first_speech_seconds": self.first_speech_seconds,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {"start": w.start, "end": w.end, "text": w.text}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
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
        # Per-word timestamps feed the TTS speech-start alignment: we align
        # the TTS clip's first spoken frame to the source's first spoken
        # frame. Roughly 10-15% extra CPU at this stage; worth it.
        word_timestamps=True,
        vad_filter=True,
        # 500 ms matches the architecture review's anti-hallucination
        # recommendation. Shorter pauses are preserved in the transcript.
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words: list[Word] = []
        for w in getattr(seg, "words", None) or []:
            words.append(Word(
                start=float(w.start),
                end=float(w.end),
                text=str(w.word).strip(),
            ))
        segments.append(Segment(
            start=float(seg.start),
            end=float(seg.end),
            text=seg.text.strip(),
            words=words,
        ))
    full_text = " ".join(s.text for s in segments).strip()

    # The first detectable word start. Falls back to segment start if
    # word_timestamps didn't produce word-level info for some reason.
    first_speech_seconds: float | None = None
    for seg in segments:
        if seg.words:
            first_speech_seconds = seg.words[0].start
            break
        if seg.start is not None:
            first_speech_seconds = seg.start
            break

    result = TranscriptionResult(
        language=info.language,
        language_probability=float(info.language_probability),
        duration=float(info.duration),
        text=full_text,
        segments=segments,
        first_speech_seconds=first_speech_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return result
