"""Stage 4 — voice cloning / TTS via XTTS-v2.

Generates `translated_audio.wav` from the translated text, using the original
speaker's voice as a reference.

CPU realism: XTTS-v2 runs at roughly 0.3–0.7× realtime on a 16-core Xeon.
A 3-second translation typically takes 5–10 s to synthesize.

Simplifications in v1 (documented limitations):
- Uses the whole Stage 1 WAV as the speaker reference rather than picking the
  longest clean segment. Works well enough when the source is single-speaker
  and well-miked.
- Generates the entire translated text in one pass. No per-segment timing
  alignment, so the TTS duration will not exactly match the source video.
- No loudness normalization or post-processing. Add in a follow-up if needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import settings


# BCP-47 -> XTTS-v2 language codes. XTTS supports these; our NLLB language
# picker is a superset, so some translations may not be synthesizable.
XTTS_LANG_CODES: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "hu": "hu", "ko": "ko", "ja": "ja",
    "hi": "hi",
    "zh": "zh-cn",  # XTTS uses the regional tag
}


class TTSError(RuntimeError):
    pass


@dataclass
class TTSResult:
    backend: str
    language: str
    reference_audio: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "language": self.language,
            "reference_audio": self.reference_audio,
            "output_path": self.output_path,
        }


_tts = None
_tts_lock = Lock()


def _get_tts():
    """Lazy-load XTTS-v2. First call downloads ~1.8 GB into HF_HOME."""
    global _tts
    if _tts is not None:
        return _tts
    with _tts_lock:
        if _tts is not None:
            return _tts

        # Coqui prompts for CPML license agreement on first load; pre-accept.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # Cache path to keep weights alongside other models in MODEL_CACHE_DIR.
        os.environ.setdefault(
            "TTS_HOME",
            str(settings.model_cache_dir / "coqui-tts"),
        )

        from TTS.api import TTS  # noqa: N811

        model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to("cpu")
        _tts = model
        return _tts


def synthesize(
    translation: dict,
    reference_audio: Path,
    output_path: Path,
) -> TTSResult:
    """Generate speech for `translation` using `reference_audio` as the voice.

    `translation` is the dict written by Stage 3 (see translate.py).
    """
    tgt = translation.get("target_language", "").lower()
    if tgt not in XTTS_LANG_CODES:
        raise TTSError(
            f"XTTS-v2 does not support target language {tgt!r}. "
            f"Supported: {sorted(XTTS_LANG_CODES)}"
        )

    text = (translation.get("text") or "").strip()
    if not text:
        raise TTSError("translation is empty — nothing to synthesize")

    if not reference_audio.exists():
        raise TTSError(f"reference audio missing: {reference_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = _get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=str(reference_audio),
        language=XTTS_LANG_CODES[tgt],
        file_path=str(output_path),
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError("XTTS produced no output")

    return TTSResult(
        backend="xtts_v2",
        language=tgt,
        reference_audio=reference_audio.name,
        output_path=output_path.name,
    )
