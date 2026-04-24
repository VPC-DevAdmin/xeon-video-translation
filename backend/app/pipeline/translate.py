"""Stage 3 — translation.

Two backends are wired up:

- `nllb` (default, CPU friendly): facebook/nllb-200-distilled-600M via transformers.
- `ollama`: hits a local Ollama server. Slow on CPU for an 8B model but wired
  for parity with the spec.

The orchestrator picks based on `settings.translate_backend`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import settings


# BCP-47 -> NLLB FLORES-200 codes. NLLB-200 supports all 200 FLORES
# targets; the list here is the subset we've validated end-to-end
# (translate + TTS backend coverage). Add a row any time a new TTS
# backend is integrated so the translate stage can feed it.
#
# Format note: NLLB uses ISO 639-3 + script tag (e.g. "spa_Latn"),
# not BCP-47.
NLLB_LANG_CODES: dict[str, str] = {
    # Originally covered
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "vi": "vie_Latn",
    # Indic languages (added alongside IndicF5 TTS backend, PR #74).
    # IndicF5 produces native speech on these; without NLLB coverage
    # the pipeline failed at translate before reaching TTS.
    "bn": "ben_Beng",  # Bengali
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "mr": "mar_Deva",  # Marathi
    "gu": "guj_Gujr",  # Gujarati
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "pa": "pan_Guru",  # Punjabi (Gurmukhi script)
    "or": "ory_Orya",  # Odia
    "as": "asm_Beng",  # Assamese
}

# Human-readable names for prompt templating (Ollama backend).
LANG_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ja": "Japanese", "zh": "Mandarin Chinese", "hi": "Hindi", "ar": "Arabic",
    "ko": "Korean", "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese",
    # Indic
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese",
}


class TranslationError(RuntimeError):
    pass


@dataclass
class TranslatedSegment:
    start: float
    end: float
    source_text: str
    text: str


@dataclass
class TranslationResult:
    source_language: str
    target_language: str
    backend: str
    text: str
    segments: list[TranslatedSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "backend": self.backend,
            "text": self.text,
            "segments": [
                {
                    "start": s.start, "end": s.end,
                    "source_text": s.source_text, "text": s.text,
                }
                for s in self.segments
            ],
        }


# --------------------------------------------------------------------------- #
# NLLB backend
# --------------------------------------------------------------------------- #

_nllb_pipeline = None
_nllb_lock = Lock()


def _get_nllb_pipeline():
    global _nllb_pipeline
    if _nllb_pipeline is not None:
        return _nllb_pipeline
    with _nllb_lock:
        if _nllb_pipeline is not None:
            return _nllb_pipeline
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        cache_dir = str(settings.model_cache_dir / "huggingface")
        tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.nllb_model, cache_dir=cache_dir)
        _nllb_pipeline = (tokenizer, model)
        return _nllb_pipeline


def _translate_segment_nllb(text: str, src: str, tgt: str) -> str:
    tokenizer, model = _get_nllb_pipeline()
    tokenizer.src_lang = src
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt)
    output_ids = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_new_tokens=512,
        num_beams=4,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


# --------------------------------------------------------------------------- #
# Ollama backend
# --------------------------------------------------------------------------- #

_OLLAMA_PROMPT = (
    "You are a professional translator. Translate the following text from "
    "{src_name} to {tgt_name}. Preserve the speaker's tone and register. "
    "Match the approximate length of the original so the translated speech "
    "fits in roughly the same time. Output only the translation, no commentary.\n\n"
    "Text: {text}"
)


def _translate_segment_ollama(text: str, src: str, tgt: str) -> str:
    src_name = LANG_NAMES.get(src, src)
    tgt_name = LANG_NAMES.get(tgt, tgt)
    body = json.dumps({
        "model": settings.ollama_model,
        "prompt": _OLLAMA_PROMPT.format(src_name=src_name, tgt_name=tgt_name, text=text),
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.ollama_host.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise TranslationError(f"ollama request failed: {e}") from e

    return payload.get("response", "").strip()


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #


def translate(
    transcript: dict[str, Any],
    output_path: Path,
    target_language: str,
    source_language: str | None = None,
) -> TranslationResult:
    """Translate a transcript dict (from Stage 2) to `target_language`.

    `target_language` and `source_language` are 2-letter BCP-47 codes.
    Source falls back to the language detected in the transcript.
    """
    src = (source_language or transcript.get("language") or "en").lower()
    tgt = target_language.lower()
    backend = settings.translate_backend

    if backend == "nllb":
        if src not in NLLB_LANG_CODES:
            raise TranslationError(f"NLLB: unsupported source language {src!r}")
        if tgt not in NLLB_LANG_CODES:
            raise TranslationError(f"NLLB: unsupported target language {tgt!r}")
        nllb_src = NLLB_LANG_CODES[src]
        nllb_tgt = NLLB_LANG_CODES[tgt]
        translate_fn = lambda t: _translate_segment_nllb(t, nllb_src, nllb_tgt)  # noqa: E731
    elif backend == "ollama":
        translate_fn = lambda t: _translate_segment_ollama(t, src, tgt)  # noqa: E731
    else:
        raise TranslationError(f"unknown translate backend: {backend!r}")

    out_segments: list[TranslatedSegment] = []
    for seg in transcript.get("segments", []):
        source_text = seg["text"].strip()
        if not source_text:
            continue
        translated = translate_fn(source_text).strip()
        out_segments.append(TranslatedSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            source_text=source_text,
            text=translated,
        ))

    full_text = " ".join(s.text for s in out_segments).strip()
    result = TranslationResult(
        source_language=src,
        target_language=tgt,
        backend=backend,
        text=full_text,
        segments=out_segments,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return result
