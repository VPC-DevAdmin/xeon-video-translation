"""Stage 4 (TTS) tests. Doesn't load XTTS-v2 — only checks validation and
the BCP-47 → XTTS language mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline import tts


def test_lang_map_covers_demo_defaults() -> None:
    # Every language exposed in the frontend picker must have an XTTS code,
    # otherwise TTS fails for that target.
    demo = ["en", "es", "fr", "de", "ja", "zh", "hi", "pt", "it", "ko"]
    for code in demo:
        assert code in tts.XTTS_LANG_CODES, code


def test_zh_maps_to_regional() -> None:
    # XTTS uses "zh-cn" not bare "zh".
    assert tts.XTTS_LANG_CODES["zh"] == "zh-cn"


def test_synthesize_unsupported_target_raises(tmp_path: Path) -> None:
    # A language supported by NLLB but not by XTTS (e.g. Vietnamese) should
    # surface as a clear error rather than loading the model and failing mid-run.
    translation = {"target_language": "vi", "text": "xin chào"}
    with pytest.raises(tts.TTSError, match="does not support"):
        tts.synthesize(
            translation=translation,
            reference_audio=tmp_path / "ref.wav",
            output_path=tmp_path / "out.wav",
        )


def test_synthesize_empty_text_raises(tmp_path: Path) -> None:
    translation = {"target_language": "es", "text": ""}
    with pytest.raises(tts.TTSError, match="empty"):
        tts.synthesize(
            translation=translation,
            reference_audio=tmp_path / "ref.wav",
            output_path=tmp_path / "out.wav",
        )


def test_synthesize_missing_reference_raises(tmp_path: Path) -> None:
    translation = {"target_language": "es", "text": "Buen día"}
    with pytest.raises(tts.TTSError, match="reference audio missing"):
        tts.synthesize(
            translation=translation,
            reference_audio=tmp_path / "does-not-exist.wav",
            output_path=tmp_path / "out.wav",
        )
