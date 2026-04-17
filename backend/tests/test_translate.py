"""Stage 3 tests. Doesn't load NLLB — just verifies the dispatch and language map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline import translate as t


def test_lang_map_covers_demo_defaults() -> None:
    # Every language we expose in the UI must be in NLLB_LANG_CODES.
    demo = ["en", "es", "fr", "de", "ja", "zh", "hi", "pt"]
    for code in demo:
        assert code in t.NLLB_LANG_CODES, code
        assert code in t.LANG_NAMES, code


def test_translate_unknown_target_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(t.settings, "translate_backend", "nllb")
    transcript = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]}
    with pytest.raises(t.TranslationError):
        t.translate(transcript, tmp_path / "out.json", target_language="zz")


def test_translate_unknown_backend_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(t.settings, "translate_backend", "bogus")
    transcript = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]}
    with pytest.raises(t.TranslationError):
        t.translate(transcript, tmp_path / "out.json", target_language="es")
