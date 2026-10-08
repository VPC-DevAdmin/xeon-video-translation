"""Persona capture checks: script matching and voice level rules (no models)."""

import numpy as np

from app.api.personas import RULES, script_match, voice_level_checks


def test_script_match_counts_words_in_order():
    script = "The quick brown fox jumps over the lazy dog."
    assert script_match(script, "the quick brown fox jumps over the lazy dog") == 1.0
    assert 0.5 < script_match(script, "the quick fox jumps over a dog") < 1.0
    assert script_match(script, "completely different words here") < 0.2
    assert script_match("", "anything") == 0.0


def test_voice_level_checks_flag_short_quiet_and_clipped_recordings():
    rate = 24000
    good = (0.2 * np.sin(np.linspace(0, 2 * np.pi * 220 * 26, rate * 26))).astype(np.float32)
    result = voice_level_checks(good, rate)
    assert result["ok"] and result["duration_seconds"] == 26.0 and -25 < result["level_dbfs"] < -10
    short = voice_level_checks(good[: rate * 5], rate)
    assert not short["ok"] and any("at least" in p for p in short["problems"])
    quiet = voice_level_checks(good * 0.002, rate)
    assert any("too quiet" in p for p in quiet["problems"])
    clipped = voice_level_checks(np.clip(good * 20, -1, 1), rate)
    assert any("clips" in p for p in clipped["problems"])
    silence = voice_level_checks(np.zeros(rate * 20, np.float32), rate)
    assert not silence["ok"]
    assert RULES["voice_min_seconds"] == 15
