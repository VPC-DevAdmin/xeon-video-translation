"""Pause compression inside a synthesized take (needs ffmpeg + soundfile)."""

import shutil

import numpy as np
import pytest

from app.pipeline import tts

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone(seconds, rate=24000, freq=220.0):
    t = np.arange(int(seconds * rate)) / rate
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_long_pause_is_capped_and_words_are_kept(tmp_path):
    sf = pytest.importorskip("soundfile")
    rate = 24000
    audio = np.concatenate([_tone(1.0), np.zeros(int(3.0 * rate), np.float32), _tone(1.0), np.zeros(int(0.2 * rate), np.float32), _tone(0.5)])
    path = tmp_path / "take.wav"
    sf.write(str(path), audio, rate)
    removed = tts._compress_pauses(path, max_gap=0.35)
    assert 2.4 < removed < 2.8
    out, _ = sf.read(str(path), dtype="float32")
    assert abs(len(out) / rate - (5.7 - removed)) < 0.05
    # the words survive: total loud samples unchanged
    assert abs((np.abs(out) > 0.1).sum() - (np.abs(audio) > 0.1).sum()) < rate * 0.05


def test_short_pauses_are_left_alone(tmp_path):
    sf = pytest.importorskip("soundfile")
    rate = 24000
    audio = np.concatenate([_tone(1.0), np.zeros(int(0.3 * rate), np.float32), _tone(1.0)])
    path = tmp_path / "take.wav"
    sf.write(str(path), audio, rate)
    assert tts._compress_pauses(path, max_gap=0.35) == 0.0
