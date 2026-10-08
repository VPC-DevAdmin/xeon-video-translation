"""Lines the timeline plan speeds up: XTTS re-speaks them faster, and atempo
(not rubberband) covers whatever is left."""

import subprocess
import wave

from app.pipeline import tts


def test_respeak_keeps_the_shortest_verified_take_and_its_speed(tmp_path, monkeypatch):
    path = tmp_path / "line.wav"
    path.write_bytes(b"slow")
    durations = {path: 4.0}
    monkeypatch.setattr(tts, "_probe_duration", lambda p: durations[p])
    calls = []

    def retake(out, speed):
        calls.append(speed)
        out.write_bytes(b"fast%d" % len(calls))
        durations[path] = durations[out] = [3.6, 3.3][len(calls) - 1]
        return durations[out]

    kept = tts._respeak_faster(path, target=3.2, factor=1.25, retake=retake, base_speed=1.2)
    assert calls == [1.5, 1.5]  # compounded with the speed the kept take was made at
    assert kept == 1.5 and path.read_bytes() == b"fast2"


def test_respeak_rejects_unverified_or_longer_takes(tmp_path, monkeypatch):
    path = tmp_path / "line.wav"
    path.write_bytes(b"original")
    monkeypatch.setattr(tts, "_probe_duration", lambda p: 4.0)
    results = iter([None, 4.5])
    assert tts._respeak_faster(path, 3.0, 1.3, lambda out, speed: next(results), 1.0) is None
    assert path.read_bytes() == b"original" and sorted(p.name for p in tmp_path.iterdir()) == ["line.wav"]


def test_stretch_uses_atempo_and_hits_the_target(tmp_path):
    path = tmp_path / "speech.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                    "-ar", "24000", str(path)], check=True)
    tts._maybe_time_stretch(path, 1.6, max_speed=1.5)
    with wave.open(str(path)) as audio:
        assert abs(audio.getnframes() / audio.getframerate() - 1.6) < 0.03


def test_prepended_silence_keeps_16_bit_speech(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "speech.wav"
    rate = 24000
    t = np.arange(rate) / rate
    original = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    sf.write(str(path), original, rate, subtype="PCM_16")
    tts._prepend_silence(path, 0.25)
    padded, padded_rate = sf.read(str(path), dtype="int16")
    lead = round(0.25 * rate)
    assert padded_rate == rate and not padded[:lead].any()
    # bit-exact: the 8-bit concat path changed every sample by up to 255
    assert np.array_equal(padded[lead:lead + rate], original)
