import json

import pytest

from app.pipeline import tts


def test_final_phrase_can_use_trailing_video_silence(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    monkeypatch.setattr(
        tts, "_xtts_to_file", lambda text, ref, lang, path, **k: path.write_bytes(b"audio")
    )
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: True)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 1.20)
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    segments = [{"start": 0.43, "end": 1.15, "text": "Buenos días."}]
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"reference")
    out = tmp_path / "out.wav"
    tts._synthesize_per_segment(
        segments,
        segments,
        reference,
        "es",
        out,
        source_duration_seconds=1.766667,
        options={"rewrite_overruns": False},
    )
    timing = json.loads(out.with_suffix(".timing.json").read_text())[0]
    assert timing["start"] == 0.43
    assert timing["slot_end"] == 1.766667
    assert timing["speech_seconds"] == 1.2
    assert segments[0]["end"] == 1.15
    # Without the trailing silence it does not fit (last-resort speed off here).
    monkeypatch.setattr(tts.settings, "tts_max_speed_last_resort", 1.0)
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
        tts._synthesize_per_segment(
            segments, segments, reference, "es", out, options={"rewrite_overruns": False}
        )
