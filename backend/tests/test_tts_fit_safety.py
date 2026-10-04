import pytest

from app.pipeline import quality, tts


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_segment_retries", 0)
    monkeypatch.setattr(tts.settings, "tts_fit_retries", 1)
    monkeypatch.setattr(tts.settings, "tts_max_speed", 1.15)
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    monkeypatch.setattr(
        tts, "_xtts_to_file", lambda text, ref, lang, path, **k: path.write_bytes(b"audio")
    )
    monkeypatch.setattr(tts, "_probe_duration", lambda *a: 2.0)
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: True)

    # These failure cases must never assemble or publish final audio.
    def unexpected_assembly(*args):
        pytest.fail("Rejected speech reached assembly")

    monkeypatch.setattr(tts, "_assemble_timeline", unexpected_assembly)
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"reference")
    segments = [{"start": 0, "end": 1, "text": "Buenos días."}]
    return segments, reference, tmp_path / "out.wav"


def test_faithful_rewrite_failure_is_actionable(fitting, monkeypatch):
    def cannot_shorten(*args):
        raise ValueError("no shorter faithful translation produced")

    monkeypatch.setattr(quality, "rewrite", cannot_shorten)
    segments, reference, out = fitting
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
        tts._synthesize_per_segment(
            segments, segments, reference, "es", out, options={"rewrite_overruns": True}
        )
    assert not out.exists()


def test_rewritten_take_must_match_all_words(fitting, monkeypatch):
    monkeypatch.setattr(quality, "rewrite", lambda *a: "Buen día.")
    decisions = iter([True, False])
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: next(decisions))
    segments, reference, out = fitting
    with pytest.raises(tts.TTSError, match="rewritten speech does not match"):
        tts._synthesize_per_segment(
            segments, segments, reference, "es", out, options={"rewrite_overruns": True}
        )
    assert segments[0]["text"] == "Buenos días."
    assert not out.exists()
