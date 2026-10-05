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


def _fitting_at(monkeypatch, tmp_path, durations):
    """Like `fitting`, but takes have the given durations in order and may assemble."""
    monkeypatch.setattr(tts.settings, "tts_segment_retries", 0)
    monkeypatch.setattr(tts.settings, "tts_fit_retries", 2)
    monkeypatch.setattr(tts.settings, "tts_max_speed", 1.15)
    monkeypatch.setattr(tts.settings, "tts_max_speed_hard", 1.3)
    monkeypatch.setattr(tts, "_select_reference", lambda *a: None)
    takes = iter(durations)
    current = {}

    def take(text, ref, lang, path, **k):
        current[path] = next(takes)
        path.write_bytes(b"audio")

    monkeypatch.setattr(tts, "_xtts_to_file", take)
    monkeypatch.setattr(tts, "_probe_duration", lambda path, *a: current.get(path, current.get("best", 0.0)))
    monkeypatch.setattr(tts, "_trim_to_speech", lambda *a: None)
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: True)
    stretched = []

    def stretch(path, target_duration, max_speed=None):
        stretched.append((target_duration, max_speed))
        current[path] = target_duration

    monkeypatch.setattr(tts, "_maybe_time_stretch", stretch)
    monkeypatch.setattr(tts, "_assemble_timeline", lambda *a: None)
    # copyfile moves takes between paths; follow the durations along.
    import shutil

    real_copy = shutil.copyfile

    def copy(src, dst, *a, **k):
        real_copy(src, dst, *a, **k)
        if src in current:
            current[dst] = current[src]

    monkeypatch.setattr(shutil, "copyfile", copy)
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"reference")
    segments = [{"start": 0, "end": 1, "text": "Buenos días."}]
    return segments, reference, tmp_path / "out.wav", stretched


def test_small_overrun_is_stretched_when_no_rewrite_exists(tmp_path, monkeypatch):
    """A 1.18x overrun with no shorter faithful translation is stretched to the
    hard ceiling instead of failing the job (the 9 s stream clip on 5 Oct)."""

    def cannot_shorten(*args):
        raise ValueError("no shorter faithful translation produced")

    monkeypatch.setattr(quality, "rewrite", cannot_shorten)
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.18])
    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": True}
    )
    assert stretched == [(1.0, 1.3)]
    assert segments[0]["text"] == "Buenos días."


def test_slower_rewrite_take_never_replaces_the_best_one(tmp_path, monkeypatch):
    """The LLM shortened the text but XTTS rendered it slower: keep the first take."""
    rewrites = iter(["Buen día.", "Buen día."])
    monkeypatch.setattr(quality, "rewrite", lambda *a: next(rewrites))
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.2, 1.9, 1.7])
    import json

    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": True}
    )
    timing = json.loads(out.with_suffix(".timing.json").read_text())[0]
    # The 1.2 s first take was kept (1.2x, within the hard ceiling) and stretched
    # into the slot; the slower rewritten takes were dropped.
    assert stretched == [(1.0, 1.3)]
    assert timing["speech_seconds"] == 1.0
    assert timing["text"] == "Buenos días."
    assert segments[0]["text"] == "Buenos días."
    assert "original_text" not in segments[0]


def test_large_overrun_still_fails_without_discarding(tmp_path, monkeypatch):
    def cannot_shorten(*args):
        raise ValueError("no shorter faithful translation produced")

    monkeypatch.setattr(quality, "rewrite", cannot_shorten)
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.6])
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
        tts._synthesize_per_segment(
            segments, segments, reference, "es", out, options={"rewrite_overruns": True}
        )
    assert stretched == []


def test_verified_takes_still_lose_their_trailing_silence(tmp_path, monkeypatch):
    """A whisper-verified take used to skip the silence trim, so XTTS's trailing
    silence was stretched into the slot and the speech ended early (the source
    mouth then showed through for the last 0.6 s of the 5 Oct clip)."""
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [0.9])
    trimmed = []
    monkeypatch.setattr(tts, "_trim_to_speech", lambda path: trimmed.append(path))
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: True)
    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": True}
    )
    assert len(trimmed) == 1
