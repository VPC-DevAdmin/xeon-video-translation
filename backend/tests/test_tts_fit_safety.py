import pytest

from app.pipeline import quality, tts


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.settings, "tts_segment_retries", 0)
    monkeypatch.setattr(tts.settings, "tts_fit_retries", 1)
    monkeypatch.setattr(tts.settings, "tts_max_speed", 1.15)
    monkeypatch.setattr(tts.settings, "tts_overrun_retries", 2)
    monkeypatch.setattr(tts.settings, "tts_max_speed_last_resort", 1.0)
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
    decisions = iter([True, True, True, False])  # three outlier takes, then the rewritten one
    monkeypatch.setattr(tts, "_trim_tail_via_whisper", lambda *a: next(decisions))
    segments, reference, out = fitting
    # The rejected rewrite never replaces the verified take; the 2x overrun then
    # fails as an overflow, with the original text intact and nothing assembled.
    with pytest.raises(tts.TTSError, match="No speech was discarded"):
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
    monkeypatch.setattr(tts.settings, "tts_short_slot_seconds", 0.0)
    monkeypatch.setattr(tts.settings, "tts_max_speed_last_resort", 1.0)
    monkeypatch.setattr(tts.settings, "tts_overrun_retries", 2)
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
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.18, 1.19, 1.18])
    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": True}
    )
    assert stretched == [(1.0, 1.3)]
    assert segments[0]["text"] == "Buenos días."


def test_slower_rewrite_take_never_replaces_the_best_one(tmp_path, monkeypatch):
    """The LLM shortened the text but XTTS rendered it slower: keep the first take."""
    rewrites = iter(["Buen día.", "Buen día."])
    monkeypatch.setattr(quality, "rewrite", lambda *a: next(rewrites))
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.2, 1.3, 1.25, 1.9, 1.7])
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
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.6, 1.6, 1.6])
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


def test_outlier_take_earns_extra_attempts_and_the_shortest_wins(tmp_path, monkeypatch):
    """XTTS gave 13.5 s for a 9.1 s slot once and the job failed. A take beyond
    the hard ceiling now earns extra attempts and the shortest verified take is kept."""
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.5, 1.4, 1.2])
    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": False}
    )
    assert stretched == [(1.0, 1.3)]  # the 1.2 s take fitted within the hard ceiling


def test_best_of_n_stops_at_the_first_take_that_fits(tmp_path, monkeypatch):
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.5, 1.1, 9.9])
    tts._synthesize_per_segment(
        segments, segments, reference, "es", out, options={"rewrite_overruns": False}
    )
    assert stretched == [(1.0, 1.3)]  # the 1.1 s take was fitted; the third take was never generated


def test_short_slot_allows_the_short_ceiling_and_retries_speak_faster(tmp_path, monkeypatch):
    """A 1.6 s turn: the Spanish needed 1.44x and failed the job on 7 Oct 2026."""
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.5, 1.45, 1.44])
    monkeypatch.setattr(tts.settings, "tts_short_slot_seconds", 3.0)
    monkeypatch.setattr(tts.settings, "tts_max_speed_short", 1.5)
    speeds = []
    real = tts._xtts_to_file

    def spy(text, ref, lang, path, **k):
        speeds.append(k.get("speed"))
        return real(text, ref, lang, path, **k)

    monkeypatch.setattr(tts, "_xtts_to_file", spy)
    tts._synthesize_per_segment(segments, segments, reference, "es", out, options={"rewrite_overruns": False})
    assert stretched == [(1.0, 1.5)]
    assert speeds[0] is None                                          # first take at normal speed
    assert speeds[1:] and all(s == tts.settings.tts_retry_speed for s in speeds[1:])


def test_plan_borrows_the_following_pause_before_speeding_up():
    # line 1 overruns its 2 s slot by 0.6 s; line 2 has room after it
    placements, factors = tts._plan_timeline([0.0, 2.0], [2.6, 1.0], [2.0, 5.0], [1.3, 1.3], 5.0, max_drift=1.5)
    assert factors == [1.0, 1.0]
    assert placements == pytest.approx([0.0, 2.6])


def test_plan_speeds_up_at_a_hard_end_within_the_ceiling():
    placements, factors = tts._plan_timeline([0.0], [2.0], [1.5], [1.5], 1.5)
    assert factors[0] == pytest.approx(2.0 / 1.5)
    assert tts._plan_timeline([0.0], [2.4], [1.5], [1.5], 1.5) is None


def test_plan_respects_the_drift_cap():
    # unhurried, line 2 would start 2 s late; the plan speeds line 1 up just enough
    placements, factors = tts._plan_timeline([0.0, 1.0], [3.0, 0.5], [1.0, 4.0], [1.3, 1.3], 4.0, max_drift=1.5)
    assert placements[1] - 1.0 <= 1.5 + 1e-6
    assert 1.15 <= factors[0] <= 1.3


def test_last_resort_speed_rather_than_failing(tmp_path, monkeypatch):
    segments, reference, out, stretched = _fitting_at(monkeypatch, tmp_path, [1.6, 1.6, 1.6])
    monkeypatch.setattr(tts.settings, "tts_max_speed_last_resort", 1.7)
    tts._synthesize_per_segment(segments, segments, reference, "es", out, options={"rewrite_overruns": False})
    import json

    timing = json.loads(out.with_suffix(".timing.json").read_text())[0]
    assert timing["last_resort"] is True and timing["speed"] == pytest.approx(1.6, abs=1e-3)
    assert stretched == [(1.0, 1.7)]


def test_numbers_are_compared_as_words():
    assert "ochenta y ocho" in tts._spell_numbers("unos 88 años", "es")
    assert "por ciento" in tts._spell_numbers("al 85%", "es")


def test_near_match_is_ambiguous_not_different():
    assert tts._near_match("estoesunafrasecompletaylarga", "estoesunafrasecompletaylargas") is None
    assert tts._near_match("estoesunafrase", "otracosadistinta") is False
