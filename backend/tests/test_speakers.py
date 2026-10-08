"""Speaker analysis: word labels, voice-to-face matching, regrouping. No models."""

import numpy as np
import pytest

from app.pipeline import speakers, streaming


def _words(spec):
    """spec: list of (start, end, text)."""
    return [{"start": a, "end": b, "text": t} for a, b, t in spec]


def test_word_labels_take_the_majority_window():
    times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    words = _words([(0.4, 0.9, "hi"), (2.2, 2.8, "there")])
    assert speakers.word_labels(words, times, labels) == [0, 1]


def test_single_short_word_between_same_speaker_is_smoothed():
    words = _words([(0, 0.5, "a"), (0.5, 0.7, "b"), (0.7, 1.2, "c")])
    assert speakers.smooth_labels(words, [0, 1, 0]) == [0, 0, 0]
    # a long word is a real turn
    words[1]["end"] = 1.4
    assert speakers.smooth_labels(words, [0, 1, 0]) == [0, 1, 0]


def test_voice_is_matched_to_the_moving_mouth():
    fps, frames = 25, 250
    still = [0.05] * frames
    moving = [0.05 + (0.2 if (i // 3) % 2 else 0.0) for i in range(frames)]
    # person 0 (left) talks in 0-5 s, person 1 in 5-10 s
    left = moving[:125] + still[125:]
    right = still[:125] + moving[125:]
    activity = speakers.mouth_activity([left, right])
    words = _words([(t, t + 0.4, "w") for t in np.arange(0, 10, 0.5)])
    labels = [1 if w["start"] < 5 else 0 for w in words]  # voice 1 speaks first
    scores = speakers.voice_face_scores(words, labels, 2, activity, fps)
    assert speakers.match_voices(scores) == [1, 0]


def test_absent_frames_do_not_count_as_movement():
    activity = speakers.mouth_activity([[None] * 10 + [0.1, 0.3] * 5])
    assert np.all(activity[0, :10] == 0)


def test_resegment_never_crosses_a_speaker_change_and_splits_sentences():
    words = _words([(0, 1, "Hi,"), (1, 2, "I'm"), (2, 3.2, "Seamus."), (3.3, 3.6, "And"), (3.6, 4.0, "I'm"), (4.0, 4.5, "Chris.")])
    who = ["S0", "S0", "S0", "S1", "S1", "S1"]
    segs = speakers.resegment(words, who)
    assert [s["speaker"] for s in segs] == ["S0", "S1"]
    assert segs[0]["text"] == "Hi, I'm Seamus." and segs[1]["text"] == "And I'm Chris."
    assert segs[1]["start"] == 3.3 and segs[1]["end"] == 4.5


def test_resegment_caps_long_monologues_at_a_clause():
    words = _words([(i, i + 0.9, f"w{i}," if i == 8 else f"w{i}") for i in range(20)])
    segs = speakers.resegment(words, ["S0"] * 20, max_seconds=15.0)
    assert len(segs) == 2
    assert segs[0]["words"][-1]["text"] == "w8,"
    assert all(s["end"] - s["start"] <= 15.0 for s in segs)


def test_analyze_ties_voices_to_faces(monkeypatch):
    rng = np.random.default_rng(0)
    a, b = np.zeros(512), np.zeros(512)
    a[0], b[1] = 1, 1
    times = np.arange(0.75, 10, 0.5)
    emb = np.stack([(a if t < 5 else b) + 0.05 * rng.standard_normal(512) for t in times])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    monkeypatch.setattr(speakers, "embed_windows", lambda *a, **k: (times, emb))
    frames = 250
    still = [0.05] * frames
    moving = [0.05 + (0.2 if (i // 3) % 2 else 0.0) for i in range(frames)]
    people = {"people": [{"id": 0, "presence": 1.0}, {"id": 1, "presence": 1.0}],
              "mouth": [still[:125] + moving[125:], moving[:125] + still[125:]]}
    words = _words([(t, t + 0.4, "w.") for t in np.arange(0.2, 9.8, 0.5)])
    transcript = {"language": "en", "segments": [{"start": 0.2, "end": 9.8, "text": "x", "words": words}]}
    out = speakers.analyze("audio.wav", transcript, "video.mp4", people)
    # the first voice (S0) speaks while person 1's mouth moves
    assert out["speakers"][0]["id"] == "S0" and out["speakers"][0]["face_identity"] == 1
    assert out["speakers"][1]["face_identity"] == 0
    assert {s["speaker"] for s in out["segments"] if s["end"] <= 5} == {"S0"}
    assert speakers.needs_span_render(out)
    assert speakers.face_map(out) == {"S0": 1, "S1": 0}


def test_single_voice_on_a_two_person_clip_still_picks_the_talking_face(monkeypatch):
    rng = np.random.default_rng(1)
    times = np.arange(0.75, 10, 0.5)
    emb = np.stack([np.eye(512)[0] + 0.02 * rng.standard_normal(512) for _ in times])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    monkeypatch.setattr(speakers, "embed_windows", lambda *a, **k: (times, emb))
    frames = 250
    moving = [0.05 + (0.2 if (i // 3) % 2 else 0.0) for i in range(frames)]
    people = {"people": [{"id": 0, "presence": 1.0}, {"id": 1, "presence": 0.9}], "mouth": [[0.05] * frames, moving]}
    words = _words([(t, t + 0.4, "w") for t in np.arange(0.2, 9.8, 0.5)])
    out = speakers.analyze("a.wav", {"segments": [{"start": 0.2, "end": 9.8, "text": "x", "words": words}]}, "v.mp4", people)
    assert len(out["speakers"]) == 1 and out["speakers"][0]["face_identity"] == 1


def test_no_word_timings_leaves_the_transcript_alone():
    t = {"segments": [{"start": 0, "end": 1, "text": "hi"}]}
    assert speakers.analyze("a.wav", t, "v.mp4", None) is t


def test_spans_split_at_speaker_changes():
    segs = [{"start": 0.0, "end": 2.0, "speaker": "S0"}, {"start": 2.3, "end": 4.0, "speaker": "S1"},
            {"start": 4.2, "end": 5.0, "speaker": "S1"}]
    spans = streaming.speech_spans(segs, 10.0, gap=0.6, pad=0.25)
    assert [s.speaker for s in spans] == ["S0", "S1"]
    assert spans[0].segments == [0] and spans[1].segments == [1, 2]
    # padded spans meet at the frame midway between 2.0 and 2.3
    assert spans[0].end == pytest.approx(2.16) and spans[1].start == pytest.approx(2.16)


def test_spans_of_one_speaker_merge_as_before():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 2.3, "end": 4.0}]
    spans = streaming.speech_spans(segs, 10.0)
    assert len(spans) == 1 and spans[0].segments == [0, 1]


def test_turn_boundary_snaps_to_the_sentence_start():
    words = _words([(0, 1, "Chris"), (1, 2, "Branch."), (2, 2.2, "I"), (2.2, 3, "invited"), (3, 4, "Chris.")])
    # the window vote put "I" with the previous speaker
    assert speakers.snap_to_sentences(words, [1, 1, 1, 0, 0]) == [1, 1, 0, 0, 0]
    # a change mid-sentence with no sentence start nearby is left alone
    words2 = _words([(0, 1, "and"), (1, 2, "so"), (2, 3, "on"), (3, 4, "we"), (4, 5, "go.")])
    assert speakers.snap_to_sentences(words2, [1, 1, 1, 0, 0]) == [1, 1, 1, 0, 0]
