"""Sentence splitting and PCM events for the plan-then-speak assistant endpoint."""

import base64

import numpy as np

from app.api.assistant import pcm_event, split_sentences


def test_split_sentences_keeps_units_and_cuts_run_ons_at_clauses():
    text = "Sure thing. Here is what I found! Does that help?"
    assert split_sentences(text) == ["Sure thing.", "Here is what I found!", "Does that help?"]
    long = ", ".join(["clause number %d is here" % i for i in range(12)]) + "."
    parts = split_sentences(long, max_len=80)
    assert len(parts) > 1 and all(len(p) <= 81 for p in parts)
    assert " ".join(parts).replace(" ,", ",") .startswith("clause number 0")


def test_pcm_event_is_int16_base64_with_clipping():
    event = pcm_event(np.array([0.0, 0.5, 2.0, -3.0], dtype=np.float32), 3, True, "hi")
    pcm = np.frombuffer(base64.b64decode(event["pcm_b64"]), dtype=np.int16)
    assert pcm.tolist() == [0, 16383, 32767, -32767]
    assert (event["samples"], event["sentence_id"], event["final"], event["sample_rate"]) == (4, 3, True, 24000)
