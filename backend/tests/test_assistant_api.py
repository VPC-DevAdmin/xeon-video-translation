"""Sentence splitting and PCM events for the plan-then-speak assistant endpoint."""

import base64

import numpy as np

import app.api.assistant as assistant
from app.api.assistant import pcm_event, split_sentences


def test_split_sentences_keeps_units_and_cuts_run_ons_at_clauses():
    text = "Sure thing. Here is what I found! Does that help?"
    assert split_sentences(text) == ["Sure thing. Here is what I found! Does that help?"]   # short units merge
    assert split_sentences("Good morning! It's great to see you all here today, really.") == ["Good morning! It's great to see you all here today, really."]
    long = ", ".join(["clause number %d is here" % i for i in range(12)]) + "."
    parts = split_sentences(long, max_len=80)
    assert len(parts) > 1 and all(len(p) <= 81 for p in parts)
    assert " ".join(parts).replace(" ,", ",") .startswith("clause number 0")


def test_pcm_event_is_int16_base64_with_clipping():
    event = pcm_event(np.array([0.0, 0.5, 2.0, -3.0], dtype=np.float32), 3, True, "hi")
    pcm = np.frombuffer(base64.b64decode(event["pcm_b64"]), dtype=np.int16)
    assert pcm.tolist() == [0, 16383, 32767, -32767]
    assert (event["samples"], event["sentence_id"], event["final"], event["sample_rate"]) == (4, 3, True, 24000)


class _Word:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _Transcript:
    def __init__(self, words):
        self.segments = [type("Seg", (), {"words": words})()]
        self.text = " ".join(w.text for w in words)


def test_verified_sentence_keeps_the_word_span_and_falls_back_to_the_stock_voice(monkeypatch):
    """Three cloned takes that do not carry the words: the stock voice says the sentence
    once; without a stock voice the best take is returned cut to its words, not clipped
    to a duration estimate."""
    import app.pipeline.tts as tts_module
    sentence = "The meeting is at three tomorrow."
    used = []

    class Model:
        def inference(self, text, lang, latent, embedding, **kw):
            used.append(latent)
            return {"wav": np.ones(24000 * 6, dtype=np.float32) * 0.1}      # a 6 s take

    def fake_transcribe(wav, json_path, language):
        if used[-1] == "clone":
            return _Transcript([_Word("blah", 0.5, 0.9), _Word("blah", 1.0, 1.4)])
        return _Transcript([_Word(w, 1.0 + 0.3 * i, 1.2 + 0.3 * i) for i, w in enumerate(sentence.rstrip(".").split())])

    monkeypatch.setattr(tts_module, "_trim_tail_via_whisper", lambda *a, **k: False)
    monkeypatch.setattr(assistant.transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(assistant.tts, "XTTS_LANG_CODES", {"en": "en"}, raising=False)
    clone = {"gpt_cond_latent": "clone", "speaker_embedding": None}
    stock = {"gpt_cond_latent": "stock", "speaker_embedding": None}

    audio, match, heard, takes, fallback = assistant._verified_sentence(Model(), clone, sentence, "en", fallback_conditioning=stock)
    assert fallback is True and takes == 4 and match == 1.0 and heard.startswith("The meeting")
    # cut to the recognized span: first word start 1.0 - 0.3 pad .. last word end (1.2 + 0.3*5) + 0.35 pad
    assert abs(len(audio) / 24000 - ((2.7 + 0.35) - 0.7)) < 0.02

    used.clear()
    audio, match, heard, takes, fallback = assistant._verified_sentence(Model(), clone, sentence, "en")
    assert fallback is False and takes == 3 and match == 0.0
    assert abs(len(audio) / 24000 - ((1.4 + 0.35) - 0.2)) < 0.02         # the babble's own word span, not 0.09*len+1.5


def test_alignment_accepts_contractions_and_interjection_spellings():
    words = [_Word("I'm", 0.2, 0.4), _Word("going", 0.4, 0.6), _Word("to", 0.6, 0.7), _Word("look", 0.7, 0.9), _Word("that", 0.9, 1.0), _Word("up.", 1.0, 1.3)]
    matched, first, last = assistant._aligned_span("I am going to look that up.", words)
    assert matched == 1.0 and (first, last) == (0.2, 1.3)
    matched, _, _ = assistant._aligned_span("Mm-hmm, getting closer.", [_Word("Mmm,", 0.1, 0.5), _Word("getting", 0.5, 0.8), _Word("closer.", 0.8, 1.2)])
    assert matched == 1.0
    matched, _, _ = assistant._aligned_span("Alright, here we go.", [_Word("All", 0.1, 0.2), _Word("right,", 0.2, 0.4), _Word("here", 0.4, 0.5), _Word("we", 0.5, 0.6), _Word("go.", 0.6, 0.8)])
    assert matched == 1.0


def test_sentences_get_a_pause_and_a_fade():
    audio = assistant.with_pause(np.ones(2400, np.float32), "Is that right?")
    assert len(audio) == 2400 + int((assistant.SENTENCE_PAUSE + 0.2) * 24000)
    assert audio[-1] == 0 and audio[2399] == 0 and 0 < audio[2200] < 1 and audio[0] == 1
    assert len(assistant.with_pause(np.ones(2400, np.float32), "Fine.")) == 2400 + int(assistant.SENTENCE_PAUSE * 24000)
