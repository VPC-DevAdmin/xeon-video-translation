import json
from pathlib import Path

import pytest

from app.pipeline import translate as t


@pytest.fixture
def transcript():
    return {"language": "en", "segments": [{"start": 0, "end": 2, "text": "Do not ship 12 boxes."}]}


def test_review_corrects_negation_and_writes_audit(tmp_path, monkeypatch, transcript):
    monkeypatch.setattr(t.llm, "configured", lambda: True)
    replies = iter(
        [
            "Envíe 12 cajas.",
            json.dumps(
                {
                    "translation": "No envíe 12 cajas.",
                    "changes": ["Restored negation."],
                    "unresolved_issues": [],
                }
            ),
        ]
    )
    monkeypatch.setattr(t.llm, "chat", lambda *a, **k: next(replies))
    output = tmp_path / "translation.json"
    result = t.translate(transcript, output, "es", backend_override="llm", quality_review=True)
    assert result.text == "No envíe 12 cajas."
    audit = json.loads(output.with_suffix(".review.json").read_text())
    assert audit["segments"][0]["draft"] == "Envíe 12 cajas."
    assert "not independent" in audit["method"]


@pytest.mark.parametrize(
    "decision",
    [
        {"translation": "No envíe 13 cajas.", "changes": [], "unresolved_issues": []},
        {
            "translation": "No envíe 12 cajas.",
            "changes": [],
            "unresolved_issues": ["Ambiguous speaker intent."],
        },
    ],
)
def test_unresolved_review_fails_with_saved_audit(tmp_path, monkeypatch, transcript, decision):
    monkeypatch.setattr(t.llm, "configured", lambda: True)
    replies = iter(["No envíe 12 cajas.", json.dumps(decision)])
    monkeypatch.setattr(t.llm, "chat", lambda *a, **k: next(replies))
    out = tmp_path / "translation.json"
    with pytest.raises(t.TranslationError, match="unresolved"):
        t.translate(transcript, out, "es", backend_override="llm", quality_review=True)
    assert out.with_suffix(".review.json").exists()
    assert json.loads(out.read_text())["segments"][0]["text"] == decision["translation"]


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "[]",
        '{"translation":""}',
        '{"translation":"hola","changes":"bad","unresolved_issues":[]}',
    ],
)
def test_malformed_review_is_not_accepted(monkeypatch, raw):
    monkeypatch.setattr(t.llm, "chat", lambda *a, **k: raw)
    with pytest.raises(t.TranslationError, match="review failed"):
        t._review_translation("Hello", "Hola", "en", "es", "", {})


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"translation": "Hola", "changes": [], "unresolved_issues": []}\n```',
        'Here is the review:\n{"translation": "Hola", "changes": [], "unresolved_issues": [], "confidence": 0.9}',
        '{"translation": "Hola"}',
        '{"translation": "Hola", "changes": ["minor wording"], "unresolved_issues": null}',
    ],
)
def test_review_accepts_fenced_extra_key_or_sparse_replies(raw):
    decision = t._parse_review(raw)
    assert decision["translation"] == "Hola"
    assert isinstance(decision["changes"], list) and isinstance(decision["unresolved_issues"], list)


def test_review_audit_written_once_per_job(tmp_path, monkeypatch):
    monkeypatch.setattr(t.llm, "configured", lambda: True)
    transcript = {"language": "en", "segments": [
        {"start": 0, "end": 1, "text": "One."}, {"start": 1, "end": 2, "text": "Two."}, {"start": 2, "end": 3, "text": "Three."}]}
    replies = iter(["Uno.", json.dumps({"translation": "Uno.", "changes": [], "unresolved_issues": []}),
                    "Dos.", json.dumps({"translation": "Dos.", "changes": [], "unresolved_issues": []}),
                    "Tres.", json.dumps({"translation": "Tres.", "changes": [], "unresolved_issues": []})])
    monkeypatch.setattr(t.llm, "chat", lambda *a, **k: next(replies))
    writes = []
    original = Path.write_text

    def counting(self, data, *a, **k):
        if self.name.endswith(".review.json"):
            writes.append(self)
        return original(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", counting)
    out = tmp_path / "translation.json"
    t.translate(transcript, out, "es", backend_override="llm", quality_review=True)
    assert len(writes) == 1
    assert len(json.loads(out.with_suffix(".review.json").read_text())["segments"]) == 3
