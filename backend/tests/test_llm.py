"""Shared OpenAI-compatible chat client: request shape, errors, SSE parsing."""

import io
import json
import urllib.error

import pytest

from app import llm
from app.config import settings


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch, body):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return _Response(body if isinstance(body, bytes) else json.dumps(body).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_unconfigured_raises(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "")
    assert not llm.configured()
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_posts_openai_payload_and_returns_content(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_base_url", "http://llm.test/v1/")
    monkeypatch.setattr(settings, "llm_model", "test-model")
    key = tmp_path / "key"
    key.write_text("s3cret\n")
    monkeypatch.setattr(settings, "llm_api_key_file", str(key))
    calls = _capture(monkeypatch, {"choices": [{"message": {"content": "  Hola  "}}]})

    assert llm.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=9) == "Hola"
    request, _ = calls[0]
    assert request.full_url == "http://llm.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer s3cret"
    payload = json.loads(request.data)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 9
    assert payload["temperature"] == 0.1


def test_chat_http_error_becomes_llm_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "http://llm.test/v1")

    def fail(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 503, "down", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    with pytest.raises(llm.LLMError, match="503"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_stream_yields_deltas_until_done(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "http://llm.test/v1")
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " there."}}]},
    ]
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    body += b"data: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\n"
    _capture(monkeypatch, body)
    assert list(llm.stream([{"role": "user", "content": "hi"}])) == ["Hello", " there."]
