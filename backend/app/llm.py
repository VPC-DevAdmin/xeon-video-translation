"""OpenAI-compatible chat client.

On the GPU host this talks to the vLLM server (Qwen3-30B-A3B-Instruct) over
its ``/v1`` API. Any server that speaks the OpenAI chat-completions protocol
works, so the same code covers a local Ollama ``/v1`` endpoint or a hosted
model. Translation, overrun rewriting and the avatar assistant all route
through here.

Secrets never travel through ``.env``: an optional API key is read from the
file named by ``LLM_API_KEY_FILE`` (mounted read-only into the container).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The chat endpoint is unconfigured, unreachable or returned garbage."""


def configured() -> bool:
    return bool(settings.llm_base_url)


def _api_key() -> str:
    path = settings.llm_api_key_file
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _request(payload: dict, timeout: float) -> urllib.request.Request:
    if not configured():
        raise LLMError("LLM_BASE_URL is not configured")
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return urllib.request.Request(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float | None = None,
) -> str:
    """Return the assistant message for `messages` (non-streaming)."""
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = _request(payload, timeout or settings.llm_timeout_seconds)
    try:
        with urllib.request.urlopen(request, timeout=timeout or settings.llm_timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"LLM request failed: HTTP {e.code} {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"LLM request failed: {e}") from e
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"LLM response malformed: {json.dumps(body)[:300]}") from e


def stream(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 160,
    timeout: float = 30,
) -> Iterator[str]:
    """Yield content deltas from a streaming chat completion (SSE)."""
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    request = _request(payload, timeout)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = json.loads(data)
                if event.get("error"):
                    raise LLMError(str(event["error"]))
                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"LLM stream failed: HTTP {e.code} {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"LLM stream failed: {e}") from e
