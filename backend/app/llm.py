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
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from gpu_runtime import span
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The chat endpoint is unconfigured, unreachable or returned garbage."""


@lru_cache(maxsize=4)
def _limits(lane, concurrent, pending):
    return threading.BoundedSemaphore(concurrent), threading.BoundedSemaphore(concurrent + pending)


def _lane_limits(lane):
    if lane == "interactive":
        return settings.llm_interactive_max_concurrent, settings.llm_interactive_max_pending
    return settings.llm_max_concurrent, settings.llm_max_pending


@contextmanager
def _admission(lane="batch"):
    """Bound in-flight requests per lane. `batch` covers translation and
    rewrites; `interactive` covers avatar streams, which hold their slot for
    the whole reply and must not block batch work."""
    concurrent, pending = _lane_limits(lane)
    active, total = _limits(lane, concurrent, pending)
    if not total.acquire(blocking=False):
        raise LLMError("LLM queue is full")
    acquired = False
    try:
        with span("llm.queue"):
            acquired = active.acquire(timeout=settings.llm_queue_timeout_seconds)
        if not acquired:
            raise LLMError("LLM queue wait timed out")
        yield
    finally:
        if acquired:
            active.release()
        total.release()


def _validate(messages, max_tokens):
    if not 1 <= max_tokens <= settings.llm_max_output_tokens:
        raise LLMError("LLM output token limit exceeded")
    if sum(len(str(m.get("content", ""))) for m in messages) > settings.llm_max_input_chars:
        raise LLMError("LLM input limit exceeded; shorten the request")


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
    _validate(messages, max_tokens)
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = _request(payload, timeout or settings.llm_timeout_seconds)
    try:
        with (
            _admission(),
            span("llm.chat"),
            urllib.request.urlopen(
                request, timeout=timeout or settings.llm_timeout_seconds
            ) as resp,
        ):
            raw = resp.read(settings.llm_max_response_bytes + 1)
            if len(raw) > settings.llm_max_response_bytes:
                raise LLMError("LLM response size limit exceeded")
            body = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"LLM request failed: HTTP {e.code} {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
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
    _validate(messages, max_tokens)
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    request = _request(payload, timeout)
    try:
        with (
            _admission("interactive"),
            span("llm.stream"),
            urllib.request.urlopen(request, timeout=timeout) as resp,
        ):
            deadline = time.monotonic() + timeout
            size = 0
            while True:
                raw = resp.readline(settings.llm_max_response_bytes + 1)
                if not raw:
                    break
                size += len(raw)
                if size > settings.llm_max_response_bytes or time.monotonic() > deadline:
                    raise LLMError("LLM stream size or duration limit exceeded")
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
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise LLMError(f"LLM stream failed: {e}") from e


_ready_cache: dict = {}
_ready_lock = threading.Lock()


def ready(timeout: float = 3.0, ttl: float = 0.0) -> bool:
    """Check the configured served model without generating text or logging secrets.

    `ttl` > 0 reuses the last answer for that many seconds, so readiness
    polling does not issue a live request to the LLM server on every call."""
    if not configured():
        return False
    key = (settings.llm_base_url, settings.llm_model)
    if ttl > 0:
        with _ready_lock:
            cached = _ready_cache.get(key)
            if cached and time.monotonic() - cached[0] < ttl:
                return cached[1]
    result = _ready_uncached(timeout)
    if ttl > 0:
        with _ready_lock:
            _ready_cache[key] = (time.monotonic(), result)
    return result


def _ready_uncached(timeout: float) -> bool:
    headers = {}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{settings.llm_base_url.rstrip('/')}/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            return False
        body = json.loads(raw)
        return isinstance(body, dict) and isinstance(body.get("data"), list) and any(
            isinstance(model, dict) and model.get("id") == settings.llm_model
            for model in body["data"]
        )
    except (urllib.error.URLError, OSError, ValueError):
        return False
