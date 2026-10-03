import asyncio
import base64
import hashlib
import hmac
import json
import time
import numpy as np
import pytest
from app import main, avatar
from app.identity import owner, require
from app.preview import TranscriptPreview
from app.vad import Detector
from fastapi import HTTPException


def test_turn_credentials_expire_and_are_owner_bound(monkeypatch):
    monkeypatch.setenv("ICE_SERVERS_JSON", "[]")
    monkeypatch.setenv("TURN_SHARED_SECRET", "secret")
    monkeypatch.setenv("TURN_URLS_JSON", '["turn:example.org:3478"]')
    token = owner.set("alice")
    try:
        config = main.ice_servers()[0]
    finally:
        owner.reset(token)
    expiry, name = config["username"].split(":")
    assert name == "alice" and 599 <= int(expiry) - time.time() <= 600
    assert (
        config["credential"]
        == base64.b64encode(
            hmac.new(b"secret", config["username"].encode(), hashlib.sha1).digest()
        ).decode()
    )


def test_preview_bounded_and_ownership(tmp_path):
    preview = TranscriptPreview(tmp_path, "http://unused", "en", lambda _: None)
    for i in range(1000):
        preview.push(np.zeros(320, dtype=np.int16))
    assert preview.size <= 128000
    session = avatar.Avatar("a", tmp_path, np.zeros((8, 8, 3), dtype=np.uint8), "en")
    session.owner = "alice"
    with pytest.raises(HTTPException):
        require(session)


def test_acknowledgments_only_commit_current_played_sentences(tmp_path):
    session = avatar.Avatar("a", tmp_path, np.zeros((8, 8, 3), dtype=np.uint8), "en")
    assistant = {"role": "assistant", "content": ""}
    session.history = [assistant]
    session.turn_history = {0: {"assistant": assistant, "sentences": {}}}
    session.pending_played = {
        "a": dict(generation=0, sentence=0, text="Heard."),
        "b": dict(generation=0, sentence=1, text="Not heard."),
    }
    session.acknowledge("a")
    assert assistant["content"] == "Heard."
    session.interrupt()
    session.acknowledge("b")
    assert assistant["content"] == "Heard."


def test_energy_vad_and_invalid_configuration(monkeypatch):
    monkeypatch.setenv("AVATAR_VAD_BACKEND", "energy")
    detector = Detector()
    assert not detector.speech(np.zeros(320, dtype=np.int16))
    assert detector.speech(np.full(320, 1000, dtype=np.int16))
    monkeypatch.setenv("AVATAR_VAD_BACKEND", "unsupported")
    with pytest.raises(ValueError):
        Detector()


def test_silero_real_onnx_when_configured(monkeypatch):
    import os

    path = os.getenv("SILERO_TEST_MODEL")
    if not path:
        pytest.skip("Set SILERO_TEST_MODEL to the verified downloaded model")
    monkeypatch.setenv("AVATAR_VAD_BACKEND", "silero")
    monkeypatch.setenv("SILERO_VAD_MODEL", path)
    detector = Detector()
    for _ in range(100):
        assert not detector.speech(np.zeros(320, dtype=np.int16))
    assert detector.pending.size < 512 and detector.state.shape == (2, 1, 128)


@pytest.mark.asyncio
async def test_avatar_storage_limit_stops_before_backend(tmp_path, monkeypatch):
    (tmp_path / "existing.npy").write_bytes(b"x" * 1025)
    monkeypatch.setenv("AVATAR_MAX_STORAGE_MB", "0")
    session = avatar.Avatar("a", tmp_path, np.zeros((8, 8, 3), dtype=np.uint8), "en")
    events = []
    session.notify = lambda kind, **values: events.append((kind, values))
    await session.reply(np.zeros(16000, dtype=np.int16), 0)
    assert events[0][0] == "error" and not list(tmp_path.glob("*/input.wav"))


@pytest.mark.asyncio
@pytest.mark.parametrize("render_fails", [False, True])
async def test_avatar_reply_fallback_ack_and_artifact_cleanup(
    tmp_path, monkeypatch, render_fails
):
    import httpx, wave

    class Response:
        def __init__(self, events=()):
            self.events = events

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for event in self.events:
                yield json.dumps(event)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, *args, **kwargs):
            directory = __import__("pathlib").Path(kwargs["json"]["audio_path"]).parent
            output = directory / "reply.wav"
            with wave.open(str(output), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(np.ones(12000, dtype=np.int16).tobytes())
            return Response(
                [
                    {"type": "transcript", "text": "Hello"},
                    {
                        "type": "audio",
                        "path": str(output),
                        "sentence_id": 0,
                        "text": "Welcome.",
                        "final": True,
                    },
                ]
            )

        async def post(self, *args, **kwargs):
            if render_fails:
                raise httpx.ConnectError("renderer down")
            np.save(
                kwargs["json"]["output_path"], np.zeros((13, 8, 8, 3), dtype=np.uint8)
            )
            return Response()

    monkeypatch.setattr(avatar.httpx, "AsyncClient", Client)
    session = avatar.Avatar("a", tmp_path, np.zeros((8, 8, 3), dtype=np.uint8), "en")
    events = []
    session.notify = lambda kind, **values: events.append((kind, values))
    await session.reply(np.ones(16000, dtype=np.int16), 0)
    assert session.metrics["render_fallbacks"] == int(render_fails)
    assert session.playback.clips and session.history[-1]["content"] == ""
    token = next(v["token"] for kind, v in events if kind == "playout")
    session.acknowledge(token)
    assert session.history[-1]["content"] == "Welcome."
    if not render_fails:
        assert not list(tmp_path.rglob("reply*"))
