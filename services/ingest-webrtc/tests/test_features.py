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
async def test_assistant_reply_renders_a_real_silent_tail_across_chunk_boundaries(tmp_path):
    from app.assistant import Assistant

    class Renderer:
        samples = 16000

        def __init__(self):
            self.calls = []

        async def render(self, pcm, reset, generation, motion):
            self.calls.append((len(pcm), reset))
            values = [100 if np.any(pcm[i:i + 640]) else 0 for i in range(0, len(pcm), 640)]
            return np.asarray(values, np.uint8)[:, None, None, None].repeat(3, axis=3), 0.0

    session = Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    session.renderer = Renderer()
    session.timeline.add_idle(np.zeros((25, 1, 1, 3), np.uint8))
    try:
        frames, audio, chunks = await session.render_reply_piece(np.ones(12000, np.int16), True, 0, True)
        assert chunks == 2 and session.renderer.calls == [(16000, True), (4800, False)]
        assert set(np.unique(frames)) == {0, 100} and np.all(frames[-1] == 0)
        assert len(audio) == len(frames) * 48000 // 25 and np.all(audio[-4800:] == 0)
        tail, silence, chunks = await session.render_reply_piece(np.zeros(0, np.int16), True, 0, False)
        assert chunks == 1 and session.renderer.calls[-1] == (12800, False)
        assert np.all(tail == 0) and np.all(silence == 0)
    finally:
        await session.client.aclose()


@pytest.mark.asyncio
async def test_assistant_idle_replacement_keeps_old_video_visible_while_rendering(tmp_path):
    from app.assistant import Assistant

    class Renderer:
        samples = 16000
        motion = "reply:front"

        def __init__(self):
            self.lock = asyncio.Lock()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def render_locked(self, pcm, reset, generation, motion, pose):
            self.started.set()
            await self.release.wait()
            self.motion = motion
            return np.full((25, 1, 1, 3), 90, np.uint8), 0.0

    session = Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    session.renderer = Renderer()
    session.timeline.add_idle(np.full((25, 1, 1, 3), 20, np.uint8))
    try:
        task = asyncio.create_task(session.grow_idle("front", 1.0))
        await asyncio.wait_for(session.renderer.started.wait(), 1)
        assert session.timeline.loop("front").ready
        assert int(session.timeline.frame_at(0)[0][0, 0, 0]) == 20
        session.renderer.release.set()
        await asyncio.wait_for(task, 1)
        assert int(session.timeline.loop("front").frames[0, 0, 0, 0]) == 90
    finally:
        await session.client.aclose()


@pytest.mark.asyncio
async def test_assistant_rotates_thinking_and_lookup_openers(tmp_path, monkeypatch):
    from app import assistant as module

    monkeypatch.setattr(module, "PROGRESS_FILLERS", True)
    session = module.Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    for kind in ("opener_think", "opener_lookup"):
        session.clips[kind].append(module.Clip(kind, kind, np.zeros(1920, np.int16),
                                                np.zeros((1, 1, 1, 3), np.uint8), "front"))
    try:
        modes = [session.pick_opener()[1] for _ in range(4)]
        assert set(modes) == {"think", "lookup"}
        assert all(a != b for a, b in zip(modes, modes[1:]))
    finally:
        await session.client.aclose()


@pytest.mark.asyncio
async def test_assistant_keeps_thinking_pose_until_the_final_acknowledgement(tmp_path):
    from app.assistant import Assistant, Clip

    session = Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    session.timeline.add_idle(np.zeros((25, 1, 1, 3), np.uint8))
    session.timeline.set_mode(3.05, "thinking", session.timeline.generation)
    session.clips["closer"].append(Clip("closer", "Okay, got it.", np.zeros(48000, np.int16),
                                        np.zeros((25, 1, 1, 3), np.uint8), "front"))
    try:
        await session.fill_gap({"opener_end": 3.0, "mode": "think"}, {"reply_start": 9.0}, 0, 0.0)
        assert session.timeline.mode_at(7.8) == "thinking"
        assert session.timeline.mode_at(7.85) == "front"
        closer = next(c for c in session.timeline.clips if c[4] == "filler")
        assert closer[0] == pytest.approx(7.85) and closer[1] == pytest.approx(8.85)
    finally:
        await session.client.aclose()


@pytest.mark.asyncio
async def test_assistant_handoff_preserves_a_scheduled_progress_phrase(tmp_path, monkeypatch):
    from app import assistant as module
    from app.assistant import Assistant, Clip

    monkeypatch.setattr(module, "FILLER_CUTOFF", False)
    session = Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    session.timeline.add_idle(np.zeros((25, 1, 1, 3), np.uint8))
    session.clips["closer"].append(Clip("closer", "Okay, got it.", np.zeros(48000, np.int16),
                                        np.zeros((25, 1, 1, 3), np.uint8), "front"))
    session.timeline.schedule(5.0, np.ones(96000, np.int16), np.full((50, 1, 1, 3), 70, np.uint8), 0, tag="filler")
    try:
        await session.fill_gap({"opener_end": 3.0, "mode": "think"}, {"reply_start": 10.0}, 0, 0.0)
        filler = next(c for c in session.timeline.clips if c[0] == 5.0)
        assert filler[1] == 7.0 and len(filler[3]) == 50
        closer = next(c for c in session.timeline.clips if c[0] > 7.0)
        assert closer[0] == pytest.approx(8.85) and closer[1] == pytest.approx(9.85)
    finally:
        await session.client.aclose()


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


@pytest.mark.asyncio
async def test_assistant_handoff_cuts_a_talking_progress_phrase_by_default(tmp_path, monkeypatch):
    from app import assistant as module
    from app.assistant import Assistant, Clip

    monkeypatch.setattr(module, "FILLER_CUTOFF", True)
    session = Assistant("a", tmp_path, np.zeros((1, 1, 3), np.uint8), "en", "test")
    session.timeline.add_idle(np.zeros((25, 1, 1, 3), np.uint8))
    session.clips["closer"].append(Clip("closer", "Okay, got it.", np.zeros(48000, np.int16),
                                        np.zeros((25, 1, 1, 3), np.uint8), "front"))
    session.timeline.schedule(5.0, np.ones(96000, np.int16), np.full((50, 1, 1, 3), 70, np.uint8), 0, tag="filler")   # 5.0 - 7.0
    try:
        await session.fill_gap({"opener_end": 3.0, "mode": "think"}, {"reply_start": 7.5}, 0, 0.0)
        filler = next(c for c in session.timeline.clips if c[0] == 5.0)
        assert filler[1] == pytest.approx(6.35) and len(filler[3]) == 34                 # cut at the hand-back, audio faded
        closer = next(c for c in session.timeline.clips if c[0] > 6.0)
        assert closer[0] == pytest.approx(6.35) and closer[1] == pytest.approx(7.35)
    finally:
        await session.client.aclose()
