import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
import pytest
import httpx
from fastapi import HTTPException
from app import main
from app.playback import Playback, AudioOutput, VideoOutput


@pytest.fixture(autouse=True)
def sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_sessions", {})
    monkeypatch.setattr(main, "INGEST_DIR", tmp_path)


@pytest.mark.asyncio
async def test_stop_retry_preserves_recording_and_idempotency(tmp_path, monkeypatch):
    path = tmp_path / "input.mp4"
    path.write_bytes(b"media")
    recorder = SimpleNamespace(stop=AsyncMock())
    pc = SimpleNamespace(close=AsyncMock())
    session = main.Session("a" * 12, "es", "quality", None, pc, recorder, path)
    main._sessions[session.session_id] = session
    responses = [
        httpx.ConnectError("down"),
        httpx.Response(201, json={"job_id": "b" * 32}),
    ]
    posted = []

    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kw):
            posted.append(kw["data"].copy())
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(main.httpx, "AsyncClient", Client)
    with pytest.raises(HTTPException) as failure:
        await main.stop(session.session_id)
    assert failure.value.status_code == 502 and path.exists()
    result = await main.stop(session.session_id)
    assert result == await main.stop(session.session_id)
    assert len(posted) == 2 and posted[0]["request_id"] == posted[1]["request_id"]
    assert posted[0]["mode"] == "quality"
    recorder.stop.assert_awaited_once()
    pc.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejects_malformed_session_identifier():
    with pytest.raises(HTTPException) as error:
        await main.get_session("../other")
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_avatar_shared_clock_and_interrupt(monkeypatch):
    import app.playback as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    playback = Playback(np.zeros((16, 16, 3), dtype=np.uint8))
    playback.clips.append(
        (
            0,
            1,
            np.full(48000, 1200, dtype=np.int16),
            np.full((25, 16, 16, 3), 80, dtype=np.uint8),
        )
    )
    a, v = AudioOutput(playback), VideoOutput(playback)
    first_audio = await a.recv()
    first_video = await v.recv()
    assert (
        float(first_audio.pts * first_audio.time_base)
        == float(first_video.pts * first_video.time_base)
        == 0
    )
    assert np.all(first_audio.to_ndarray() == 1200)
    assert np.all(first_video.to_ndarray(format="bgr24") == 80)
    # A stalled consumer skips old packets and remains aligned on the shared clock.
    clock[0] = 100.4
    audio, video = await a.recv(), await v.recv()
    assert (
        abs(float(audio.pts * audio.time_base) - float(video.pts * video.time_base))
        <= 0.04
    )
    playback.interrupt()
    clock[0] = 100.8
    assert np.all((await a.recv()).to_ndarray() == 0)
    assert np.all((await v.recv()).to_ndarray(format="bgr24") == 0)


@pytest.mark.asyncio
async def test_avatar_queue_backpressure_can_be_interrupted():
    playback = Playback(np.zeros((16, 16, 3), dtype=np.uint8))
    audio = np.zeros(48000, dtype=np.int16)
    frames = np.zeros((25, 16, 16, 3), dtype=np.uint8)
    for _ in range(3):
        await playback.enqueue(audio, frames, 0)
    blocked = asyncio.create_task(playback.enqueue(audio, frames, 0))
    await asyncio.sleep(0.03)
    assert not blocked.done()
    playback.interrupt()
    await asyncio.wait_for(blocked, 0.2)
    assert not playback.clips


@pytest.mark.asyncio
async def test_offer_with_missing_tracks_closes_peer(monkeypatch):
    pc = SimpleNamespace(
        on=lambda event: lambda fn: fn,
        setRemoteDescription=AsyncMock(),
        close=AsyncMock(),
    )
    recorder = SimpleNamespace(stop=AsyncMock())
    monkeypatch.setattr(main, "RTCPeerConnection", lambda *a: pc)
    monkeypatch.setattr(main, "MediaRecorder", lambda *a: recorder)
    with pytest.raises(HTTPException) as error:
        await main.offer("c" * 12, main.OfferIn(sdp="invalid", target_language="es"))
    assert error.value.status_code == 422
    pc.close.assert_awaited_once()
    assert main._sessions["c" * 12].stopped
