"""Real local ICE/DTLS/SRTP recording test; no camera or GPU required."""

import asyncio
import av
import pytest
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    AudioStreamTrack,
    VideoStreamTrack,
)
from app import main


@pytest.mark.asyncio
async def test_real_webrtc_audio_video_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "INGEST_DIR", tmp_path)
    monkeypatch.setattr(main, "_sessions", {})
    monkeypatch.setenv("ICE_SERVERS_JSON", "[]")
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    client.addTrack(AudioStreamTrack())
    client.addTrack(VideoStreamTrack())
    identifier = "d" * 32
    try:
        await client.setLocalDescription(await client.createOffer())
        answer = await main.offer(
            identifier,
            main.OfferIn(
                sdp=client.localDescription.sdp, type="offer", target_language="es"
            ),
        )
        await client.setRemoteDescription(
            RTCSessionDescription(**{k: answer[k] for k in ("sdp", "type")})
        )

        async def connected():
            while client.connectionState != "connected":
                if client.connectionState == "failed":
                    raise AssertionError("ICE failed")
                await asyncio.sleep(0.02)

        await asyncio.wait_for(connected(), 10)
        await asyncio.sleep(1)
        session = main._sessions[identifier]
        await main.close_recording(session)
        with av.open(str(session.input_path)) as media:
            assert {s.type for s in media.streams} == {"audio", "video"}
            assert sum(1 for _ in media.decode(video=0)) >= 10
        with av.open(str(session.input_path)) as media:
            assert sum(f.samples for f in media.decode(audio=0)) > 16000
    finally:
        await client.close()
        if identifier in main._sessions:
            await main.discard(identifier)


@pytest.mark.asyncio
async def test_avatar_returns_live_audio_and_video(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from starlette.datastructures import UploadFile
    from app import avatar

    monkeypatch.setattr(avatar, "ROOT", tmp_path)
    monkeypatch.setattr(avatar, "_sessions", {})
    monkeypatch.setenv("ICE_SERVERS_JSON", "[]")
    image = BytesIO()
    Image.new("RGB", (64, 64), (40, 80, 120)).save(image, format="PNG")
    image.seek(0)
    created = await avatar.create(UploadFile(image, filename="portrait.png"), "en")
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    client.addTrack(AudioStreamTrack())
    client.addTransceiver("video", direction="recvonly")
    client.createDataChannel("events")
    received = {}
    tasks = []

    @client.on("track")
    def track(track):
        async def frame():
            received[track.kind] = await track.recv()

        tasks.append(asyncio.create_task(frame()))

    try:
        await client.setLocalDescription(await client.createOffer())
        answer = await avatar.offer(
            created["session_id"], avatar.Offer(sdp=client.localDescription.sdp)
        )
        await client.setRemoteDescription(RTCSessionDescription(**answer))
        await asyncio.wait_for(asyncio.gather(*tasks), 10)
        assert set(received) == {"audio", "video"}
        assert received["video"].width == 64
        assert received["audio"].samples > 0
    finally:
        await client.close()
        await avatar.delete(created["session_id"])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not avatar._sessions
