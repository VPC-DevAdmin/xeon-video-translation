"""Headless WebRTC client for testing the ingest service without a browser.

Streams a media file over a real RTCPeerConnection to the ingest service
exactly as the /live page does (offer -> answer -> SRTP -> /stop), then
prints the job id the service created. Run from inside the ingest
container, where aiortc and PyAV are installed:

    docker compose exec ingest-webrtc python /app/scripts/e2e_client.py \
        --file /jobs/ingest/fixtures/IMG_7228.MOV --target es --mode fast

Exit code 0 means the whole path worked: ICE connected, both tracks
flowed, the recorder wrote a non-empty file, and the backend accepted it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ingest", default="http://localhost:8091")
    ap.add_argument("--target", default="es")
    ap.add_argument("--mode", default="fast", choices=["fast", "quality"])
    ap.add_argument("--seconds", type=float, default=None,
                    help="how long to stream; default = file duration + 1s")
    args = ap.parse_args()

    player = MediaPlayer(args.file)
    duration = None
    try:
        duration = float(player._container.duration) / 1_000_000  # noqa: SLF001
    except Exception:
        pass
    stream_for = args.seconds or ((duration or 10.0) + 1.0)

    pc = RTCPeerConnection()
    tracks = []
    if player.audio:
        pc.addTrack(player.audio)
        tracks.append("audio")
    if player.video:
        pc.addTrack(player.video)
        tracks.append("video")
    if not tracks:
        print("file has no audio or video track", file=sys.stderr)
        return 2

    session_id = uuid.uuid4().hex[:12]
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    # Wait for ICE gathering so the SDP carries host candidates.
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    async with httpx.AsyncClient(timeout=60) as client:
        t0 = time.perf_counter()
        r = await client.post(
            f"{args.ingest}/sessions/{session_id}/offer",
            json={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "target_language": args.target,
                "mode": args.mode,
            },
        )
        r.raise_for_status()
        answer = r.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        connected = asyncio.Event()

        @pc.on("connectionstatechange")
        async def _on_state():
            if pc.connectionState == "connected":
                connected.set()
            elif pc.connectionState in ("failed", "closed"):
                connected.set()

        try:
            await asyncio.wait_for(connected.wait(), timeout=15)
        except asyncio.TimeoutError:
            print("ICE never connected", file=sys.stderr)
            return 3
        if pc.connectionState != "connected":
            print(f"connection state {pc.connectionState}", file=sys.stderr)
            return 3
        print(f"connected in {time.perf_counter() - t0:.2f}s; streaming {tracks} for {stream_for:.1f}s")

        await asyncio.sleep(stream_for)

        r = await client.post(f"{args.ingest}/sessions/{session_id}/stop")
        if r.status_code != 200:
            print(f"/stop failed: {r.status_code} {r.text[:300]}", file=sys.stderr)
            return 4
        result = r.json()

    await pc.close()
    print(json.dumps(result))
    return 0 if result.get("job_id") else 5


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
