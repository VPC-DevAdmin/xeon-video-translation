"""End-to-end assistant turn over a real WebRTC peer: acknowledgement, scheduled reply, cadence, interruption.

Run inside the ingest container:
  python /app/scripts/assistant_e2e.py --image /models/avatar-warmup/portrait.png \
      --audio /models/avatar-warmup/speech.wav --output /jobs/ab/assistant-e2e.json
Measures from the server's endpoint event: time to acknowledgement audio, time to the
reply's first frame, received video fps during the reply, stalls reported by the
server, and interruption latency. One run is a smoke measurement, not a p95.
"""
import argparse
import asyncio
import json
import os
import time
import wave
from fractions import Fraction
from pathlib import Path

import av
import httpx
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription


class Microphone(MediaStreamTrack):
    kind = "audio"

    def __init__(self, samples, lead_seconds=2.0):
        super().__init__()
        self.samples = np.concatenate([np.zeros(int(16000 * lead_seconds), np.int16), samples])
        self.index = 0
        self.epoch = None

    async def recv(self):
        if self.epoch is None:
            self.epoch = time.monotonic()
        await asyncio.sleep(max(0, self.epoch + self.index / 16000 - time.monotonic()))
        block = np.zeros(320, np.int16)
        part = self.samples[self.index:self.index + 320]
        block[:len(part)] = part
        frame = av.AudioFrame.from_ndarray(block[None], format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 16000, self.index, Fraction(1, 16000)
        self.index += 320
        return frame


async def run(args):
    with wave.open(args.audio) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("audio must be mono PCM16 at 16000 Hz")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
    pc = RTCPeerConnection()
    mic = Microphone(samples)
    pc.addTrack(mic)
    pc.addTransceiver("video", direction="recvonly")
    channel = pc.createDataChannel("events")
    epoch = time.monotonic()
    events, tasks = [], []
    marks = {}
    video_times, audio_active = [], []
    frame_change = []
    last_pixels = [None]
    listening_after_reply = asyncio.Event()
    interrupt_at = [None]
    interrupt_ack = [None]

    @channel.on("message")
    def on_message(raw):
        value = json.loads(raw)
        value["t"] = round(time.monotonic() - epoch, 3)
        events.append(value)
        kind = value.get("type")
        if kind in ("thinking", "acknowledging", "reply_scheduled", "speaking", "reply", "transcript") and kind not in marks:
            marks[kind] = time.monotonic()
        if kind == "listening":
            if interrupt_at[0] is not None and interrupt_ack[0] is None:
                interrupt_ack[0] = time.monotonic() - interrupt_at[0]
            if "speaking" in marks:
                marks.setdefault("listening_after_reply", time.monotonic())
                listening_after_reply.set()

    async def receive(track):
        try:
            while True:
                frame = await track.recv()
                now = time.monotonic()
                if track.kind == "video":
                    video_times.append(now)
                    pixels = frame.to_ndarray(format="rgb24").astype(np.float32)
                    if last_pixels[0] is not None:
                        frame_change.append((now, float(np.abs(pixels - last_pixels[0]).mean())))
                    last_pixels[0] = pixels
                else:
                    rms = float(np.sqrt(np.mean(frame.to_ndarray().astype(np.float32) ** 2)))
                    if rms > 100:
                        audio_active.append(now)
        except Exception as exc:
            if pc.connectionState != "closed":
                events.append({"type": "receiver_end", "detail": type(exc).__name__})

    @pc.on("track")
    def on_track(track):
        tasks.append(asyncio.create_task(receive(track)))

    session = None
    headers = {"x-internal-key": os.getenv("INTERNAL_API_KEY", "")}
    async with httpx.AsyncClient(base_url=args.ingest, headers=headers, timeout=300) as client:
        try:
            created_at = time.monotonic()
            result = await client.post("/assistant/sessions", files={"image": ("portrait.png", Path(args.image).read_bytes(), "image/png")},
                                       data={"language": "en"})
            result.raise_for_status()
            body = result.json()
            session = body["session_id"]
            prepare_seconds = time.monotonic() - created_at
            await pc.setLocalDescription(await pc.createOffer())
            result = await client.post(f"/assistant/sessions/{session}/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"})
            result.raise_for_status()
            await pc.setRemoteDescription(RTCSessionDescription(**result.json()))
            try:
                await asyncio.wait_for(listening_after_reply.wait(), args.timeout)
            except asyncio.TimeoutError:
                events.append({"type": "test_timeout"})
            if args.interrupt_audio:
                # Second turn, interrupted two seconds into its reply.
                with wave.open(args.interrupt_audio) as wav:
                    second = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
                marks.pop("speaking", None); listening_after_reply.clear()
                mic.samples = np.concatenate([mic.samples[:mic.index], np.zeros(8000, np.int16), second, np.zeros(16000 * 60, np.int16)])
                deadline = time.monotonic() + args.timeout
                while "speaking" not in marks and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                await asyncio.sleep(2)
                interrupt_at[0] = time.monotonic()
                channel.send("interrupt")
                await asyncio.sleep(2)
            status = await client.get(f"/assistant/sessions/{session}")
            status.raise_for_status()
            endpoint = marks.get("thinking")
            speaking = marks.get("speaking")
            reply_video = [t for t in video_times if speaking and t >= speaking and t <= marks.get("listening_after_reply", t)]
            span = reply_video[-1] - reply_video[0] if len(reply_video) > 1 else 0
            first_ack_audio = next((t for t in audio_active if endpoint and t >= endpoint), None)
            reply_audio = next((t for t in audio_active if speaking and t >= speaking), None)
            motion_in_reply = [c for t, c in frame_change if speaking and t >= speaking]
            report = {
                "session": session, "connection": pc.connectionState, "prepare_seconds": round(prepare_seconds, 2),
                "server_prepare": body.get("prepare"),
                "endpoint_to_ack_audio_seconds": round(first_ack_audio - endpoint, 3) if first_ack_audio and endpoint else None,
                "endpoint_to_reply_scheduled_event_seconds": round(marks["reply_scheduled"] - endpoint, 3) if "reply_scheduled" in marks and endpoint else None,
                "endpoint_to_reply_speaking_seconds": round(speaking - endpoint, 3) if speaking and endpoint else None,
                "endpoint_to_reply_audio_seconds": round(reply_audio - endpoint, 3) if reply_audio and endpoint else None,
                "reply_video_fps": round((len(reply_video) - 1) / span, 2) if span else None,
                "reply_frames": len(reply_video),
                "reply_mean_frame_change": round(float(np.mean(motion_in_reply)), 3) if motion_in_reply else None,
                "interrupt_ack_seconds": interrupt_ack[0],
                "last_audio_after_interrupt_seconds": max([t - interrupt_at[0] for t in audio_active if interrupt_at[0] and t >= interrupt_at[0]], default=None),
                "server": status.json(), "events": events,
            }
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if k not in ("server", "events")}, indent=2), flush=True)
            return 0 if speaking and report["reply_video_fps"] else 1
        finally:
            if session:
                await client.delete(f"/assistant/sessions/{session}")
            await pc.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--interrupt-audio", help="second utterance; its reply is interrupted after 2 s")
    parser.add_argument("--ingest", default="http://localhost:8091")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
