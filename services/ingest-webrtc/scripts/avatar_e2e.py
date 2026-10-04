"""Exercise avatar ICE/SRTP, speech reply, moving frames and interruption.

Run inside ingest with --image portrait.png --audio mono-16k.wav.
A smoke measurement is not a p95 or perceptual-quality qualification.
"""
import argparse
import asyncio
from fractions import Fraction
import json
import os
from pathlib import Path
import time
import wave

import av
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
import httpx
import numpy as np


class Microphone(MediaStreamTrack):
    kind = "audio"

    def __init__(self, samples):
        super().__init__()
        self.samples = np.concatenate([np.zeros(32000, np.int16), samples])
        active_ends = [min(i + 320, len(samples)) for i in range(0, len(samples), 320)
                       if np.sqrt(np.mean(samples[i:i+320].astype(np.float32) ** 2)) > 150]
        self.active_end_offset = 2 + (max(active_ends, default=len(samples)) / 16000)
        self.index = 0
        self.epoch = None
        self.speech_end = None

    async def recv(self):
        if self.epoch is None:
            self.epoch = time.monotonic()
            self.speech_end = self.epoch + self.active_end_offset
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
    events, tasks = [], []
    first_audio, first_motion = None, None
    video_times, audio_active = [], []
    static = None
    playback_started = None
    epoch = time.monotonic()
    replied = asyncio.Event()
    interrupted_at = None
    interrupt_ack = None

    @channel.on("message")
    def event(raw):
        nonlocal interrupt_ack, playback_started
        value = json.loads(raw)
        value["client_elapsed"] = time.monotonic() - epoch
        events.append(value)
        if value.get("type") == "latency":
            playback_started = time.monotonic()
        if interrupted_at is not None and value.get("type") == "listening":
            interrupt_ack = time.monotonic() - interrupted_at

    async def receive(track):
        nonlocal first_audio, first_motion, static
        try:
            while True:
                frame = await track.recv()
                now = time.monotonic()
                if track.kind == "video":
                    video_times.append(now)
                    pixels = frame.to_ndarray(format="bgr24").astype(np.float32)
                    if static is None or playback_started is None:
                        static = pixels
                    elif first_motion is None and np.abs(pixels - static).mean() > 0.3:
                        first_motion = now
                else:
                    rms = float(np.sqrt(np.mean(frame.to_ndarray().astype(np.float32) ** 2)))
                    if rms > 100:
                        audio_active.append(now)
                        if first_audio is None:
                            first_audio = now
                            replied.set()
        except Exception as exc:
            if pc.connectionState != "closed":
                events.append({"type": "receiver_end", "detail": type(exc).__name__})

    @pc.on("track")
    def track(track):
        tasks.append(asyncio.create_task(receive(track)))

    session = None
    headers = {"x-internal-key": os.getenv("INTERNAL_API_KEY", "")}
    async with httpx.AsyncClient(base_url=args.ingest, headers=headers, timeout=60) as client:
        try:
            result = await client.post("/avatar/sessions", files={"image": ("portrait.png", Path(args.image).read_bytes(), "image/png")}, data={"language": "en"})
            result.raise_for_status()
            session = result.json()["session_id"]
            await pc.setLocalDescription(await pc.createOffer())
            result = await client.post(f"/avatar/sessions/{session}/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"})
            result.raise_for_status()
            await pc.setRemoteDescription(RTCSessionDescription(**result.json()))
            try:
                await asyncio.wait_for(replied.wait(), args.timeout)
                await asyncio.sleep(1)
                interrupted_at = time.monotonic()
                channel.send("interrupt")
                await asyncio.sleep(2)
            except asyncio.TimeoutError:
                events.append({"type": "test_timeout"})
            status = await client.get(f"/avatar/sessions/{session}")
            status.raise_for_status()
            span = video_times[-1] - video_times[0] if len(video_times) > 1 else 0
            steady = [t for t in video_times if t >= video_times[0] + 1] if video_times else []
            steady_span = steady[-1] - steady[0] if len(steady) > 1 else 0
            endpoints = [e["client_elapsed"] for e in events if e.get("type") == "thinking"]
            endpoint = epoch + endpoints[-1] if endpoints else None
            report = {
                "session": session, "connection": pc.connectionState,
                "endpoint_to_heard_seconds": first_audio - endpoint if first_audio and endpoint else None,
                "endpoint_to_motion_seconds": first_motion - endpoint if first_motion and endpoint else None,
                "first_heard_after_input_end_seconds": first_audio - mic.speech_end if first_audio and mic.speech_end else None,
                "first_motion_after_input_end_seconds": first_motion - mic.speech_end if first_motion and mic.speech_end else None,
                "received_video_fps": (len(steady) - 1) / steady_span if steady_span else 0,
                "whole_session_video_fps": (len(video_times) - 1) / span if span else 0,
                "motion_measurement": "change from last idle frame after reply enqueue; not a lip-sync score",
                "interrupt_ack_seconds": interrupt_ack,
                "last_audio_after_interrupt_seconds": max([t - interrupted_at for t in audio_active if interrupted_at and t >= interrupted_at], default=0),
                "server": status.json(), "events": events,
                "quality_status": "unreviewed", "sample_count": 1,
                "speech_end_reference": "last 20ms source block with RMS > 150 PCM16 units; proxy requiring human review",
            }
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            return 0 if first_audio and first_motion and not status.json()["metrics"]["render_fallbacks"] else 1
        finally:
            if session:
                await client.delete(f"/avatar/sessions/{session}")
            await pc.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ingest", default="http://localhost:8091")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
