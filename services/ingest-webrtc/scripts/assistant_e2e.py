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
        self.speech = samples
        self.samples = np.zeros(int(16000 * lead_seconds), np.int16)
        self.hold = False                 # while True, keep sending silence; speech is appended on release
        self.index = 0
        self.epoch = None

    def release(self):
        if self.hold:
            self.hold = False
            played = np.zeros(max(self.index, len(self.samples)), np.int16)
            played[:len(self.samples)] = self.samples
            self.samples = np.concatenate([played, np.zeros(8000, np.int16), self.speech])

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
    if args.wait_ready or args.start_after:
        mic.hold = True
    else:
        mic.samples = np.concatenate([mic.samples, samples])
    pc.addTrack(mic)
    pc.addTransceiver("video", direction="recvonly")
    channel = pc.createDataChannel("events")
    epoch = time.monotonic()
    events, tasks = [], []
    marks = {}
    video_times, audio_active = [], []
    recorded = []                                   # (monotonic time, int16 48 kHz samples)
    frame_change = []
    last_pixels = [None]
    listening_after_reply = asyncio.Event()
    interrupt_at = [None]
    interrupt_ack = [None]

    @channel.on("open")
    def on_open():
        marks.setdefault("channel_open", time.monotonic())

    @channel.on("message")
    def on_message(raw):
        value = json.loads(raw)
        value["t"] = round(time.monotonic() - epoch, 3)
        events.append(value)
        kind = value.get("type")
        if kind == "ready":
            marks.setdefault("ready", time.monotonic())
            mic.release()
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
                    pcm = frame.to_ndarray().reshape(-1)
                    if args.record:
                        recorded.append((now, pcm.astype(np.int16)))
                    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
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
            if args.persona:
                result = await client.post("/assistant/sessions", data={"language": "en", "persona_id": args.persona})
            else:
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
            if args.wait_ready:
                deadline = time.monotonic() + args.wait_ready
                while "ready" not in marks and time.monotonic() < deadline:
                    await asyncio.sleep(0.2)
                if mic.hold:
                    events.append({"type": "ready_timeout", "t": round(time.monotonic() - epoch, 3)})
                    mic.release()
            if args.start_after:
                # Speak at a chosen moment of the cold preparation (after the first
                # acknowledgement, during idle growth) instead of waiting for "ready".
                await asyncio.sleep(max(0.0, args.start_after - (time.monotonic() - created_at)))
                marks.setdefault("released", time.monotonic())
                mic.release()
            try:
                await asyncio.wait_for(listening_after_reply.wait(), args.timeout)
            except asyncio.TimeoutError:
                events.append({"type": "test_timeout"})
            if args.hold_after_reply:
                await asyncio.sleep(args.hold_after_reply)        # watch the idle loop (and background work) afterwards
            if args.interrupt_audio:
                # Second turn, interrupted two seconds into its reply.
                with wave.open(args.interrupt_audio) as wav:
                    second = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
                marks.pop("speaking", None); listening_after_reply.clear()
                played = np.zeros(mic.index, np.int16)          # everything up to now, silence beyond the first clip
                played[:min(len(mic.samples), mic.index)] = mic.samples[:mic.index]
                mic.samples = np.concatenate([played, np.zeros(8000, np.int16), second, np.zeros(16000 * 60, np.int16)])
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
            # First turn window from the event trace (second turn is the interrupted one).
            def first(kind, after=0.0):
                return next((epoch + e["t"] for e in events if e.get("type") == kind and e["t"] >= after), None)
            speaking = first("speaking")
            reply_end = first("listening", (speaking - epoch) if speaking else 0.0) if speaking else None
            reply_video = [t for t in video_times if speaking and t >= speaking and (reply_end is None or t <= reply_end)]
            span = reply_video[-1] - reply_video[0] if len(reply_video) > 1 else 0
            first_ack_audio = next((t for t in audio_active if endpoint and t >= endpoint), None)
            reply_audio = next((t for t in audio_active if speaking and t >= speaking), None)
            motion_in_reply = [c for t, c in frame_change if speaking and t >= speaking and (reply_end is None or t <= reply_end)]

            def idle_stats(changes):
                """Frame-to-frame change over idle footage; a snap is a change far above the typical motion."""
                if len(changes) < 10:
                    return None
                values = np.array([c for _, c in changes])
                median = float(np.median(values))
                threshold = max(3.0 * median, 6.0)
                return {"frames": len(values), "median": round(median, 2), "p99": round(float(np.percentile(values, 99)), 2),
                        "max": round(float(values.max()), 2), "snaps": int((values > threshold).sum()), "snap_threshold": round(threshold, 2),
                        "snap_times": [round(t - created_at, 2) for (t, c) in changes if c > threshold][:20]}
            idle_before = [(t, c) for t, c in frame_change if endpoint is None or t < endpoint - 0.5]
            idle_after = [(t, c) for t, c in frame_change if reply_end and t > reply_end + 1.0]
            report = {
                "session": session, "connection": pc.connectionState, "prepare_seconds": round(prepare_seconds, 2),
                "ready_after_seconds": round(marks["ready"] - created_at, 2) if "ready" in marks else None,
                "server_prepare": body.get("prepare"),
                "endpoint_to_ack_audio_seconds": round(first_ack_audio - endpoint, 3) if first_ack_audio and endpoint else None,
                "endpoint_to_reply_scheduled_event_seconds": round(marks["reply_scheduled"] - endpoint, 3) if "reply_scheduled" in marks and endpoint else None,
                "endpoint_to_reply_speaking_seconds": round(speaking - endpoint, 3) if speaking and endpoint else None,
                "endpoint_to_reply_audio_seconds": round(reply_audio - endpoint, 3) if reply_audio and endpoint else None,
                "reply_video_fps": round((len(reply_video) - 1) / span, 2) if span else None,
                "reply_frames": len(reply_video),
                "reply_mean_frame_change": round(float(np.mean(motion_in_reply)), 3) if motion_in_reply else None,
                "idle_before_turn": idle_stats(idle_before), "idle_after_reply": idle_stats(idle_after),
                "ack_ready_after_seconds": round(next((epoch + e["t"] for e in events if e.get("type") == "ack_ready"), created_at) - created_at, 2),
                "event_times": [(e["type"], round(epoch + e["t"] - created_at, 2)) for e in events if e.get("type") not in ("transcript", "reply")][:40],
                "released_after_seconds": round(marks["released"] - created_at, 2) if "released" in marks else None,
                "channel_open_after_seconds": round(marks["channel_open"] - created_at, 2) if "channel_open" in marks else None,
                "interrupt_ack_seconds": interrupt_ack[0],
                "last_audio_after_interrupt_seconds": max([t - interrupt_at[0] for t in audio_active if interrupt_at[0] and t >= interrupt_at[0]], default=None),
                "server": status.json(), "events": events,
            }
            if args.record and recorded:
                import wave as _wave
                with _wave.open(args.record, "wb") as out:
                    out.setparams((1, 2, 48000, 0, "NONE", "nc"))
                    out.writeframes(np.concatenate([p for _, p in recorded]).tobytes())
                report["recording"] = {"path": args.record, "seconds": round(sum(len(p) for _, p in recorded) / 48000, 2),
                                       "starts_at_client_seconds": round(recorded[0][0] - epoch, 3)}
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
    parser.add_argument("--image", help="portrait upload (or use --persona)")
    parser.add_argument("--persona", help="persona id from the backend /personas API")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--interrupt-audio", help="second utterance; its reply is interrupted after 2 s")
    parser.add_argument("--ingest", default="http://localhost:8091")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--wait-ready", type=float, default=0, help="hold the utterance until the session reports ready (seconds max)")
    parser.add_argument("--start-after", type=float, default=0, help="hold the utterance until this many seconds after session creation")
    parser.add_argument("--hold-after-reply", type=float, default=0, help="keep the session open this long after the reply to observe the idle loop")
    parser.add_argument("--record", help="write the received audio (48 kHz mono) to this wav")
    parser.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
