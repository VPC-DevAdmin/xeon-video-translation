import asyncio
import fractions
from types import SimpleNamespace
import av
import pytest
from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.codecs.h264 import H264Decoder
from aiortc.jitterbuffer import JitterFrame
from app import gpu_media


def cpu_encoder(w, h, bitrate=2000000):
    c = av.CodecContext.create("libx264", "w")
    c.width, c.height = w, h
    c.bit_rate = bitrate
    c.pix_fmt = "yuv420p"
    c.time_base = fractions.Fraction(1, 30)
    c.options = {"tune": "zerolatency", "preset": "ultrafast"}
    c.max_b_frames = 0
    return c


def frame(i, w=128, h=128):
    f = av.VideoFrame(w, h, "yuv420p")
    for p in f.planes:
        p.update(bytes(p.buffer_size))
    f.pts, f.time_base = i * 3000, fractions.Fraction(1, 90000)
    return f


def test_adapter_packetizes_and_preserves_rtp_clock(monkeypatch):
    monkeypatch.setattr(gpu_media, "encoder", cpu_encoder)
    encode = gpu_media.NvencEncoder()
    decode = H264Decoder()
    for i in range(3):
        payloads, timestamp = encode.encode(frame(i), force_keyframe=i == 2)
        assert payloads and timestamp == i * 3000
        raw = b"".join(
            gpu_media.codecs.depayload(
                SimpleNamespace(mimeType="video/H264", name="H264"), p
            )
            for p in payloads
        )
        frames = decode.decode(JitterFrame(raw, timestamp))
        assert len(frames) == 1 and frames[0].width == 128


def test_encoder_reconfigures_and_requests_keyframe(monkeypatch):
    monkeypatch.setattr(gpu_media, "encoder", cpu_encoder)
    e = gpu_media.NvencEncoder()
    e.encode(frame(0))
    e.target_bitrate = 500000
    payloads, _ = e.encode(frame(1, 160, 128))
    assert payloads and e.codec.width == 160 and e.codec.bit_rate == 500000


def test_gpu_encoder_failure_is_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("NVENC unavailable")

    monkeypatch.setattr(
        gpu_media, "av", SimpleNamespace(CodecContext=SimpleNamespace(create=fail))
    )
    with pytest.raises(RuntimeError, match="NVENC"):
        gpu_media.encoder(128, 128)


@pytest.mark.asyncio
async def test_h264_only_negotiation(monkeypatch):
    monkeypatch.setenv("WEBRTC_GPU_CODECS", "1")
    pc = RTCPeerConnection()
    try:
        gpu_media.prefer_h264(pc)
        offer = await pc.createOffer()
        assert "H264/90000" in offer.sdp and "VP8/90000" not in offer.sdp
    finally:
        await pc.close()


def test_install_is_version_fenced(monkeypatch):
    monkeypatch.setenv("WEBRTC_GPU_CODECS", "1")
    monkeypatch.setattr(gpu_media, "version", lambda _: "future-version")
    with pytest.raises(RuntimeError, match="qualify upgrades"):
        gpu_media.install()


@pytest.mark.asyncio
async def test_recording_selects_nvenc_and_reports_task_failure(monkeypatch, tmp_path):
    # Exercise pinned upstream recorder context layout without a GPU.
    recorder = gpu_media.GpuRecorder(str(tmp_path / "record.mp4"))
    original = recorder._MediaRecorder__container
    calls = []

    class Container:
        def add_stream(self, name, **kwargs):
            calls.append(name)
            return original.add_stream("libx264", **kwargs)

        def close(self):
            original.close()

        def mux(self, packet):
            original.mux(packet)

    recorder._MediaRecorder__container = Container()
    track = VideoStreamTrack()
    recorder.addTrack(track)
    assert calls == ["h264_nvenc"]
    context = recorder._MediaRecorder__tracks[track]

    async def failed():
        raise RuntimeError("hardware encoder lost")

    context.task = asyncio.create_task(failed())
    await asyncio.gather(context.task, return_exceptions=True)
    with pytest.raises(RuntimeError, match="GPU recording failed"):
        await recorder.stop()
    track.stop()


def test_bitrate_update_preserves_encoder_session(monkeypatch):
    monkeypatch.setattr(gpu_media, "encoder", cpu_encoder)
    encoder = gpu_media.NvencEncoder()
    encoder.encode(frame(0))
    context = encoder.codec
    encoder.target_bitrate = 1000000
    packets, pts = encoder.encode(frame(1))
    assert packets and pts == 3000
    assert encoder.codec is context and context.bit_rate == 1000000


@pytest.mark.asyncio
async def test_audio_waits_for_delayed_video_dimensions(tmp_path):
    from aiortc import AudioStreamTrack

    class DelayedVideo(VideoStreamTrack):
        first = True

        async def recv(self):
            if self.first:
                self.first = False
                await asyncio.sleep(0.25)
            pts, base = await self.next_timestamp()
            value = av.VideoFrame(320, 240, "yuv420p")
            for plane in value.planes:
                plane.update(bytes(plane.buffer_size))
            value.pts, value.time_base = pts, base
            return value

    path = tmp_path / "delayed-video.mp4"
    recorder = gpu_media.GpuRecorder(str(path))
    original = recorder._MediaRecorder__container

    class Container:
        format = original.format

        def add_stream(self, name, **kwargs):
            return original.add_stream("libx264" if name == "h264_nvenc" else name, **kwargs)

        def mux(self, packet):
            # Any audio-first mux would open the video codec at its default size.
            assert original.streams.video[0].width == 320
            assert original.streams.video[0].height == 240
            original.mux(packet)

        def close(self):
            original.close()

    recorder._MediaRecorder__container = Container()
    audio, video = AudioStreamTrack(), DelayedVideo()
    recorder.addTrack(audio)
    recorder.addTrack(video)
    recorder._MediaRecorder__tracks[video].stream.options = {
        "preset": "ultrafast", "tune": "zerolatency"
    }
    try:
        await recorder.start()
        await asyncio.sleep(0.7)
        await recorder.stop()
    finally:
        audio.stop()
        video.stop()
    with av.open(str(path)) as media:
        assert media.streams.video[0].width == 320
        assert sum(1 for _ in media.decode(video=0)) >= 5
    with av.open(str(path)) as media:
        assert sum(frame.samples for frame in media.decode(audio=0)) > 0


@pytest.mark.asyncio
async def test_cancelled_stop_drains_native_worker_before_closing(tmp_path, monkeypatch):
    import threading

    recorder = gpu_media.GpuRecorder(str(tmp_path / "drain.mp4"))
    track = VideoStreamTrack()
    entered, release = threading.Event(), threading.Event()
    order = []
    context = SimpleNamespace(started=True, task=None)
    recorder._MediaRecorder__tracks[track] = context

    def encode(context, frame):
        entered.set()
        assert release.wait(timeout=5), "test failed to release native worker"
        order.append("encoded")

    def finish():
        order.append("closed")
        recorder._MediaRecorder__container.close()
        recorder._MediaRecorder__container = None
        recorder._MediaRecorder__tracks = {}

    monkeypatch.setattr(recorder, "_encode_and_mux", encode)
    monkeypatch.setattr(recorder, "_finish", finish)
    context.task = asyncio.create_task(recorder._MediaRecorder__run_track(track, context))
    stopping = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stopping = asyncio.create_task(recorder.stop())
        await asyncio.sleep(0.02)
        stopping.cancel()
        await asyncio.sleep(0.02)
        assert not stopping.done() and order == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert order == ["encoded", "closed"]
        await recorder.stop()  # idempotent after a cancelled caller
    finally:
        release.set()
        track.stop()
        if stopping and not stopping.done():
            await asyncio.gather(stopping, return_exceptions=True)
