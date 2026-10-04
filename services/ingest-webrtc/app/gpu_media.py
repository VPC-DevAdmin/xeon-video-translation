"""NVENC/NVDEC adapter for the pinned aiortc 1.15 / PyAV 17 transport.

Frames still cross host memory at aiortc's boundary. This moves the codecs to
NVIDIA hardware without claiming a zero-copy media worker. No CPU fallback.
"""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import fractions
import os
from importlib.metadata import version
import av
import aiortc.codecs as codecs
from aiortc import RTCRtpSender, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.codecs.h264 import H264Encoder, H264Decoder
from aiortc.contrib.media import MediaRecorder, MediaRecorderContext


def enabled():
    return os.getenv("WEBRTC_GPU_CODECS", "0") == "1"


def encoder(width, height, bitrate=2000000):
    codec = av.CodecContext.create("h264_nvenc", "w")
    codec.width, codec.height = width, height
    codec.bit_rate = bitrate
    codec.pix_fmt = "yuv420p"
    codec.framerate = fractions.Fraction(30)
    codec.time_base = fractions.Fraction(1, 30)
    codec.max_b_frames = 0
    codec.gop_size = 50
    codec.options = {
        "preset": "p1",
        "tune": "ull",
        "rc": "cbr",
        "zerolatency": "1",
        "rc-lookahead": "0",
        "delay": "0",
        "forced-idr": "1",
        "profile": "baseline",
        "level": "3.1",
    }
    return codec


class NvencEncoder(H264Encoder):
    def _encode_frame(self, frame, force_keyframe):
        changed = self.codec and (
            self.codec.width != frame.width
            or self.codec.height != frame.height
        )
        if self.codec is None or changed:
            self.codec = encoder(frame.width, frame.height, self.target_bitrate)
            force_keyframe = True
        elif abs(self.target_bitrate - self.codec.bit_rate) / max(1, self.codec.bit_rate) > 0.1:
            # FFmpeg's NVENC wrapper reconfigures bitrate on the existing session.
            # Reopening CUDA/NVENC for each REMB update stalls live video.
            self.codec.bit_rate = self.target_bitrate
        frame.pict_type = (
            av.video.frame.PictureType.I
            if force_keyframe
            else av.video.frame.PictureType.NONE
        )
        data = b"".join(bytes(packet) for packet in self.codec.encode(frame))
        yield from self._split_bitstream(data)


class NvdecDecoder(H264Decoder):
    def __init__(self):
        self.codec = av.CodecContext.create("h264_cuvid", "r")
        self.codec.options = {"flags": "+low_delay"}

    def decode(self, encoded_frame):
        # Own software-format planes before crossing from the NVDEC worker
        # to the recorder thread. The aiortc boundary is host-backed.
        return [frame.reformat(format="yuv420p") for frame in super().decode(encoded_frame)]


def install():
    if not enabled():
        return
    if version("aiortc") != "1.15.0" or version("av") != "17.1.0":
        raise RuntimeError(
            "GPU transport requires the tested aiortc/PyAV lock; qualify upgrades first"
        )
    codecs.H264Encoder = NvencEncoder
    codecs.H264Decoder = NvdecDecoder


def prefer_h264(pc):
    if not enabled():
        return
    video = [t for t in pc.getTransceivers() if t.kind == "video"]
    if not video:
        video = [pc.addTransceiver("video", direction="recvonly")]
    supported = [
        c
        for c in RTCRtpSender.getCapabilities("video").codecs
        if c.mimeType.lower() == "video/h264"
    ]
    for transceiver in video:
        transceiver.setCodecPreferences(supported)


def probe():
    """Exercise PyAV's own codec libraries, not the unrelated ffmpeg binary."""
    if not enabled():
        return {"ready": True, "video_codecs": "software", "host_frame_exchange": True}
    install()
    encode = encoder(256, 256)
    decode = av.CodecContext.create("h264_cuvid", "r")
    frames = 0
    for index in range(4):
        frame = av.VideoFrame(256, 256, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.pts, frame.time_base = index, fractions.Fraction(1, 30)
        for packet in encode.encode(frame):
            frames += len(decode.decode(packet))
    for packet in encode.encode(None):
        frames += len(decode.decode(packet))
    frames += len(decode.decode(None))
    if frames != 4:
        raise RuntimeError(f"NVENC/NVDEC round trip returned {frames}/4 frames")
    return {"ready": True, "video_codecs": "nvenc/nvdec", "host_frame_exchange": True}


class GpuRecorder(MediaRecorder):
    """Serialize NVENC/mux work off the event loop, preserving timestamps.

    Private recorder contexts are pinned and covered by adapter tests. A future
    aiortc upgrade must replace/requalify this adapter before changing the lock.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvenc-recorder")
        self._stop_lock = asyncio.Lock()
        self._stop_task = None

    async def _MediaRecorder__run_track(self, track, context):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            future = asyncio.get_running_loop().run_in_executor(
                self._worker, self._encode_and_mux, context, frame
            )
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                # Native NVENC work cannot be cancelled. Drain before closing files.
                await asyncio.shield(future)
                raise

    def _encode_and_mux(self, context, frame):
        context.started = True
        for packet in context.stream.encode(frame):
            self._MediaRecorder__container.mux(packet)

    async def start(self):
        tracks = self._MediaRecorder__tracks
        pending = {track for track, context in tracks.items()
                   if track.kind == "video" and not context.started}
        configured = asyncio.Event()
        if not pending:
            configured.set()

        class PrimedTrack(MediaStreamTrack):
            def __init__(self, source, context):
                super().__init__()
                self.source, self.context, self.kind = source, context, source.kind

            async def recv(self):
                if self.kind == "audio":
                    # Muxing the first AAC packet opens ALL container codecs.
                    # Wait until video dimensions are known, before NVENC opens.
                    await asyncio.wait_for(configured.wait(), timeout=15)
                frame = await self.source.recv()
                if self.kind == "video" and not self.context.started:
                    self.context.stream.width = frame.width
                    self.context.stream.height = frame.height
                    self.context.started = True
                    pending.discard(self.source)
                    if not pending:
                        configured.set()
                return frame

        for track, context in tracks.items():
            if context.task is None:
                context.task = asyncio.ensure_future(
                    self._MediaRecorder__run_track(PrimedTrack(track, context), context)
                )

    def addTrack(self, track):
        if track.kind != "video":
            return super().addTrack(track)
        stream = self._MediaRecorder__container.add_stream("h264_nvenc", rate=30)
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.options = {
            "preset": "p4",
            "tune": "ll",
            "rc": "vbr",
            "cq": "18",
            "b": "0",
        }
        self._MediaRecorder__tracks[track] = MediaRecorderContext(stream)

    def _finish(self):
        container = self._MediaRecorder__container
        try:
            for context in self._MediaRecorder__tracks.values():
                if context.started:
                    for packet in context.stream.encode(None):
                        container.mux(packet)
        finally:
            try:
                container.close()
            finally:
                self._MediaRecorder__container = None
                self._MediaRecorder__tracks = {}

    async def stop(self):
        # Cancellation of a request must not close a muxer while its native
        # encoder is still using it, or abandon the worker and output file.
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await asyncio.shield(self._stop_task)
            raise

    async def _stop(self):
        async with self._stop_lock:
            if self._MediaRecorder__container is None:
                return
            tasks = [c.task for c in self._MediaRecorder__tracks.values() if c.task]
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [r for r in results if isinstance(r, Exception)]
            future = asyncio.get_running_loop().run_in_executor(self._worker, self._finish)
            try:
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    await asyncio.shield(future)
                    raise
            finally:
                self._worker.shutdown(wait=False)
            if failures:
                raise RuntimeError("GPU recording failed") from failures[0]


def recorder(path):
    return GpuRecorder(path) if enabled() else MediaRecorder(path)
