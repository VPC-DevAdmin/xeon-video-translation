"""Chunk arithmetic for the FlashHead render service. No torch."""

import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from app.chunking import AudioContext, ChunkSpec, decode_pcm, fit_chunk  # noqa: E402

PRO = ChunkSpec(frame_num=33, motion_frames=5, fps=25, sample_rate=16000, cached_seconds=8)


def test_pro_chunk_is_28_frames_and_17920_samples():
    assert (PRO.frames, PRO.samples, round(PRO.seconds, 2)) == (28, 17920, 1.12)
    assert PRO.as_dict()["samples_per_chunk"] == 17920


def test_fit_chunk_pads_and_reports_covered_frames():
    audio, covered = fit_chunk(np.ones(8000, np.float32), PRO)
    assert len(audio) == 17920 and audio[8000:].sum() == 0
    assert covered == 13          # 0.5 s of audio -> ceil(12.5) frames
    audio, covered = fit_chunk(np.ones(17920, np.float32), PRO)
    assert covered == 28
    audio, covered = fit_chunk(np.ones(20000, np.float32), PRO)
    assert len(audio) == 17920 and covered == 28
    assert fit_chunk(np.zeros(0, np.float32), PRO)[1] == 0


def test_pcm_round_trip_and_odd_payload():
    pcm = (np.array([0, 16384, -32768], np.int16)).tobytes()
    out = decode_pcm(base64.b64encode(pcm).decode())
    assert np.allclose(out, [0, 0.5, -1.0])
    try:
        decode_pcm(base64.b64encode(b"\x00").decode())
    except ValueError:
        pass
    else:
        raise AssertionError("odd byte count should fail")


def test_audio_context_window_is_the_last_frame_num_frames():
    ctx = AudioContext(PRO)
    assert ctx.window == (200 - 33, 200)
    window = ctx.push(np.ones(PRO.samples, np.float32))
    assert len(window) == 8 * 16000 and window[-1] == 1.0 and window[0] == 0.0
    ctx.reset()
    assert np.asarray(ctx.cache).sum() == 0
