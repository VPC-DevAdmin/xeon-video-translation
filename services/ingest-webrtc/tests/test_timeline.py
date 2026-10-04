"""Timeline scheduling, idle loop, stall accounting and the head-start rule."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from app.timeline import Timeline, head_start_required, idle_frame  # noqa: E402


def frames(n, value):
    return np.full((n, 2, 2, 3), value, dtype=np.uint8)


def test_future_clips_play_in_order_and_idle_fills_gaps():
    tl = Timeline(fps=25, still=frames(1, 0)[0], idle_frames=frames(3, 9))
    gen = tl.generation
    assert tl.schedule(1.0, np.ones(48000, np.int16), frames(25, 1), gen) == (1.0, 2.0)
    assert tl.schedule(5.0, np.ones(24000, np.int16), frames(25, 2), gen) == (5.0, 6.0)   # frames outlast audio
    assert tl.frame_at(0.5)[1] == "idle"
    assert tl.frame_at(1.5)[0][0, 0, 0] == 1 and tl.frame_at(1.5)[1] == "clip"
    assert tl.frame_at(3.0)[1] == "idle"
    assert tl.frame_at(5.9)[0][0, 0, 0] == 2
    # idle loop with a crossfaded wrap: 8 frames, 2 blended -> loop of 6; frame 0 leans on frame 6
    seq = np.arange(8, dtype=np.uint8) * 10
    tl2 = Timeline(fps=4, idle_frames=seq[:, None, None, None])
    assert tl2.idle_crossfade == 2
    shown = [int(tl2.frame_at(t / 4)[0].reshape(-1)[0]) for t in range(8)]
    assert shown[2:6] == [20, 30, 40, 50] and shown[6] == shown[0] and 0 < shown[0] < 60 and shown[0] > shown[1]


def test_audio_packets_follow_clip_timing():
    tl = Timeline(fps=25)
    tl.schedule(1.0, np.full(48000, 7, np.int16), frames(25, 1), tl.generation)
    assert tl.audio_packet(0.5, 960).sum() == 0
    packet = tl.audio_packet(0.99, 960)            # straddles the clip start
    assert packet[:480].sum() == 0 and packet[-100:].tolist() == [7] * 100
    assert tl.audio_packet(1.5, 960).tolist() == [7] * 960


def test_stale_generation_is_dropped_and_interrupt_clears_everything():
    tl = Timeline(fps=25, still=frames(1, 0)[0])
    gen = tl.generation
    tl.schedule(0.0, np.ones(4800, np.int16), frames(3, 1), gen)
    tl.promise(0.0, 10.0, gen)
    tl.interrupt()
    assert tl.schedule(0.0, np.ones(4800, np.int16), frames(3, 1), gen) is None
    assert tl.clips == [] and tl.promises == []
    assert tl.frame_at(0.05)[1] == "still" and tl.stalls == 0


def test_stalls_count_only_inside_promised_windows_without_a_clip():
    tl = Timeline(fps=25, still=frames(1, 0)[0])
    gen = tl.generation
    tl.promise(2.0, 4.0, gen)
    tl.frame_at(1.0); tl.frame_at(2.5); tl.frame_at(3.0)
    assert tl.stalls == 2
    tl.schedule(2.0, np.ones(96000, np.int16), frames(50, 1), gen)   # the promise is fulfilled
    assert tl.promises == []
    tl.frame_at(2.5)
    assert tl.stalls == 2


def test_head_start_rule():
    assert head_start_required(14.0, 0.87, 1.3) == 1.3 + 14 * 0.13 + 1.0
    assert head_start_required(60.0, 1.2, 4.0) == 5.0                 # faster than real time: no deficit
    assert head_start_required(60.0, 0.87, 1.3) > head_start_required(14.0, 0.87, 1.3)


def test_idle_frame_wrap_is_a_dissolve_from_the_continuation():
    seq = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.uint8)[:, None, None, None]
    k = 3                                   # loop plays 0..6, frames 7,8,9 fade into 0,1,2
    assert int(idle_frame(seq, 6, k)[0, 0, 0]) == 60
    first = int(idle_frame(seq, 7, k)[0, 0, 0])        # index 7 wraps to j=0
    assert 50 < first <= 70 and int(idle_frame(seq, 1, k)[0, 0, 0]) < first
    assert int(idle_frame(seq, 3, k)[0, 0, 0]) == 30
    assert int(idle_frame(seq, 0, 0)[0, 0, 0]) == 0 and int(idle_frame(seq[:1], 5, 3)[0, 0, 0]) == 0
