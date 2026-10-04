"""Timeline scheduling, idle loop, stall accounting and the head-start rule."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from app.timeline import Timeline, head_start_required  # noqa: E402


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
    # idle ping-pong: period 2n-2 = 4 frames -> indices 0,1,2,1
    tl2 = Timeline(fps=1, idle_frames=np.arange(3)[:, None, None, None].astype(np.uint8))
    assert [int(tl2.frame_at(t)[0].reshape(-1)[0]) for t in (0, 1, 2, 3, 4)] == [0, 1, 2, 1, 0]


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
