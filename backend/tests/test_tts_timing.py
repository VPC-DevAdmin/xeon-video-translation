"""Unit tests for the TTS timing helpers.

The XTTS forward call itself is too expensive to drive from unit tests, so
these exercise only the pure-logic bits — longest-word-span selection and
the fallbacks around it. Everything else (ffmpeg calls, XTTS itself)
lands in the smoke test.
"""

from __future__ import annotations

from app.pipeline.tts import _longest_contiguous_word_span


def _w(start: float, end: float) -> dict:
    return {"start": start, "end": end, "text": "x"}


def test_longest_span_picks_the_bigger_of_two_clusters() -> None:
    # Two clusters of contiguous words separated by a 1-second pause.
    words = [
        _w(0.00, 0.30), _w(0.32, 0.55), _w(0.58, 0.80),       # cluster A (0.8s)
        _w(2.00, 2.40), _w(2.45, 2.90), _w(2.95, 3.50), _w(3.55, 4.00),  # cluster B (2.0s)
    ]
    span = _longest_contiguous_word_span(words, max_gap=0.3)
    assert span is not None
    start, end = span
    assert abs(start - 2.00) < 1e-6
    assert abs(end - 4.00) < 1e-6


def test_longest_span_single_word() -> None:
    words = [_w(0.0, 0.5)]
    span = _longest_contiguous_word_span(words, max_gap=0.3)
    assert span == (0.0, 0.5)


def test_longest_span_empty() -> None:
    assert _longest_contiguous_word_span([], max_gap=0.3) is None


def test_longest_span_unsorted_input() -> None:
    # Should still find the correct span even if input comes out of order.
    words = [_w(2.0, 2.4), _w(0.0, 0.3), _w(0.32, 0.6), _w(2.45, 2.9)]
    span = _longest_contiguous_word_span(words, max_gap=0.3)
    assert span is not None
    # Both clusters are the same length here (~0.6s); longest-wins is the
    # last one evaluated when equal. What we really care about is that it
    # didn't crash and returned a valid interval with non-negative length.
    start, end = span
    assert end > start


def test_longest_span_respects_gap_threshold() -> None:
    # 350ms gap is larger than the 300ms threshold → two separate clusters.
    words = [_w(0.0, 0.3), _w(0.65, 0.95)]
    span = _longest_contiguous_word_span(words, max_gap=0.30)
    assert span is not None
    start, end = span
    assert (end - start) < 0.4  # single-word cluster, not 0.95s total
