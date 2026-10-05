"""Streaming translation: span planning, the head-start rule and the growing playlist."""

from pathlib import Path

from app.pipeline.streaming import Playlist, head_start_ready, speech_spans


def test_speech_spans_merge_short_pauses_pad_and_snap_to_frames():
    segments = [{"start": 1.0, "end": 2.5}, {"start": 2.9, "end": 4.0}, {"start": 6.0, "end": 7.0}, {"start": 9.9, "end": 10.3}]
    spans = speech_spans(segments, total_seconds=10.0, gap=0.6, pad=0.25)
    assert [(s.start, s.end, s.segments) for s in spans] == [(0.72, 4.28, [0, 1]), (5.72, 7.28, [2]), (9.64, 10.0, [3])]
    assert all(round(s.start * 25, 6) == int(round(s.start * 25)) for s in spans)        # on the 25 fps grid
    assert speech_spans([{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 2.0}], 2.0, gap=0.1, pad=0.5)[0].segments == [0, 1]


def test_head_start_rule_matches_the_assistant():
    assert head_start_ready(ready=5.0, remaining=20.0, rate=1.2)            # faster than real time: only the margin matters
    assert not head_start_ready(ready=5.0, remaining=20.0, rate=0.5)        # needs 10 s + margin
    assert head_start_ready(ready=12.5, remaining=20.0, rate=0.5)
    assert head_start_ready(ready=0.0, remaining=0.0, rate=0.0)


def test_playlist_grows_and_ends(tmp_path: Path):
    playlist = Playlist(tmp_path)
    playlist.add("stream-00000.ts", 3.2)
    text = (tmp_path / "stream.m3u8").read_text()
    assert "#EXT-X-PLAYLIST-TYPE:EVENT" in text and "#EXTINF:3.200," in text and "#EXT-X-ENDLIST" not in text
    assert "#EXT-X-TARGETDURATION:4" in text
    playlist.add("stream-00001.ts", 8.0)
    playlist.write(ended=True)
    text = (tmp_path / "stream.m3u8").read_text()
    assert text.count("#EXTINF") == 2 and text.rstrip().endswith("#EXT-X-ENDLIST") and "#EXT-X-TARGETDURATION:8" in text
