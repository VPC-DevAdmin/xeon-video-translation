"""Streaming translation: translate a clip per speech span and publish each
finished piece while the rest renders.

The whole-clip pipeline runs transcribe, translate, TTS and lipsync as stages
over the entire video, so nothing is watchable until the last window has
rendered. This module keeps the shared audio and transcribe stages and then
works per *speech span* (transcript segments merged across pauses shorter than
`gap`, padded by `pad`):

  for each span in time order
      synthesize the span's translated speech (per-segment TTS with its slot
      fitting and whisper verification), aligned to the span's own clock
      render only the span's frames, in windows, with the shared face track
      publish each rendered window and each passthrough gap as an HLS segment

Frames outside speech spans are copied from the source untouched, and inside a
span LatentSync changes only the face region, so the output is the original
video with mouth patches on speaking frames. The playlist is an EVENT playlist
the browser can follow while it grows; `lipsynced.mp4` and `translated_audio.wav`
are assembled at the end so the usual mux stage produces the final file.

Head start, as in the video assistant: with media produced at `rate` seconds per
wall second and `remaining` seconds still to come, playback from the start cannot
stall once `ready >= (1 - rate) * remaining + margin`. Every `stream_segment`
event carries the numbers so the player can decide when to start.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import settings
from . import lipsync, tts, translate
from .windowed import encode, run_ffmpeg, duration

log = logging.getLogger(__name__)
FPS = 25


@dataclass
class Span:
    index: int
    start: float                # seconds, on the 25 fps grid
    end: float
    segments: list[int]         # transcript / translation segment indices inside it
    speaker: str | None = None  # speakers.py id; spans never cross a speaker change

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class Piece:
    index: int
    start: float
    end: float
    kind: str                   # "gap" (passthrough) or "speech" (rendered)
    path: Path
    audio: Path | None = None
    render_seconds: float = 0.0


def speech_spans(segments: list[dict], total_seconds: float, gap: float = 0.6, pad: float = 0.25) -> list[Span]:
    """Merge transcript segments separated by less than `gap` seconds into spans,
    pad them, snap to the frame grid and clamp to the clip. Segments of
    different speakers never share a span: each span is rendered on its
    speaker's face, and where two speakers' padded spans meet they are cut at
    the frame midway between the two segments."""
    spans: list[Span] = []
    for i, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        who = seg.get("speaker")
        if spans and start - spans[-1].end <= gap and spans[-1].speaker == who:
            spans[-1].end = end
            spans[-1].segments.append(i)
        else:
            spans.append(Span(len(spans), start, end, [i], who))
    out: list[Span] = []
    for span in spans:
        start = max(0.0, span.start - pad)
        end = min(total_seconds, span.end + pad)
        if out and start <= out[-1].end:
            if out[-1].speaker == span.speaker:
                out[-1].end = max(out[-1].end, math.ceil(end * FPS) / FPS)
                out[-1].segments += span.segments
                continue
            # Different speaker: cut at the frame midway between the two segments.
            previous_speech_end = max(float(segments[i]["end"]) for i in out[-1].segments)
            cut = round(((previous_speech_end + span.start) / 2) * FPS) / FPS
            out[-1].end = min(out[-1].end, cut)
            start = cut
        out.append(Span(len(out), math.floor(start * FPS) / FPS if not out or start > out[-1].end else start,
                        math.ceil(end * FPS) / FPS, list(span.segments), span.speaker))
    return [s for s in out if s.end > s.start]


def head_start_ready(ready: float, remaining: float, rate: float, margin: float = 2.0) -> bool:
    """Playback from the start cannot stall when enough media is ready for the
    production rate to keep ahead of real time for the rest of the clip."""
    if remaining <= 0:
        return True
    deficit = max(0.0, remaining * (1.0 - min(rate, 1.0)))
    return ready >= deficit + margin


class Playlist:
    """An HLS EVENT playlist that grows one segment at a time (atomic rewrite)."""

    def __init__(self, directory: Path, name: str = "stream"):
        self.directory = directory
        self.name = name
        self.entries: list[tuple[str, float]] = []
        self.target = 1

    def segment_name(self, index: int) -> str:
        return f"{self.name}-{index:05d}.ts"

    def add(self, filename: str, seconds: float, ended: bool = False) -> None:
        self.entries.append((filename, seconds))
        self.target = max(self.target, int(math.ceil(seconds)))
        self.write(ended)

    def write(self, ended: bool = False) -> None:
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{self.target}",
                 "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:EVENT"]
        for filename, seconds in self.entries:
            lines += [f"#EXTINF:{seconds:.3f},", filename]
        if ended:
            lines.append("#EXT-X-ENDLIST")
        path = self.directory / f"{self.name}.m3u8"
        tmp = path.with_suffix(".m3u8.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(path)


def _ts_segment(video: Path, audio: Path | None, start: float, out: Path) -> None:
    """Mux a piece into an MPEG-TS segment whose timestamps continue the stream."""
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video)]
    if audio is not None:
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    args += ["-c:v", "copy", "-output_ts_offset", f"{start:.3f}", "-f", "mpegts", str(out)]
    result = subprocess.run(args, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1500:])


def _cut_frames(video: Path, left: float, right: float, video_seconds: float, out: Path) -> None:
    """The source frames in [left, right) on the 25 fps grid (clone-padded past EOF)."""
    span = right - left
    seek = min(left, max(0.0, video_seconds - 0.08))
    encode(["-ss", seek, "-i", video, "-an", "-vf",
            f"fps={FPS},tpad=stop_mode=clone:stop_duration={span},trim=end_frame={round(span * FPS)},setpts=PTS-STARTPTS",
            "-r", FPS, "-fps_mode", "cfr"], out)


def _cut_audio(audio: Path, left: float, span: float, out: Path, rate: int = 24000) -> None:
    """`span` seconds of `audio` from `left`, padded with silence past its end."""
    if left >= duration(audio):
        run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono", "-t", span, out])
        return
    run_ffmpeg(["-ss", left, "-i", audio, "-af", f"apad=whole_dur={span},atrim=end={span},asetpts=PTS-STARTPTS",
                "-ar", rate, "-ac", "1", out])


def _shift(segment: dict, offset: float) -> dict:
    """A segment dict (translation or transcript) moved earlier by `offset`."""
    out = dict(segment)
    out["start"] = round(float(segment["start"]) - offset, 3)
    out["end"] = round(float(segment["end"]) - offset, 3)
    if "words" in segment and segment["words"]:
        out["words"] = [{**w, "start": round(float(w["start"]) - offset, 3), "end": round(float(w["end"]) - offset, 3)}
                        for w in segment["words"]]
    return out


def synthesize_span(span: Span, translation: dict, transcript: dict, reference_audio: Path, out: Path,
                    backend: str | None, options: dict | None) -> Path:
    """The span's translated speech on the span's clock (silence before the first
    word, segments at their original onsets, slot fitting), padded to the span."""
    segments = [_shift(translation["segments"][i], span.start) for i in span.segments]
    sub = {"target_language": translation["target_language"], "source_language": translation.get("source_language"),
           "backend": translation.get("backend"), "text": " ".join(s["text"] for s in segments), "segments": segments}
    raw = out.with_name(out.stem + "-raw.wav")
    # A span's last line may run into the next span's first moments when nothing
    # else fits (tts.py); that tail is kept in the span audio and mixed under the
    # next span by assemble_audio.
    tail = float(settings.stream_tail_overlap_seconds)
    options = {**(options or {}), "tail_overlap_seconds": tail}
    # The full, unshifted transcript keeps the voice reference selection on the real clip.
    tts.synthesize(translation=sub, reference_audio=reference_audio, output_path=raw,
                   first_speech_seconds=segments[0]["start"], source_duration_seconds=span.seconds,
                   transcript_segments=transcript.get("segments") or None, backend=backend, options=options)
    _cut_audio(raw, 0.0, span.seconds + tail if duration(raw) > span.seconds + 0.01 else span.seconds, out)
    raw.unlink(missing_ok=True)
    return out


def assemble_audio(placed: list[tuple[float, Path]], total_seconds: float, out: Path, rate: int = 24000) -> None:
    """translated_audio.wav: every span's speech at its offset over silence."""
    import numpy as np
    import soundfile as sf

    timeline = np.zeros(int(round(total_seconds * rate)), dtype=np.float32)
    for start, path in placed:
        if not path.exists():
            continue
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != rate:
            tmp = path.with_suffix(".24k.wav")
            run_ffmpeg(["-i", path, "-ar", rate, "-ac", "1", tmp])
            data, _ = sf.read(str(tmp), dtype="float32", always_2d=False)
            tmp.unlink(missing_ok=True)
        offset = int(round(start * rate))
        end = min(len(timeline), offset + len(data))
        if end > offset:
            timeline[offset:end] += data[: end - offset]
    sf.write(str(out), np.clip(timeline, -1.0, 1.0), rate)


class StreamJob:
    """Per-span translate, render and publish for one job. Synchronous: the
    orchestrator runs it in a worker thread and relays `emit` to the event log."""

    def __init__(self, job_dir: Path, input_path: Path, translation: dict, transcript: dict, *,
                 backend: str, steps: int | None, tts_backend: str | None, options: dict | None,
                 emit: Callable[[str, dict], None], cancel: threading.Event | None = None,
                 window_seconds: float | None = None):
        self.dir = job_dir
        self.input = input_path
        self.translation = translation
        self.transcript = transcript
        self.backend = backend
        self.steps = steps
        self.tts_backend = tts_backend
        self.options = options or {}
        self.emit = emit
        self.cancel = cancel
        self.window = float(window_seconds or settings.stream_window_seconds)
        self.overlap = float(settings.window_overlap_seconds)
        self.root = job_dir / "stream-pieces"
        self.root.mkdir(exist_ok=True)
        from .speakers import face_map

        self.faces = face_map(transcript)      # speaker -> face identity on screen
        self.playlist = Playlist(job_dir)
        self.pieces: list[Piece] = []
        self.total = duration(input_path)
        self.started = time.perf_counter()
        self.rendered_media = 0.0
        self.render_wall = 0.0
        self.timings: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise RuntimeError("streaming translation cancelled")

    def ready_seconds(self) -> float:
        return self.pieces[-1].end if self.pieces else 0.0

    def rate(self) -> float:
        """Media seconds produced per wall second so far (gaps are free, speech costs render time)."""
        elapsed = time.perf_counter() - self.started
        return self.ready_seconds() / elapsed if elapsed > 0 else 0.0

    def _publish(self, piece: Piece) -> None:
        segment = self.root.parent / self.playlist.segment_name(piece.index)
        _ts_segment(piece.path, piece.audio, piece.start, segment)
        self.pieces.append(piece)
        self.playlist.add(segment.name, piece.end - piece.start)
        remaining = self.total - self.ready_seconds()
        rate = self.rate()
        self.emit("stream_segment", {
            "index": piece.index, "kind": piece.kind, "start": round(piece.start, 3), "end": round(piece.end, 3),
            "ready_seconds": round(self.ready_seconds(), 3), "total_seconds": round(self.total, 3),
            "rate": round(rate, 3), "render_seconds": round(piece.render_seconds, 2),
            "elapsed_seconds": round(time.perf_counter() - self.started, 2),
            "can_play": head_start_ready(self.ready_seconds(), remaining, rate),
            "playlist": self.playlist.directory.joinpath(f"{self.playlist.name}.m3u8").name,
        })

    def _gap(self, index: int, start: float, end: float, original_audio: Path) -> None:
        """Frames with no speech pass through from the source, with the original sound."""
        if end - start < 1.0 / FPS:
            return
        clip = self.root / f"gap-{index:05d}.mp4"
        sound = self.root / f"gap-{index:05d}.wav"
        _cut_frames(self.input, start, end, self.total, clip)
        _cut_audio(original_audio, start, end - start, sound, rate=48000)
        self._publish(Piece(index, start, end, "gap", clip, sound))

    def _render_span(self, span: Span, span_audio: Path, next_index: int) -> int:
        """Render the span in windows of `self.window` seconds with `self.overlap`
        context on both sides; publish each window's core as soon as it exists."""
        from .lipsync import prepare as prepare_window

        pad = round(self.overlap * FPS) / FPS
        cores = []
        t = span.start
        while t < span.end - 1e-6:
            core_end = min(span.end, t + self.window)
            if span.end - core_end < 1.0 and core_end < span.end:      # avoid a sub-second tail window
                core_end = span.end
            cores.append((t, core_end))
            t = core_end
        windows = [(s, e, max(span.start, s - pad), min(span.end, e + pad)) for s, e in cores]

        def overrides(left: float) -> dict:
            merged: dict[str, Any] = {"face_track_source": str(self.input), "face_track_offset_frames": int(round(left * FPS))}
            if span.speaker in self.faces:
                merged["face_track_identity"] = self.faces[span.speaker]
            if self.steps is not None:
                merged["num_inference_steps"] = self.steps
            return merged

        def cut(i: int):
            s, e, left, right = windows[i]
            source = self.root / f"source-{span.index:03d}-{i:03d}.mp4"
            sound = self.root / f"audio-{span.index:03d}-{i:03d}.wav"
            _cut_frames(self.input, left, right, self.total, source)
            _cut_audio(span_audio, left - span.start, right - left, sound)
            return source, sound

        def cut_and_prepare(i: int):
            try:
                source, sound = cut(i)
                if self.backend == "latentsync":
                    prepare_window(self.backend, source, sound, quality_overrides=overrides(windows[i][2]))
            except Exception as exc:
                log.warning("span %d window %d prefetch failed: %s", span.index, i, str(exc)[-200:])

        prefetch: dict[int, threading.Thread] = {}
        for i, (s, e, left, right) in enumerate(windows):
            self._check_cancel()
            thread = prefetch.pop(i, None)
            if thread is not None:
                thread.join()
            source = self.root / f"source-{span.index:03d}-{i:03d}.mp4"
            sound = self.root / f"audio-{span.index:03d}-{i:03d}.wav"
            if not (source.exists() and sound.exists()):
                cut(i)
            if i + 1 < len(windows):
                thread = threading.Thread(target=cut_and_prepare, args=(i + 1,), daemon=True, name=f"stream-prefetch-{span.index}-{i + 1}")
                thread.start()
                prefetch[i + 1] = thread
            rendered = self.root / f"rendered-{span.index:03d}-{i:03d}.mp4"
            started = time.perf_counter()
            lipsync.run(self.backend, source, sound, rendered, quality_overrides=overrides(left))
            render_seconds = time.perf_counter() - started
            part = self.root / f"part-{span.index:03d}-{i:03d}.mp4"
            first, last = round((s - left) * FPS), round((e - left) * FPS)
            encode(["-i", rendered, "-an", "-vf", f"fps={FPS},trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS",
                    "-r", FPS, "-fps_mode", "cfr"], part)
            part_audio = self.root / f"part-{span.index:03d}-{i:03d}.wav"
            _cut_audio(span_audio, s - span.start, e - s, part_audio, rate=48000)
            self.rendered_media += e - s
            self.render_wall += render_seconds
            self.timings.append({"span": span.index, "window": i, "seconds": round(e - s, 2), "render_seconds": round(render_seconds, 2)})
            self._publish(Piece(next_index, s, e, "speech", part, part_audio, render_seconds))
            next_index += 1
            for scratch in (source, sound, rendered):
                scratch.unlink(missing_ok=True)
        for thread in prefetch.values():
            thread.join()
        return next_index

    # ------------------------------------------------------------------- run
    def run(self, reference_audio: Path, original_audio: Path) -> dict:
        segments = self.transcript.get("segments") or []
        spans = speech_spans(segments, self.total, settings.stream_span_gap_seconds, settings.stream_span_pad_seconds)
        self.emit("stream_plan", {"spans": [{"start": s.start, "end": s.end, "segments": len(s.segments), "speaker": s.speaker,
                                             "face": self.faces.get(s.speaker)} for s in spans],
                                  "speech_seconds": round(sum(s.seconds for s in spans), 2), "total_seconds": round(self.total, 2),
                                  "window_seconds": self.window})
        index = 0
        cursor = 0.0
        tts_seconds = 0.0
        from concurrent.futures import ThreadPoolExecutor

        def synthesize(span: Span) -> tuple[Path, float]:
            started = time.perf_counter()
            path = synthesize_span(span, self.translation, self.transcript, reference_audio,
                                   self.root / f"span-{span.index:03d}.wav", self.tts_backend, self.options)
            return path, time.perf_counter() - started

        # Speech for span N+1 is synthesized (speech GPU) while span N renders (pool GPUs),
        # so the renderer never waits for TTS after the first span.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-tts") as pool:
            futures = {spans[0].index: pool.submit(synthesize, spans[0])} if spans else {}
            for position, span in enumerate(spans):
                self._check_cancel()
                if position + 1 < len(spans):
                    futures[spans[position + 1].index] = pool.submit(synthesize, spans[position + 1])
                if span.start > cursor:
                    self._gap(index, cursor, span.start, original_audio)
                    index += 1
                span_audio, took = futures.pop(span.index).result()
                tts_seconds += took
                self.emit("stream_span_audio", {"span": span.index, "seconds": round(span.seconds, 2), "tts_seconds": round(took, 2)})
                index = self._render_span(span, span_audio, index)
                cursor = span.end
        if cursor < self.total:
            self._gap(index, cursor, self.total, original_audio)
            index += 1
        self.playlist.write(ended=True)
        # Final artifacts for the regular mux stage.
        listing = self.root / "concat.txt"
        listing.write_text("\n".join(f"file '{p.path.name}'" for p in self.pieces))
        run_ffmpeg(["-f", "concat", "-safe", "1", "-i", listing, "-an", "-c:v", "copy", "-movflags", "+faststart",
                    self.dir / "lipsynced.mp4"])
        assemble_audio([(s.start, self.root / f"span-{s.index:03d}.wav") for s in spans], self.total,
                       self.dir / "translated_audio.wav")
        changed = sum(p.end - p.start for p in self.pieces if p.kind == "speech")
        return {"spans": len(spans), "pieces": len(self.pieces), "speech_seconds": round(changed, 2),
                "total_seconds": round(self.total, 2), "changed_fraction": round(changed / self.total, 3) if self.total else None,
                "render_wall_seconds": round(self.render_wall, 2), "tts_seconds": round(tts_seconds, 2),
                "elapsed_seconds": round(time.perf_counter() - self.started, 2),
                "first_segment_seconds": self.timings[0]["render_seconds"] if self.timings else None,
                "windows": self.timings}
