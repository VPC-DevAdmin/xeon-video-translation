"""Stage 4 — voice cloning / TTS via XTTS-v2.

Generates `translated_audio.wav` from the translated text, using the original
speaker's voice as a reference.

CPU realism: XTTS-v2 runs at roughly 0.3–0.7× realtime on a 16-core Xeon.
A 3-second translation typically takes 5–10 s to synthesize.

Processing pipeline (per request):

1. Pick the cleanest contiguous span of source speech (via Whisper word
   timestamps) as the XTTS voice reference. Fall back to the whole audio
   when word timestamps aren't available or no ≥3 s clean span exists.
2. Per-segment synthesis when the transcript has multiple segments —
   preserves the source clip's pause structure. Single-shot otherwise.
3. Optional formant-preserving time-stretch (rubberband) if assembled
   audio overshoots the source video's available window.
4. Prepend silence to align the first spoken frame with the source.
5. Loudness normalization (EBU R128 / −16 LUFS) so dialog lands at a
   consistent broadcast level regardless of the XTTS take.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import settings

log = logging.getLogger(__name__)


# BCP-47 -> XTTS-v2 language codes. XTTS supports these; our NLLB language
# picker is a superset, so some translations may not be synthesizable.
XTTS_LANG_CODES: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "hu": "hu", "ko": "ko", "ja": "ja",
    "hi": "hi",
    "zh": "zh-cn",  # XTTS uses the regional tag
}


class TTSError(RuntimeError):
    pass


@dataclass
class TTSResult:
    backend: str
    language: str
    reference_audio: str
    output_path: str
    # True when we ran per-segment synthesis, false for the single-shot path.
    # Surfaced in the pipeline meta.json so the UI can show "preserved
    # pause structure" when it's true.
    per_segment: bool = False
    segments_synthesized: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "language": self.language,
            "reference_audio": self.reference_audio,
            "output_path": self.output_path,
            "per_segment": self.per_segment,
            "segments_synthesized": self.segments_synthesized,
        }


_tts = None
_tts_lock = Lock()


def _get_tts():
    """Lazy-load XTTS-v2. First call downloads ~1.8 GB into HF_HOME."""
    global _tts
    if _tts is not None:
        return _tts
    with _tts_lock:
        if _tts is not None:
            return _tts

        # Coqui prompts for CPML license agreement on first load; pre-accept.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # Cache path to keep weights alongside other models in MODEL_CACHE_DIR.
        os.environ.setdefault(
            "TTS_HOME",
            str(settings.model_cache_dir / "coqui-tts"),
        )

        from TTS.api import TTS  # noqa: N811

        model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to("cpu")
        _tts = model
        return _tts


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #


def synthesize(
    translation: dict,
    reference_audio: Path,
    output_path: Path,
    first_speech_seconds: float | None = None,
    source_duration_seconds: float | None = None,
    transcript_segments: list[dict] | None = None,
) -> TTSResult:
    """Generate speech for `translation` using `reference_audio` as the voice.

    `translation` is the dict written by Stage 3 (see translate.py).

    `first_speech_seconds` (from the transcript) lets us prepend a matching
    amount of silence so the TTS first-frame lines up with the source's
    first spoken frame — otherwise a 1 s "speaker pauses then talks" clip
    becomes a "speaker starts talking immediately" clip.

    `source_duration_seconds` enables formant-preserving time-stretch
    (rubberband) when the post-trim TTS is still a bit longer than the
    remaining source video. Stretching is skipped when the ratio would be
    aggressive — we'd rather freeze-pad video than produce chipmunk audio.

    `transcript_segments` (from Stage 2) enables two quality wins:
    - smart reference selection (longest clean contiguous speech span)
    - per-segment synthesis preserving original pause structure
    When missing, we fall back to single-shot synthesis on the whole
    translated text.
    """
    tgt = translation.get("target_language", "").lower()
    if tgt not in XTTS_LANG_CODES:
        raise TTSError(
            f"XTTS-v2 does not support target language {tgt!r}. "
            f"Supported: {sorted(XTTS_LANG_CODES)}"
        )

    text = (translation.get("text") or "").strip()
    if not text:
        raise TTSError("translation is empty — nothing to synthesize")

    if not reference_audio.exists():
        raise TTSError(f"reference audio missing: {reference_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Reference-audio selection -------------------------------------
    ref_for_xtts, ref_label = _select_reference(
        reference_audio, transcript_segments, output_path.parent
    )

    # --- 2. Decide per-segment vs single-shot ----------------------------
    trans_segs = translation.get("segments") or []
    can_do_per_segment = (
        transcript_segments is not None
        and len(transcript_segments) > 1
        and len(trans_segs) == len(transcript_segments)
    )

    language = XTTS_LANG_CODES[tgt]
    segments_synthesized = 0

    if can_do_per_segment:
        log.info("per-segment TTS: %d segments", len(trans_segs))
        segments_synthesized = _synthesize_per_segment(
            translation_segments=trans_segs,
            transcript_segments=transcript_segments,
            reference_audio=ref_for_xtts,
            language=language,
            output_path=output_path,
        )
    else:
        log.info("single-shot TTS (segments=%d)", len(trans_segs))
        _synthesize_whole(
            text=text,
            reference_audio=ref_for_xtts,
            language=language,
            output_path=output_path,
        )
        # Single-shot: XTTS wraps its speech in clicks/breaths and trailing
        # silence — trim those away. Per-segment path already trimmed each
        # segment individually, so this step only runs here.
        try:
            _trim_to_speech(output_path)
        except Exception as e:
            log.warning("silence trim failed (%s); keeping untrimmed output", e)
        segments_synthesized = 1

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError("TTS produced no output")

    # --- 3. Optional speed-up: fit assembled audio into source window ----
    if source_duration_seconds is not None:
        target = source_duration_seconds - (first_speech_seconds or 0.0)
        if target > 0:
            try:
                _maybe_time_stretch(output_path, target_duration=target)
            except Exception as e:
                log.warning("time-stretch failed (%s); keeping untouched TTS", e)

    # --- 4. Align to source: prepend silence so TTS first-frame lines up  -
    if first_speech_seconds and first_speech_seconds > 0.01:
        try:
            _prepend_silence(output_path, seconds=first_speech_seconds)
        except Exception as e:
            log.warning("silence-prepend failed (%s); keeping un-aligned TTS", e)

    # --- 5. Loudness normalization to -16 LUFS ---------------------------
    try:
        _loudnorm(output_path)
    except Exception as e:
        log.warning("loudness normalization failed (%s); shipping unnormalized", e)

    return TTSResult(
        backend="xtts_v2",
        language=tgt,
        reference_audio=ref_label,
        output_path=output_path.name,
        per_segment=can_do_per_segment,
        segments_synthesized=segments_synthesized,
    )


# --------------------------------------------------------------------------- #
# Reference-audio selection
#
# XTTS sounds noticeably cleaner when its speaker reference is a single
# contiguous span of clean speech vs. an audio file that begins with
# silence and/or noise. We use Whisper's word-level timestamps to find the
# longest span where consecutive words have < _REF_GAP_SECONDS between
# them, then trim the source WAV to that span.
# --------------------------------------------------------------------------- #

_REF_GAP_SECONDS = 0.30          # word-to-word gap threshold for "contiguous"
_REF_MIN_SPAN_SECONDS = 3.0      # XTTS's documented minimum reference length
_REF_MAX_SPAN_SECONDS = 12.0     # longer doesn't help; cap to keep loads fast


def _select_reference(
    reference_audio: Path,
    transcript_segments: list[dict] | None,
    work_dir: Path,
) -> tuple[Path, str]:
    """Return (path_to_use, label_for_logs).

    If word timestamps are available and there's a ≥3 s clean span, write
    a trimmed WAV into `work_dir` and return that. Otherwise fall through
    to the original reference.
    """
    if not transcript_segments:
        return reference_audio, reference_audio.name

    # Flatten all words across segments.
    words: list[dict] = []
    for seg in transcript_segments:
        for w in (seg.get("words") or []):
            if w.get("start") is not None and w.get("end") is not None:
                words.append(w)
    if len(words) < 3:
        return reference_audio, reference_audio.name

    span = _longest_contiguous_word_span(words, max_gap=_REF_GAP_SECONDS)
    if span is None:
        return reference_audio, reference_audio.name
    span_start, span_end = span
    span_len = span_end - span_start
    if span_len < _REF_MIN_SPAN_SECONDS:
        log.info(
            "longest clean speech span is only %.2fs — using whole source "
            "as XTTS reference",
            span_len,
        )
        return reference_audio, reference_audio.name

    # Cap the span length — XTTS doesn't benefit from extremely long refs.
    if span_len > _REF_MAX_SPAN_SECONDS:
        span_end = span_start + _REF_MAX_SPAN_SECONDS
        span_len = _REF_MAX_SPAN_SECONDS

    trimmed = work_dir / "xtts_reference.wav"
    try:
        _ffmpeg_atrim(reference_audio, trimmed, span_start, span_end)
    except Exception as e:
        log.warning("failed to cut reference audio (%s); using whole clip", e)
        return reference_audio, reference_audio.name
    log.info(
        "selected XTTS reference: %.2fs span (%.2f–%.2fs of original)",
        span_len, span_start, span_end,
    )
    return trimmed, f"trimmed {span_len:.2f}s"


def _longest_contiguous_word_span(
    words: list[dict], max_gap: float,
) -> tuple[float, float] | None:
    """Find the longest window of consecutive words where no pair is
    separated by > `max_gap` seconds. Returns (start, end) in seconds.
    """
    if not words:
        return None
    # Sort by start in case segments arrived unordered.
    words = sorted(words, key=lambda w: float(w["start"]))
    best_s = float(words[0]["start"])
    best_e = float(words[0]["end"])
    cur_s = best_s
    cur_e = best_e
    for w in words[1:]:
        s, e = float(w["start"]), float(w["end"])
        if s - cur_e <= max_gap:
            cur_e = e
        else:
            if cur_e - cur_s > best_e - best_s:
                best_s, best_e = cur_s, cur_e
            cur_s, cur_e = s, e
    if cur_e - cur_s > best_e - best_s:
        best_s, best_e = cur_s, cur_e
    return best_s, best_e


# --------------------------------------------------------------------------- #
# Per-segment synthesis
#
# Produces an output WAV where each segment's translated audio starts at
# roughly the same clock time as its source counterpart. Inter-segment
# pauses come from the original transcript — so a "Hi. …how are you?"
# source rhythm is preserved rather than being collapsed to a single
# continuous utterance.
# --------------------------------------------------------------------------- #

_MIN_SEG_TEXT_LEN = 3  # chars; skip segments shorter than this (filler, noise)


def _synthesize_per_segment(
    translation_segments: list[dict],
    transcript_segments: list[dict],
    reference_audio: Path,
    language: str,
    output_path: Path,
) -> int:
    """Synthesize each translation segment and splice into a single WAV.

    Returns the number of segments that were actually synthesized (some
    may be dropped for being too short or failing XTTS).
    """
    work_dir = output_path.parent / f"_tts_work_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        seg_paths: list[Path | None] = []
        for i, (tr_seg, src_seg) in enumerate(
            zip(translation_segments, transcript_segments)
        ):
            text = (tr_seg.get("text") or "").strip()
            if len(text) < _MIN_SEG_TEXT_LEN:
                log.info("segment %d skipped (text %r too short)", i, text)
                seg_paths.append(None)
                continue

            seg_path = work_dir / f"seg_{i:03d}.wav"
            try:
                _xtts_to_file(text, reference_audio, language, seg_path)
            except Exception as e:
                log.warning("segment %d XTTS failed (%s); skipping", i, e)
                seg_paths.append(None)
                continue

            try:
                _trim_to_speech(seg_path)
            except Exception:
                pass  # keep untrimmed — better than dropping the segment

            seg_paths.append(seg_path)

        # Fall back gracefully if nothing got synthesized.
        if not any(seg_paths):
            raise TTSError("no segments produced any TTS output")

        _assemble_timeline(
            seg_paths=seg_paths,
            transcript_segments=transcript_segments,
            output_path=output_path,
        )
        return sum(1 for p in seg_paths if p is not None)
    finally:
        # Clean up temp dir. Best-effort — if this fails we just leave it.
        try:
            import shutil as _sh
            _sh.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


def _assemble_timeline(
    seg_paths: list[Path | None],
    transcript_segments: list[dict],
    output_path: Path,
) -> None:
    """Concat segment WAVs with inter-segment silences derived from the
    source transcript's between-segment gaps. Writes to `output_path`.
    """
    # Build the ffmpeg filter graph in three lists:
    #   inputs_list:  -i file.wav for each segment that exists
    #   filter_parts: labeled audio streams ready for concat
    inputs: list[Path] = []
    filter_parts: list[str] = []
    concat_labels: list[str] = []

    # First usable segment carries the "start of output"; no leading silence
    # (the caller adds first_speech_seconds later if needed).
    first_real = next(
        (i for i, p in enumerate(seg_paths) if p is not None), None,
    )
    if first_real is None:
        raise TTSError("no segments to assemble")

    last_src_end: float | None = None
    for i, seg_path in enumerate(seg_paths):
        if seg_path is None:
            # If this segment was dropped, we still want a silence gap
            # representing its source duration so later segments stay
            # anchored to their source start times. Use the source seg
            # duration as the gap length.
            src_dur = float(transcript_segments[i].get("end", 0.0)) \
                      - float(transcript_segments[i].get("start", 0.0))
            if src_dur > 0.05:
                label = f"gap_skip_{i}"
                filter_parts.append(
                    f"anullsrc=r=24000:cl=mono,atrim=duration={src_dur:.3f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
                concat_labels.append(label)
            last_src_end = float(transcript_segments[i].get("end", last_src_end or 0.0))
            continue

        if i != first_real and last_src_end is not None:
            src_start = float(transcript_segments[i].get("start", last_src_end))
            gap = src_start - last_src_end
            if gap > 0.05:
                label = f"gap_{i}"
                filter_parts.append(
                    f"anullsrc=r=24000:cl=mono,atrim=duration={gap:.3f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
                concat_labels.append(label)

        # Convert each segment to the target 24 kHz mono to keep concat happy,
        # regardless of XTTS's sample rate. `aformat` emits a fresh label.
        input_idx = len(inputs)
        inputs.append(seg_path)
        label = f"seg_{i}"
        filter_parts.append(
            f"[{input_idx}:a]aformat=sample_rates=24000:channel_layouts=mono[{label}]"
        )
        concat_labels.append(label)
        last_src_end = float(transcript_segments[i].get("end", last_src_end or 0.0))

    concat_chain = "".join(f"[{lbl}]" for lbl in concat_labels)
    filter_parts.append(
        f"{concat_chain}concat=n={len(concat_labels)}:v=0:a=1[out]"
    )
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in inputs:
        cmd.extend(["-i", str(p)])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(output_path),
    ])

    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise TTSError(
            f"ffmpeg assemble failed: {proc.stderr.decode(errors='replace')[-1200:]}"
        )


# --------------------------------------------------------------------------- #
# XTTS call helpers
# --------------------------------------------------------------------------- #


def _xtts_to_file(
    text: str, reference_audio: Path, language: str, output: Path,
) -> None:
    """Single XTTS call that writes to `output`."""
    tts = _get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=str(reference_audio),
        language=language,
        file_path=str(output),
    )
    if not output.exists() or output.stat().st_size == 0:
        raise TTSError(f"XTTS produced no output at {output}")


def _synthesize_whole(
    text: str, reference_audio: Path, language: str, output_path: Path,
) -> None:
    """Legacy single-shot path. One XTTS call for the entire translation."""
    _xtts_to_file(text, reference_audio, language, output_path)


# --------------------------------------------------------------------------- #
# Silence trimming
# --------------------------------------------------------------------------- #

_SILENCE_THRESHOLD_DB = -25.0   # XTTS background-noise floor sits around -30 dBFS
_MIN_SILENCE_SECONDS = 0.10     # how long a quiet stretch needs to be to count
_HEADROOM_SECONDS = 0.05        # pad on either side of the kept span
_MIN_SPEECH_SECONDS = 0.30      # below this, assume detection failed and skip


def _trim_to_speech(audio_path: Path) -> None:
    """Trim `audio_path` in place to the longest non-silent span.

    XTTS tends to emit a short click at the start, a long pause, the actual
    speech, more silence, and a trailing blip. `silenceremove` can't handle
    that shape because the click is above any reasonable threshold.

    Instead: find silence boundaries via ffmpeg `silencedetect`, derive the
    non-silent spans, pick the longest one (the real speech), and copy that
    range into the original path via `ffmpeg -af atrim=...`.
    """
    spans = _non_silent_spans(audio_path)
    speech = [(s, e) for s, e in spans if (e - s) >= _MIN_SPEECH_SECONDS]
    if not speech:
        log.info("no speech span detected; leaving %s untouched", audio_path.name)
        return

    start, end = max(speech, key=lambda se: se[1] - se[0])
    total = _probe_duration(audio_path)
    start = max(0.0, start - _HEADROOM_SECONDS)
    end = min(total, end + _HEADROOM_SECONDS)

    _ffmpeg_atrim(audio_path, audio_path, start, end)
    log.info(
        "trimmed %s: %.2fs -> %.2fs (kept %.2f-%.2f)",
        audio_path.name, total, end - start, start, end,
    )


def _ffmpeg_atrim(src: Path, dst: Path, start: float, end: float) -> None:
    """Write `src[start:end]` to `dst`. In-place safe (uses a temp path)."""
    tmp = dst.with_suffix(dst.suffix + ".trim")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(src),
        "-af", f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atrim failed: {proc.stderr.decode(errors='replace')}")
    tmp.replace(dst)


def _probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        timeout=30,
    )
    return float(out.decode().strip())


def _non_silent_spans(path: Path) -> list[tuple[float, float]]:
    """Return [(start, end), ...] of non-silent spans inside `path`."""
    duration = _probe_duration(path)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-af", f"silencedetect=noise={_SILENCE_THRESHOLD_DB}dB:duration={_MIN_SILENCE_SECONDS}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end_token = line.split("silence_end:")[1].split("|")[0].strip()
            silences.append((start, float(end_token)))
            start = None

    # Merge silences < 150ms apart so brief speech bumps don't create
    # noise-sized "speech" spans between two real pauses.
    merged: list[tuple[float, float]] = []
    for s, e in silences:
        if merged and s - merged[-1][1] < 0.15:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            spans.append((cursor, s))
        cursor = e
    if cursor < duration:
        spans.append((cursor, duration))
    return spans


# --------------------------------------------------------------------------- #
# Time-stretch (rubberband)
# --------------------------------------------------------------------------- #

# Don't stretch beyond this — below ~0.85x speed real-time the formants
# start bunching up and the result sounds "fast-chipmunky". The user
# experience is worse than freeze-padding the video, which we fall back to.
_MIN_TIME_RATIO = 0.85


def _maybe_time_stretch(audio_path: Path, target_duration: float) -> None:
    """Run rubberband to shorten `audio_path` to ~`target_duration` seconds.

    No-op when the current duration is already within the target or when
    the required ratio is too aggressive. Requires `rubberband-cli` on
    PATH — which the backend Docker image installs via apt.
    """
    import shutil as _shutil

    if _shutil.which("rubberband") is None:
        log.info("rubberband-cli not installed; skipping time-stretch")
        return

    current = _probe_duration(audio_path)
    if current <= target_duration + 0.05:
        # Already at or under the target — nothing to do.
        return

    ratio = target_duration / current
    if ratio < _MIN_TIME_RATIO:
        log.info(
            "time-stretch would need %.2fx (< min %.2fx); will let mux freeze-pad instead",
            ratio, _MIN_TIME_RATIO,
        )
        return

    tmp = audio_path.with_suffix(audio_path.suffix + ".stretch")
    cmd = [
        "rubberband",
        "--time", f"{ratio:.4f}",
        # `--formant` preserves formant frequencies during the stretch so
        # the speaker still sounds like themselves.
        "--formant",
        str(audio_path),
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rubberband exit {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')[-500:]}"
        )
    tmp.replace(audio_path)
    log.info(
        "time-stretched %s: %.2fs -> %.2fs (ratio %.3f)",
        audio_path.name, current, target_duration, ratio,
    )


# --------------------------------------------------------------------------- #
# Source-aligned silence prepend
# --------------------------------------------------------------------------- #


def _prepend_silence(audio_path: Path, seconds: float) -> None:
    """Prepend `seconds` of silence to `audio_path` (in place) via ffmpeg.

    Matches sample rate and channel layout of the input. The result is an
    audio file whose first spoken frame sits at `seconds` — aligning the
    TTS with the source clip's pre-speech silence.
    """
    if seconds <= 0:
        return

    probe = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "default=nw=1",
         str(audio_path)],
        capture_output=True, timeout=30,
    )
    sr, ch = 24000, 1
    for line in probe.stdout.decode(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k == "sample_rate":
                sr = int(v)
            elif k == "channels":
                ch = int(v)

    tmp = audio_path.with_suffix(audio_path.suffix + ".pad")
    channel_layout = "mono" if ch == 1 else "stereo"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl={channel_layout}",
        "-i", str(audio_path),
        "-filter_complex",
        f"[0:a]atrim=duration={seconds:.3f}[lead];[lead][1:a]concat=n=2:v=0:a=1[out]",
        "-map", "[out]",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg prepend-silence failed: {proc.stderr.decode(errors='replace')[-500:]}"
        )
    tmp.replace(audio_path)
    log.info("prepended %.3fs silence to align with source speech-start", seconds)


# --------------------------------------------------------------------------- #
# Loudness normalization (EBU R128)
#
# Targets -16 LUFS integrated + -1.5 dBTP peak + LRA 11 LU — commonly
# cited broadcast-dialog standard. Single-pass loudnorm is used because
# two-pass adds a full analysis pass for marginal accuracy gain on clips
# this short.
# --------------------------------------------------------------------------- #

_LOUDNORM_TARGET_I = -16.0       # LUFS integrated
_LOUDNORM_TARGET_TP = -1.5       # dBTP true-peak ceiling
_LOUDNORM_TARGET_LRA = 11.0      # loudness range


def _loudnorm(audio_path: Path) -> None:
    """Apply EBU R128 loudnorm to `audio_path` in place."""
    tmp = audio_path.with_suffix(audio_path.suffix + ".lnorm")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(audio_path),
        "-af",
        f"loudnorm=I={_LOUDNORM_TARGET_I}:TP={_LOUDNORM_TARGET_TP}:"
        f"LRA={_LOUDNORM_TARGET_LRA}",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg loudnorm failed: {proc.stderr.decode(errors='replace')[-500:]}"
        )
    tmp.replace(audio_path)
    log.info("loudness-normalized %s to %.1f LUFS", audio_path.name, _LOUDNORM_TARGET_I)
