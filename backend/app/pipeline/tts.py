"""Stage 4 — voice cloning / TTS via XTTS-v2.

Generates `translated_audio.wav` from the translated text, using the original
speaker's voice as a reference.

CPU realism: XTTS-v2 runs at roughly 0.3–0.7× realtime on a 16-core Xeon.
A 3-second translation typically takes 5–10 s to synthesize.

Simplifications in v1 (documented limitations):
- Uses the whole Stage 1 WAV as the speaker reference rather than picking the
  longest clean segment. Works well enough when the source is single-speaker
  and well-miked.
- Generates the entire translated text in one pass. No per-segment timing
  alignment, so the TTS duration will not exactly match the source video.
- No loudness normalization or post-processing. Add in a follow-up if needed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "language": self.language,
            "reference_audio": self.reference_audio,
            "output_path": self.output_path,
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


def synthesize(
    translation: dict,
    reference_audio: Path,
    output_path: Path,
    first_speech_seconds: float | None = None,
    source_duration_seconds: float | None = None,
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

    tts = _get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=str(reference_audio),
        language=XTTS_LANG_CODES[tgt],
        file_path=str(output_path),
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError("XTTS produced no output")

    # XTTS wraps its speech in ~200ms clicks/breaths and trailing silence that
    # can stretch the file to 2–3× the actual spoken duration. Without this
    # post-step, the downstream mux either truncates mid-word (`-shortest`) or
    # forces the video to freeze-frame through multiple seconds of dead air.
    try:
        _trim_to_speech(output_path)
    except Exception as e:
        # Best-effort: if trimming fails, keep the original audio so we don't
        # drop the stage over a cosmetic issue.
        log.warning("silence trim failed (%s); keeping untrimmed XTTS output", e)

    # Optional speed-up: if TTS is a bit longer than the remaining source
    # video after alignment, pitch-preserving time-stretch squeezes it to
    # fit. Kept conservative — aggressive stretch destroys TTS quality.
    if source_duration_seconds is not None:
        target = source_duration_seconds - (first_speech_seconds or 0.0)
        if target > 0:
            try:
                _maybe_time_stretch(output_path, target_duration=target)
            except Exception as e:
                log.warning("time-stretch failed (%s); keeping untouched TTS", e)

    # Align to source: prepend silence so the TTS first-spoken frame sits at
    # the same clock time as the original first-spoken frame. Downstream
    # mux still freeze-pads the video if this puts us past its duration.
    if first_speech_seconds and first_speech_seconds > 0.01:
        try:
            _prepend_silence(output_path, seconds=first_speech_seconds)
        except Exception as e:
            log.warning("silence-prepend failed (%s); keeping un-aligned TTS", e)

    return TTSResult(
        backend="xtts_v2",
        language=tgt,
        reference_audio=reference_audio.name,
        output_path=output_path.name,
    )


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

    tmp = audio_path.with_suffix(audio_path.suffix + ".trim")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(audio_path),
        "-af", f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atrim failed: {proc.stderr.decode(errors='replace')}")
    tmp.replace(audio_path)
    log.info(
        "trimmed %s: %.2fs -> %.2fs (kept %.2f-%.2f)",
        audio_path.name, total, end - start, start, end,
    )


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
