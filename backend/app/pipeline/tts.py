"""Stage 4 — voice cloning / TTS.

Two backends are supported, selected per-request via ``tts_backend`` on the
job submission (form field) or the ``TTS_BACKEND`` env default:

- ``xtts``  — Coqui XTTS-v2 (default). 16 languages. CPML-licensed weights;
              ~1.8 GB; ~0.3–0.7× realtime on a 16-core Xeon.

- ``f5tts`` — F5-TTS. Newer flow-matching-on-DiT architecture. Base
              checkpoint is trained on EN + ZH only; anything else falls
              through to community fine-tunes or raises a clear error.
              Typically cleaner prosody than XTTS for its supported
              languages; comparable wall-clock on CPU.

XTTS processing pipeline (per request):

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

F5-TTS processing pipeline (per request):

1. Single-shot generation on the full translated text, conditioned on the
   source reference audio and its Whisper transcript (F5-TTS needs both).
2. Silence trim on the generated output.
3. Same steps 3–5 as XTTS (time-stretch, silence prepend, loudnorm).

F5-TTS does NOT currently use per-segment synthesis or smart reference
selection — that's a follow-up once the base backend proves out. For now,
when pause-structure preservation matters, stick with XTTS.
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


# F5-TTS base checkpoint language support. The F5-TTS_v1 base is trained
# on Emilia (EN + ZH). Community fine-tunes exist for JA/FR/DE/HI/... but
# we don't pre-download or validate those yet — if users want them they
# can set F5TTS_MODEL to the fine-tune's HF repo id and we'll try it.
F5TTS_BASE_LANGS: set[str] = {"en", "zh"}


# IndicF5 — F5-TTS architecture fine-tuned on IndicVoices-R by AI4Bharat.
# Handles the major Indic languages natively (as opposed to XTTS-v2 which
# produces phoneme approximations on Devanagari). Model loads ~1.5 GB on
# first use via transformers + trust_remote_code=True.
#
# BCP-47 → IndicF5 language tag. We pass the tag into the model call so
# it can pick the right phoneme set; the values match what AI4Bharat's
# inference code expects (see their HF model card).
INDICF5_LANG_CODES: dict[str, str] = {
    "hi": "hi",   # Hindi
    "bn": "bn",   # Bengali
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "mr": "mr",   # Marathi
    "gu": "gu",   # Gujarati
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "pa": "pa",   # Punjabi
    "or": "or",   # Odia / Oriya
    "as": "as",   # Assamese
}


# --------------------------------------------------------------------------- #
# Language-aware TTS backend selection
#
# Different TTS models have different language competencies. XTTS-v2 is
# multilingual in theory (16 languages) but its training corpus is
# Latin-script-heavy; on Devanagari (Hindi, Bengali, etc.) and CJK
# (Japanese, Korean) it produces phoneme-level approximations rather
# than coherent native speech. These preferences encode the empirical
# quality matrix as of 2026-04:
#
# - IndicF5 (ai4bharat/IndicF5): F5-TTS fine-tuned on IndicVoices-R;
#   handles Hindi/Bengali/Tamil/Telugu/Marathi/etc. natively.
#   2026-04: NOT YET INTEGRATED. Falls back to XTTS with a warning.
#
# - F5-TTS base: strong on EN + ZH.
#
# - StyleTTS2 / MeloTTS: best-in-class for Japanese and Korean.
#   2026-04: NOT YET INTEGRATED.
#
# - XTTS-v2: good default for Spanish/Portuguese/Italian/German/French
#   and usable for ~16 languages overall. The fallback backend.
#
# When `tts_backend="auto"` is requested, `_select_tts_backend_for_language`
# walks the language's preference list and returns the first backend
# that's actually installed. When a preferred backend is listed but not
# yet integrated, a WARNING is logged pointing at which follow-up PR
# will add it; the run continues on the best-available fallback.

# (languages, [backends in preference order]). First match wins.
_LANG_TTS_PREFERENCES: list[tuple[set[str], list[str]]] = [
    # Indic languages — IndicF5 preferred
    (
        {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"},
        ["indicf5", "xtts"],
    ),
    # Chinese — F5-TTS base is trained on ZH
    ({"zh", "zh-cn"}, ["f5tts", "xtts"]),
    # English — F5-TTS base's strongest language; great for testing
    ({"en"}, ["f5tts", "xtts"]),
    # Japanese — specialized backends first
    ({"ja"}, ["styletts2-jp", "melotts", "xtts"]),
    # Korean
    ({"ko"}, ["melotts", "styletts2-ko", "xtts"]),
    # Romance + Germanic + Cyrillic + Arabic + Turkish — XTTS is fine
    (
        {"es", "pt", "it", "de", "nl", "pl", "cs", "ru", "tr", "ar", "hu"},
        ["xtts"],
    ),
    # French — either works well
    ({"fr"}, ["xtts", "f5tts"]),
]

# Backends implemented in the current codebase. Update as new backend
# PRs land. The selector uses this as the "is this installed?" check.
_INTEGRATED_BACKENDS: set[str] = {"xtts", "f5tts", "indicf5"}

# Follow-up PR tracker so the warning log points users at where the
# missing backend is coming from. Keep in sync as integration PRs land.
_PENDING_BACKEND_PRS: dict[str, str] = {
    "styletts2-jp": "#75 (StyleTTS2-JP)",
    "styletts2-ko": "#75 (StyleTTS2-KO)",
    "melotts": "#74 (MeloTTS for JA/KO)",
}


def _select_tts_backend_for_language(lang: str) -> tuple[str, list[str]]:
    """Return ``(chosen_backend, full_preference_list)`` for the language.

    Walks the language's preference list and returns the first
    currently-integrated backend. The full preference list is also
    returned so the caller can log a one-time warning when the
    ideal backend isn't available.

    Unknown languages default to XTTS (broadest coverage).
    """
    lang = (lang or "").lower()
    for langs, prefs in _LANG_TTS_PREFERENCES:
        if lang in langs:
            for b in prefs:
                if b in _INTEGRATED_BACKENDS:
                    return b, prefs
            break
    return "xtts", ["xtts"]


def _warn_if_suboptimal_backend(
    lang: str, chosen: str, preferences: list[str],
) -> None:
    """If `chosen` isn't the first preference for `lang`, log one
    WARNING explaining what would be better and why it isn't available.
    """
    if not preferences:
        return
    ideal = preferences[0]
    if ideal == chosen:
        return
    pr_note = _PENDING_BACKEND_PRS.get(ideal, "(no tracking PR)")
    log.warning(
        "TTS auto-selection: target_language=%r prefers %r but that "
        "backend is not yet integrated (expected in %s). Falling back "
        "to %r — quality may be degraded. Set tts_backend=%s explicitly "
        "to silence this warning.",
        lang, ideal, pr_note, chosen, chosen,
    )


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
    backend: str | None = None,
) -> TTSResult:
    """Generate speech for `translation` using `reference_audio` as the voice.

    `translation` is the dict written by Stage 3 (see translate.py).

    `backend` selects XTTS-v2 (``"xtts"``), F5-TTS (``"f5tts"``), or
    ``"auto"`` for language-aware selection (see
    ``_select_tts_backend_for_language`` for the quality matrix).
    When ``None`` we fall back to ``settings.tts_backend`` (env default).

    `first_speech_seconds` (from the transcript) lets us prepend a matching
    amount of silence so the TTS first-frame lines up with the source's
    first spoken frame — otherwise a 1 s "speaker pauses then talks" clip
    becomes a "speaker starts talking immediately" clip.

    `source_duration_seconds` enables formant-preserving time-stretch
    (rubberband) when the post-trim TTS is still a bit longer than the
    remaining source video. Stretching is skipped when the ratio would be
    aggressive — we'd rather freeze-pad video than produce chipmunk audio.

    `transcript_segments` (from Stage 2) enables two XTTS-only quality wins:
    - smart reference selection (longest clean contiguous speech span)
    - per-segment synthesis preserving original pause structure
    When missing, we fall back to single-shot synthesis on the whole
    translated text. F5-TTS currently only runs single-shot regardless.
    """
    chosen = (backend or settings.tts_backend).lower()
    if chosen not in ("xtts", "f5tts", "indicf5", "auto"):
        raise TTSError(
            f"unknown tts backend: {chosen!r}. "
            f"Supported: xtts, f5tts, indicf5, auto"
        )

    tgt = translation.get("target_language", "").lower()

    # Auto-selection: resolve `chosen` against the language preference
    # map. If the ideal backend isn't integrated yet, fall back to the
    # best-available and log a WARNING pointing at the tracking PR.
    if chosen == "auto":
        chosen, prefs = _select_tts_backend_for_language(tgt)
        _warn_if_suboptimal_backend(tgt, chosen, prefs)
        log.info(
            "TTS auto-selected %r for target_language=%r", chosen, tgt,
        )

    # Up-front validation, ordered so the clearest error surfaces first:
    # language (per backend) → text → reference file. Matches the
    # pre-dispatch behavior tests in tests/test_tts.py rely on.
    if chosen == "xtts" and tgt not in XTTS_LANG_CODES:
        raise TTSError(
            f"XTTS-v2 does not support target language {tgt!r}. "
            f"Supported: {sorted(XTTS_LANG_CODES)}"
        )
    if chosen == "f5tts" and tgt not in F5TTS_BASE_LANGS:
        raise TTSError(
            f"F5-TTS base checkpoint ({settings.f5tts_model}) does not "
            f"officially support {tgt!r}. Base model is EN/ZH only. "
            f"Community fine-tunes exist for some other languages — set "
            f"F5TTS_MODEL to the HF repo id of one and retry, or fall "
            f"back to tts_backend=xtts for a multilingual model. "
            f"See docs/models.md for the honest language support matrix."
        )
    if chosen == "indicf5" and tgt not in INDICF5_LANG_CODES:
        raise TTSError(
            f"IndicF5 does not support target language {tgt!r}. "
            f"Supported: {sorted(INDICF5_LANG_CODES)}. "
            f"Use tts_backend=xtts for other languages, or "
            f"tts_backend=auto for automatic per-language selection."
        )

    text = (translation.get("text") or "").strip()
    if not text:
        raise TTSError("translation is empty — nothing to synthesize")

    if not reference_audio.exists():
        raise TTSError(f"reference audio missing: {reference_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if chosen == "xtts":
        result = _synthesize_xtts_full(
            translation=translation,
            text=text,
            target_language=tgt,
            reference_audio=reference_audio,
            output_path=output_path,
            transcript_segments=transcript_segments,
        )
    elif chosen == "indicf5":
        result = _synthesize_indicf5_single_shot(
            text=text,
            target_language=tgt,
            reference_audio=reference_audio,
            output_path=output_path,
            transcript_segments=transcript_segments,
        )
    else:  # f5tts (already validated above)
        result = _synthesize_f5tts_single_shot(
            text=text,
            target_language=tgt,
            reference_audio=reference_audio,
            output_path=output_path,
            transcript_segments=transcript_segments,
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(f"{chosen} produced no output")

    # --- Shared post-processing (both backends) --------------------------- #
    # Whisper-based tail trim. On short utterances XTTS tends to keep
    # generating past the real translation — hallucinated phonemes /
    # repeated syllables at speech level, which `silencedetect` can't
    # catch because it *is* speech energy. Running the expected text
    # through faster-whisper on the TTS output gives us word-level
    # timestamps we can align against; everything past the last real
    # word is tail we can safely remove. Hard-truncate at source
    # duration below is the backstop if this doesn't fire.
    try:
        _trim_tail_via_whisper(
            output_path,
            target_language=tgt,
            expected_text=text,
            # Source duration powers Tier 3's "don't cut below
            # source × 0.5" sanity floor — catches the Hindi case
            # where word-alignment matches one token and bails,
            # truncating 8s of real speech to 0.24s on a 19s source.
            source_duration_seconds=source_duration_seconds,
        )
    except Exception as e:
        log.warning("whisper tail-trim failed (%s); continuing", e)

    # Optional speed-up: fit assembled audio into remaining source window.
    if source_duration_seconds is not None:
        target = source_duration_seconds - (first_speech_seconds or 0.0)
        if target > 0:
            try:
                _maybe_time_stretch(output_path, target_duration=target)
            except Exception as e:
                log.warning("time-stretch failed (%s); keeping untouched TTS", e)

        # Hard safety net: if even after tail-trim + (optional) stretch
        # the audio is still longer than the source video window, cut it.
        # Better to crop a final syllable than to let `loop_video` fire.
        try:
            _hard_truncate(output_path, max_duration=source_duration_seconds)
        except Exception as e:
            log.warning("hard-truncate failed (%s); loop_video may fire", e)

    # Align to source: prepend silence so TTS first-frame lines up.
    if first_speech_seconds and first_speech_seconds > 0.01:
        try:
            _prepend_silence(output_path, seconds=first_speech_seconds)
        except Exception as e:
            log.warning("silence-prepend failed (%s); keeping un-aligned TTS", e)

    # Loudness normalization to -16 LUFS.
    try:
        _loudnorm(output_path)
    except Exception as e:
        log.warning("loudness normalization failed (%s); shipping unnormalized", e)

    return result


# --------------------------------------------------------------------------- #
# XTTS-v2 backend (per-segment + smart reference + loudnorm)
# --------------------------------------------------------------------------- #

_xtts = None
_xtts_lock = Lock()


def _get_xtts():
    """Lazy-load XTTS-v2. First call downloads ~1.8 GB into HF_HOME."""
    global _xtts
    if _xtts is not None:
        return _xtts
    with _xtts_lock:
        if _xtts is not None:
            return _xtts

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
        _xtts = model
        return _xtts


def _synthesize_xtts_full(
    translation: dict,
    text: str,
    target_language: str,
    reference_audio: Path,
    output_path: Path,
    transcript_segments: list[dict] | None,
) -> TTSResult:
    """XTTS-v2 synthesis with smart reference + optional per-segment path."""
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

    language = XTTS_LANG_CODES[target_language]
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

    return TTSResult(
        backend="xtts_v2",
        language=target_language,
        reference_audio=ref_label,
        output_path=output_path.name,
        per_segment=can_do_per_segment,
        segments_synthesized=segments_synthesized,
    )


# --------------------------------------------------------------------------- #
# F5-TTS backend (single-shot; per-segment/smart-ref is a follow-up)
# --------------------------------------------------------------------------- #

_f5tts = None
_f5tts_lock = Lock()

_indicf5 = None
_indicf5_lock = Lock()


def _get_f5tts():
    """Lazy-load F5-TTS. First call downloads the chosen checkpoint."""
    global _f5tts
    if _f5tts is not None:
        return _f5tts
    with _f5tts_lock:
        if _f5tts is not None:
            return _f5tts

        # Keep F5-TTS weights under MODEL_CACHE_DIR like everything else.
        os.environ.setdefault(
            "HF_HOME",
            str(settings.model_cache_dir / "huggingface"),
        )

        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise TTSError(
                "f5-tts package is not installed. Rebuild the backend image "
                "(it ships f5-tts via the Dockerfile) and retry. "
                "Original error: " + str(e),
            ) from e

        # device="cpu" forces the CPU path; F5-TTS auto-picks CUDA otherwise.
        model = F5TTS(model=settings.f5tts_model, device="cpu")
        _f5tts = model
        return _f5tts


def _f5tts_reference_text(
    reference_audio: Path, transcript_segments: list[dict] | None,
) -> str:
    """Build the F5-TTS reference text.

    F5-TTS conditions generation on both the reference audio *and* the
    text that was actually spoken in that reference. Passing the Whisper
    transcript here gives dramatically cleaner timbre than passing an
    empty string and letting F5-TTS whisper-infer it at inference time.

    Order of preference:
      1. transcript_segments (inline, no file I/O)
      2. transcript.json in the same dir as the reference audio
      3. "" (F5-TTS falls back to its own whisper-based ref inference)
    """
    if transcript_segments:
        chunks = [
            (s.get("text") or "").strip() for s in transcript_segments
        ]
        joined = " ".join(c for c in chunks if c).strip()
        if joined:
            return joined

    candidate = reference_audio.parent / "transcript.json"
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text())
            return (data.get("text") or "").strip()
        except (OSError, json.JSONDecodeError):
            pass
    return ""


def _get_indicf5():
    """Lazy-load IndicF5. First call downloads ~1.5 GB.

    IndicF5 is distributed through HuggingFace (`ai4bharat/IndicF5`)
    with `trust_remote_code=True` — the model's custom inference code
    lives in the HF repo rather than a pip package. That's the path
    their model card documents and what the community uses.

    We load on CPU; IndicF5 inherits F5-TTS's CPU support. First load
    takes ~30-60 s plus the download.
    """
    global _indicf5
    if _indicf5 is not None:
        return _indicf5
    with _indicf5_lock:
        if _indicf5 is not None:
            return _indicf5

        # Cache under MODEL_CACHE_DIR like every other HF model.
        os.environ.setdefault(
            "HF_HOME",
            str(settings.model_cache_dir / "huggingface"),
        )

        # IndicF5 is a gated repo on HuggingFace — accessing it requires
        # an authenticated user that's been granted access at
        # https://huggingface.co/ai4bharat/IndicF5. Different versions
        # of huggingface_hub / transformers look at different env-var
        # names (HF_TOKEN vs HUGGING_FACE_HUB_TOKEN) and `login()` calls
        # vs file-cached tokens, so we belt-and-suspenders all three:
        #   1. Pick up a token from either env-var spelling
        #   2. Call huggingface_hub.login() to set the in-memory token
        #   3. Write it to $HF_HOME/token so subsequent calls pick it up
        #      from the file cache too
        # When no token is set, skip silently — anonymous access still
        # works for non-gated repos and we'll get a clear 401 from the
        # actual gated download with our own from_pretrained handler.
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if token:
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=token, add_to_git_credential=False)
            except Exception as _login_err:
                log.warning(
                    "huggingface_hub.login() failed (%s); continuing — "
                    "from_pretrained will fall back to env-var auth",
                    _login_err,
                )

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise TTSError(
                "transformers is required for IndicF5 but isn't importable. "
                "Rebuild the backend image. Original error: " + str(e),
            ) from e

        repo = os.environ.get("INDICF5_MODEL", "ai4bharat/IndicF5")
        log.info(
            "loading IndicF5 from %s (first call: ~1.5 GB download + ~30-60s init)",
            repo,
        )
        try:
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        except Exception as e:
            err_str = str(e)
            # Detect gated-repo / auth failures and produce a more
            # actionable error than the generic load-failed message.
            # The HF lib throws several flavors here (GatedRepoError,
            # OSError wrapping 401, etc.); match on the wording.
            is_gated = (
                "gated repo" in err_str.lower()
                or "401" in err_str
                or "access" in err_str.lower() and "restricted" in err_str.lower()
            )
            if is_gated:
                token_set = bool(
                    os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                )
                token_hint = (
                    "an HF_TOKEN IS set in this container; the token "
                    "may not have access yet (check approval status at "
                    "https://huggingface.co/ai4bharat/IndicF5)"
                    if token_set
                    else "no HF_TOKEN found in this container's env. "
                         "Set HF_TOKEN=hf_... in .env (and "
                         "`docker compose up -d --force-recreate backend` "
                         "to pick it up). The compose file forwards "
                         "HF_TOKEN automatically."
                )
                raise TTSError(
                    f"IndicF5 is a gated HuggingFace repo and the "
                    f"current container can't authenticate to it.\n"
                    f"  Status: {token_hint}\n"
                    f"  Underlying error: {e}\n"
                    f"  To grant access: visit "
                    f"https://huggingface.co/{repo} and click "
                    f"'Request access'. Approval is usually instant."
                ) from e
            raise TTSError(
                f"IndicF5 load failed from {repo!r}: {e}. "
                f"Check that (a) the model repo is reachable, "
                f"(b) transformers + accelerate + einops are up to date, "
                f"(c) disk at $HF_HOME has enough free space (~2 GB). "
                f"Override with INDICF5_MODEL=<alternate-repo-id> if needed."
            ) from e
        # Explicit CPU placement + eval mode. Belt-and-suspenders:
        # our container already has CUDA_VISIBLE_DEVICES="" but if
        # someone runs this on a CUDA box we don't want surprise GPU
        # work without opting in.
        try:
            model = model.to("cpu")
        except Exception:
            pass  # model may not be a nn.Module; ignore if .to() fails
        try:
            model.eval()
        except Exception:
            pass
        _indicf5 = model
        return _indicf5


def _synthesize_indicf5_single_shot(
    text: str,
    target_language: str,
    reference_audio: Path,
    output_path: Path,
    transcript_segments: list[dict] | None,
) -> TTSResult:
    """Single-shot IndicF5 synthesis.

    IndicF5's inference signature (per the HF model card):

        audio = model(text, ref_audio_path=..., ref_text=...)

    Returns a numpy (or torch) array at 24000 Hz. We normalize to
    float32 + peak-limit and write as a WAV, then run the same
    silence trim / loudness stack as the other backends.

    Reference audio: we pass the source-language (e.g. English)
    reference from the pipeline. IndicF5 adapts the voice color to
    the target phoneme set; it does not require a native-language
    reference. If quality suffers on specific speaker-target pairs,
    users can override by supplying a native-language reference clip.
    """
    model = _get_indicf5()
    ref_text = _f5tts_reference_text(reference_audio, transcript_segments)
    if not ref_text:
        # IndicF5 needs a reference transcript; without it the model
        # has nothing to condition voice color against. Don't try to
        # guess — fail clearly so the user rewinds to transcribe.
        raise TTSError(
            "IndicF5 requires a reference transcript but none was found. "
            "The pipeline normally provides the Whisper transcript from "
            "Stage 2. Check that transcribe.py ran successfully and that "
            "transcript_segments is populated on the job."
        )

    # Defensive: we don't know the exact IndicF5 API version. Catch
    # TypeError (arg-name changes) and report clearly so the user can
    # pin to a known-good commit.
    try:
        import torch as _torch
        with _torch.no_grad():
            audio = model(
                text,
                ref_audio_path=str(reference_audio),
                ref_text=ref_text,
            )
    except TypeError as e:
        raise TTSError(
            f"IndicF5 call signature mismatch ({e}). The model's "
            f"inference API may have changed. Pin a known-good commit "
            f"via INDICF5_MODEL=ai4bharat/IndicF5@<sha> or check the "
            f"model card for the current signature."
        ) from e
    except Exception as e:
        raise TTSError(f"IndicF5 inference failed: {e}") from e

    # IndicF5 sample rate per the F5-TTS lineage.
    _INDICF5_SAMPLE_RATE = 24000

    # Accept a few reasonable return shapes:
    #   - numpy array (1D waveform)
    #   - torch tensor
    #   - tuple (audio, sample_rate)
    #   - dict {"audio": ..., "sampling_rate": ...}
    import numpy as _np
    sr = _INDICF5_SAMPLE_RATE
    if isinstance(audio, dict):
        sr = int(audio.get("sampling_rate", sr))
        audio = audio.get("audio", audio.get("waveform"))
    if isinstance(audio, tuple):
        audio, sr_candidate = audio[0], audio[1] if len(audio) > 1 else sr
        try:
            sr = int(sr_candidate)
        except Exception:
            pass
    if hasattr(audio, "detach"):   # torch tensor
        audio = audio.detach().cpu().numpy()
    audio = _np.asarray(audio, dtype=_np.float32)
    # Drop leading batch / channel dim if 2-D with a unit axis.
    while audio.ndim > 1 and 1 in audio.shape:
        audio = audio.squeeze(
            next(i for i, d in enumerate(audio.shape) if d == 1),
        )
    if audio.ndim != 1:
        raise TTSError(
            f"IndicF5 returned an unexpected audio shape {audio.shape!r}; "
            f"expected a 1-D waveform (or reducible to one)."
        )

    # Peak-limit to 0.95 if model over-ranges (rare but cheap to guard).
    peak = float(_np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.95

    try:
        import soundfile as _sf
    except ImportError as e:
        raise TTSError(
            "soundfile is required to write IndicF5 output. "
            "This is a pipeline dep — rebuild the backend image. "
            f"Original: {e}",
        ) from e
    _sf.write(str(output_path), audio, sr)

    # Same post-processing as F5-TTS — IndicF5 also emits short leading
    # click / trailing silence.
    try:
        _trim_to_speech(output_path)
    except Exception as e:
        log.warning(
            "silence trim failed (%s); keeping untrimmed IndicF5 output", e,
        )

    return TTSResult(
        backend="indicf5",
        language=target_language,
        reference_audio=reference_audio.name,
        output_path=output_path.name,
        per_segment=False,
        segments_synthesized=1,
    )


def _synthesize_f5tts_single_shot(
    text: str,
    target_language: str,
    reference_audio: Path,
    output_path: Path,
    transcript_segments: list[dict] | None,
) -> TTSResult:
    """Single-shot F5-TTS synthesis. Per-segment path is a follow-up."""
    f5tts = _get_f5tts()
    ref_text = _f5tts_reference_text(reference_audio, transcript_segments)

    # F5-TTS's Python API writes directly to `file_wave`.
    f5tts.infer(
        ref_file=str(reference_audio),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output_path),
    )

    # Trim the same way XTTS's single-shot path does — F5-TTS also tends
    # to emit a short leading click / trailing silence.
    try:
        _trim_to_speech(output_path)
    except Exception as e:
        log.warning("silence trim failed (%s); keeping untrimmed F5 output", e)

    return TTSResult(
        backend="f5tts",
        language=target_language,
        reference_audio=reference_audio.name,
        output_path=output_path.name,
        per_segment=False,
        segments_synthesized=1,
    )


# --------------------------------------------------------------------------- #
# Reference-audio selection (XTTS only)
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
# Per-segment synthesis (XTTS only)
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
    tts = _get_xtts()
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

# Whisper-based tail trim. XTTS on short inputs tends to over-generate —
# finishing the real translation and then continuing with hallucinated
# phonemes / repeated syllables / mumbling at roughly speech level.
# `silencedetect`-based trimming can't spot this because the tail IS
# speech-energy; only a transcription can. We reuse the faster-whisper
# singleton from the transcribe stage, align its word output against
# the expected translation text, and cut after the last real word.
_WHISPER_TRIM_HEADROOM = 0.08   # keep a little of the word's decay
_WHISPER_TRIM_MIN_GAIN = 0.10   # don't bother if we'd save less than this

# Confidence floor for the word-by-word alignment in
# `_find_last_real_word_end`. Below this fraction-of-expected-tokens
# matched, we treat the alignment as untrustworthy (almost certainly
# the case for non-Latin-script targets like Hindi where Whisper's
# tokenization disagrees with NLLB's translation tokenization). When
# the floor isn't met we fall back to Whisper's VAD-based last-segment
# end, then to no-trim, rather than cutting off real speech mid-word.
_WHISPER_TRIM_MIN_MATCH_RATIO = 0.5

# When the source video duration is known, refuse trims that cut the
# audio below this fraction of source duration. The translation is a
# *dub* — output should be roughly the same length as the source clip.
# A trim that drops audio to 5% of source is almost always an
# alignment failure, not a real "we discovered the speech ended early"
# signal. Caught the Hindi case where the trim truncated 8 s of real
# Hindi to 0.24 s on a 19 s source clip.
_WHISPER_TRIM_MIN_RATIO_OF_SOURCE = 0.5


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
    # Keep the real extension (.wav) so ffmpeg can infer the output format.
    # `.with_suffix(suffix + ".trim")` produced "foo.wav.trim", which ffmpeg
    # refuses because .trim isn't a known format.
    tmp = dst.parent / f"{dst.stem}.trim{dst.suffix}"
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


def _normalize_word(s: str) -> str:
    """Lowercase and strip to alphanumerics for cross-language comparison."""
    return "".join(c for c in s.lower() if c.isalnum())


def _find_last_real_word_end(
    expected_text: str, transcribed_words: list[tuple[float, str]],
) -> tuple[float | None, int, int]:
    """Walk Whisper's transcribed word stream against the expected
    translation text. Return the `end` timestamp of the last word we
    matched, plus the (matched_count, expected_count) so the caller
    can compute alignment confidence and decide whether to trust the
    result.

    The XTTS hallucination problem looks like this in Whisper output:

        expected: "Hola, me llamo Carlos"
        whisper:  [Hola, me, llamo, Carlos, me, lla, a, la, ...]
                                            ^^^^^^^^^^^^^^^ tail

    Strategy: sequential best-match. Walk Whisper's words; for each one
    that matches the next expected token (or is close — XTTS sometimes
    splits a word across two Whisper tokens), advance the expected
    pointer. Stop advancing once we've consumed the whole expected text
    — everything after is hallucination. After 3 consecutive non-matches
    (probably in the hallucinated tail), break.

    Returns (last_matched_end, matched_count, expected_count). The
    timestamp is None when nothing matched at all. The counts are
    always returned; the caller divides them for confidence.

    Why returning confidence matters: on non-Latin-script targets
    (Hindi, Arabic, Chinese) Whisper's tokenization frequently
    disagrees with NLLB's, so we match the first 1–2 words and then
    bail — leaving last_matched_end pointing at 0.16 s on a 19 s clip.
    The caller uses (matched / expected) to decide when to trust the
    timestamp vs fall back to VAD or skip the trim entirely.
    """
    expected_tokens = [_normalize_word(w) for w in expected_text.split()]
    expected_tokens = [t for t in expected_tokens if t]
    if not expected_tokens or not transcribed_words:
        return None, 0, len(expected_tokens)

    ei = 0                      # index into expected
    matched_count = 0
    consecutive_misses = 0
    last_matched_end: float | None = None
    for end_ts, raw in transcribed_words:
        if ei >= len(expected_tokens):
            break
        norm = _normalize_word(raw)
        if not norm:
            continue
        expected = expected_tokens[ei]
        # Match if normalized forms are equal, or one is a prefix of the
        # other with a shared 3+ char stem (handles XTTS phoneme splits
        # and Whisper's occasional partial word tokens).
        is_match = (
            norm == expected
            or (len(norm) >= 3 and expected.startswith(norm))
            or (len(expected) >= 3 and norm.startswith(expected))
        )
        if is_match:
            last_matched_end = end_ts
            matched_count += 1
            ei += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            # If Whisper emits 3+ unrelated tokens in a row, we're
            # probably in the hallucinated tail. Stop here.
            if consecutive_misses >= 3 and last_matched_end is not None:
                break

    return last_matched_end, matched_count, len(expected_tokens)


def _trim_tail_via_whisper(
    audio_path: Path,
    target_language: str,
    expected_text: str,
    source_duration_seconds: float | None = None,
) -> None:
    """Trim audio past the last real translated word — tiered approach.

    XTTS on short utterances tends to over-generate, producing
    hallucinated phonemes / repeated syllables / mumbling past the
    actual translation. `silencedetect` can't catch this because the
    tail IS speech energy. Whisper transcription can — for languages
    where its tokenization aligns with our translator's.

    Three tiers, falling back when each fails. None of them ever
    produces a trim_to value we can't justify:

    Tier 1 (preferred) — sequential word alignment of Whisper's output
        against `expected_text`. Confidence = matched / expected. If
        confidence ≥ `_WHISPER_TRIM_MIN_MATCH_RATIO` (default 0.5),
        accept the timestamp of the last matched word. Strong on
        Spanish/English/etc.

    Tier 2 (fallback) — Whisper's last VAD-detected speech-segment
        end. Doesn't depend on word identity, so works for any
        language. Misses the case where hallucinated speech is
        contiguous with real speech (no VAD-level silence between
        them), but combined with Tier 3 still produces safe behavior.

    Tier 3 (sanity floor) — if `source_duration_seconds` is known,
        refuse trims that cut below `source × _WHISPER_TRIM_MIN_RATIO_OF_SOURCE`
        (default 0.5). The pipeline is dubbing — output should be
        roughly the same duration as the source clip. A trim to 5%
        is almost always an alignment failure (the Hindi case that
        motivated this PR).

    If all tiers fail/refuse, no trim. `_hard_truncate(source_duration)`
    downstream is the structural backstop against `loop_video`.
    """
    if not expected_text.strip():
        return

    try:
        from .transcribe import _get_model
        model = _get_model()
    except Exception as e:
        log.warning("whisper-trim: model load failed (%s); skipping", e)
        return

    try:
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=target_language or None,
            beam_size=1,   # speed > accuracy for a tail-find job
            word_timestamps=True,
            vad_filter=True,
            # 300 ms splits real speech from hallucinated tail more
            # reliably than the 500 ms we use on source transcription.
            vad_parameters={"min_silence_duration_ms": 300},
            # Seed Whisper's decoder with the expected text — it biases
            # the model toward transcribing what we actually generated
            # and makes hallucinated tails easier to distinguish.
            initial_prompt=expected_text,
        )
        # Materialize segments once; we need both word stream (Tier 1)
        # and segment-end (Tier 2) from the same iteration.
        segments = list(segments_iter)
        transcribed: list[tuple[float, str]] = []
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                if w.end is None:
                    continue
                transcribed.append((float(w.end), str(w.word)))
    except Exception as e:
        log.warning("whisper-trim: transcribe failed (%s); skipping", e)
        return

    if not transcribed:
        log.info("whisper-trim: no words detected; skipping")
        return

    # ---- Tier 1: word-by-word alignment with confidence floor -------- #
    word_end, matched, expected_n = _find_last_real_word_end(
        expected_text, transcribed,
    )
    confidence = matched / max(1, expected_n)
    tier1_end: float | None = None
    if word_end is not None and confidence >= _WHISPER_TRIM_MIN_MATCH_RATIO:
        tier1_end = word_end

    # ---- Tier 2: VAD-based last-segment end (any language) ----------- #
    tier2_end: float | None = None
    valid_seg_ends = [
        float(s.end) for s in segments
        if getattr(s, "end", None) is not None
    ]
    if valid_seg_ends:
        tier2_end = max(valid_seg_ends)

    # Pick the latest available endpoint (most generous; a smaller
    # candidate would risk cutting real speech).
    candidates = [t for t in (tier1_end, tier2_end) if t is not None]
    if not candidates:
        log.info(
            "whisper-trim: no usable endpoint (alignment confidence=%.2f, "
            "expected=%d, no VAD segments); skipping",
            confidence, expected_n,
        )
        return
    chosen_end = max(candidates)
    chosen_tier = "word-align" if chosen_end == tier1_end else "vad-segment"

    current = _probe_duration(audio_path)
    trim_to = min(current, chosen_end + _WHISPER_TRIM_HEADROOM)

    # ---- Tier 3: source-duration sanity floor ------------------------ #
    if source_duration_seconds is not None and source_duration_seconds > 0:
        floor = source_duration_seconds * _WHISPER_TRIM_MIN_RATIO_OF_SOURCE
        if trim_to < floor:
            log.warning(
                "whisper-trim %s: would cut %.2fs -> %.2fs but source is "
                "%.2fs (floor %.2fs at ratio %.2f). Likely alignment failure "
                "(tier=%s, word-align confidence=%.2f); skipping. "
                "Hard-truncate will still cap at source duration.",
                audio_path.name, current, trim_to,
                source_duration_seconds, floor,
                _WHISPER_TRIM_MIN_RATIO_OF_SOURCE,
                chosen_tier, confidence,
            )
            return

    if current - trim_to < _WHISPER_TRIM_MIN_GAIN:
        log.info(
            "whisper-trim %s: chosen end %.2fs already close to current "
            "%.2fs (tier=%s, confidence=%.2f); skipping",
            audio_path.name, chosen_end, current, chosen_tier, confidence,
        )
        return

    _ffmpeg_atrim(audio_path, audio_path, 0.0, trim_to)
    log.info(
        "whisper-trim %s: %.2fs -> %.2fs (tier=%s, end=%.2fs, "
        "confidence=%.2f, cut %.2fs of tail)",
        audio_path.name, current, trim_to, chosen_tier,
        chosen_end, confidence, current - trim_to,
    )


def _hard_truncate(audio_path: Path, max_duration: float) -> None:
    """Last-resort truncate `audio_path` to `max_duration` seconds.

    Used as a safety net when `_trim_to_speech` + `_trim_tail_via_whisper`
    + `_maybe_time_stretch` still leave the audio longer than the source
    video. Without this, LatentSync's `loop_video` reverses frames to
    cover the gap and introduces a visible jitter spike.

    A hard cut at max_duration risks cropping the very last syllable —
    worse than ideal but strictly better than a reversed-frame boom at
    the end of every generated clip.
    """
    current = _probe_duration(audio_path)
    if current <= max_duration + 0.02:
        return
    log.warning(
        "hard-truncating %s: %.2fs -> %.2fs (fits source; prevents loop_video)",
        audio_path.name, current, max_duration,
    )
    _ffmpeg_atrim(audio_path, audio_path, 0.0, max_duration)


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

    tmp = audio_path.parent / f"{audio_path.stem}.stretch{audio_path.suffix}"
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

    tmp = audio_path.parent / f"{audio_path.stem}.pad{audio_path.suffix}"
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
    tmp = audio_path.parent / f"{audio_path.stem}.lnorm{audio_path.suffix}"
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
