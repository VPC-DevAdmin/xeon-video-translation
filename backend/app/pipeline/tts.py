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
3. Lines that must be faster are re-spoken at XTTS's own speed; what is
   left is sped up with ffmpeg atempo.
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
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import settings

log = logging.getLogger(__name__)


# BCP-47 -> XTTS-v2 language codes. XTTS supports these; our NLLB language
# picker is a superset, so some translations may not be synthesizable.
XTTS_LANG_CODES: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "hu": "hu",
    "ko": "ko",
    "ja": "ja",
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
    "hi": "hi",  # Hindi
    "bn": "bn",  # Bengali
    "ta": "ta",  # Tamil
    "te": "te",  # Telugu
    "mr": "mr",  # Marathi
    "gu": "gu",  # Gujarati
    "kn": "kn",  # Kannada
    "ml": "ml",  # Malayalam
    "pa": "pa",  # Punjabi
    "or": "or",  # Odia / Oriya
    "as": "as",  # Assamese
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
    lang: str,
    chosen: str,
    preferences: list[str],
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
        lang,
        ideal,
        pr_note,
        chosen,
        chosen,
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
    options: dict | None = None,
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

    `source_duration_seconds` enables a time-stretch (atempo) when the post-trim TTS is still a bit longer than the
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
        raise TTSError(f"unknown tts backend: {chosen!r}. Supported: xtts, f5tts, indicf5, auto")

    tgt = translation.get("target_language", "").lower()

    # Auto-selection: resolve `chosen` against the language preference
    # map. If the ideal backend isn't integrated yet, fall back to the
    # best-available and log a WARNING pointing at the tracking PR.
    if chosen == "auto":
        chosen, prefs = _select_tts_backend_for_language(tgt)
        _warn_if_suboptimal_backend(tgt, chosen, prefs)
        log.info(
            "TTS auto-selected %r for target_language=%r",
            chosen,
            tgt,
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

    segments = translation.get("segments") or []
    if segments and transcript_segments:
        count = _synthesize_per_segment(
            segments,
            segments,
            reference_audio,
            XTTS_LANG_CODES.get(tgt, tgt),
            output_path,
            backend=chosen,
            target_language=tgt,
            reference_segments=transcript_segments,
            options=options,
            source_duration_seconds=source_duration_seconds,
        )
        result = TTSResult(
            backend=chosen,
            language=tgt,
            reference_audio=reference_audio.name,
            output_path=output_path.name,
            per_segment=True,
            segments_synthesized=count,
        )
        first_speech_seconds = float(segments[0]["start"])
    elif chosen == "xtts":
        result = _synthesize_xtts_full(
            translation,
            text,
            tgt,
            reference_audio,
            output_path,
            transcript_segments,
        )
    else:
        generate = (
            _synthesize_indicf5_single_shot
            if chosen == "indicf5"
            else _synthesize_f5tts_single_shot
        )
        result = generate(text, tgt, reference_audio, output_path, transcript_segments)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(f"{chosen} produced no output")

    if chosen == "xtts" and not result.per_segment:
        verified = _trim_tail_via_whisper(output_path, tgt, text)
        if verified is False:
            raise TTSError("generated speech does not match the complete translation")
        if verified is not True:
            _trim_to_speech(output_path)

    # Align to source: prepend silence so TTS first-frame lines up.
    if first_speech_seconds and first_speech_seconds > 0.01:
        try:
            _prepend_silence(output_path, seconds=first_speech_seconds)
        except Exception as e:
            raise TTSError(f"speech alignment failed: {e}") from e

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
        ).to(settings.resolved_device)
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

        # Pass the device explicitly; F5-TTS would otherwise auto-pick CUDA
        # even on the CPU build.
        model = F5TTS(model=settings.f5tts_model, device=settings.resolved_device)
        _f5tts = model
        return _f5tts


def _f5tts_reference_text(
    reference_audio: Path,
    transcript_segments: list[dict] | None,
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
        chunks = [(s.get("text") or "").strip() for s in transcript_segments]
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
    """Lazy-load IndicF5 by calling F5-TTS APIs directly with IndicF5
    weights + vocab. Returns a dict of components, not a single model
    object — the synthesis function in `_synthesize_indicf5_single_shot`
    composes them per-call.

    Why this path, not `AutoModel.from_pretrained(..., trust_remote_code=True)`:

    The HuggingFace `ai4bharat/IndicF5` repo ships a `model.py` whose
    `INF5Model.__init__` is upstream-broken in two ways:

    1. The safetensors loading is commented out (lines 46-49):
            # safetensors_path = hf_hub_download(...)
            # state_dict = load_file(safetensors_path, ...)
            # self.ema_model.load_state_dict(state_dict, strict=False)

    2. The `load_model()` call is missing the required `ckpt_path`
       positional argument (line 52):
            self.ema_model = load_model(
                DiT,
                dict(dim=1024, depth=22, ...),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device,
                # ckpt_path missing — F5-TTS sig is (cls, cfg, ckpt_path, ...)
            )

    Result: `AutoModel.from_pretrained` raises
        load_model() missing 1 required positional argument: 'ckpt_path'

    Even if we passed the missing arg, the weights are never loaded
    because the safetensors block is commented out. The trust_remote_code
    path is dead.

    Our integration replicates what `INF5Model.__init__` *should* do —
    download weights + vocab via huggingface_hub, call F5-TTS's
    `load_model` with the IndicF5 DiT config + `ckpt_path` pointing at
    the safetensors, load the vocoder. About 25 lines vs. vendoring
    the whole 165-line file. No `trust_remote_code`. Same downloaded
    artifact (`model.safetensors` + `checkpoints/vocab.txt`) — only the
    invocation path differs.
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

        # IndicF5 is gated. Authenticate before any hub call so the
        # download doesn't 401. See PR #76 for the multi-spelling
        # token-discovery rationale.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            try:
                from huggingface_hub import login as _hf_login

                _hf_login(token=token, add_to_git_credential=False)
            except Exception as _login_err:
                log.warning(
                    "huggingface_hub.login() failed (%s); continuing — "
                    "hub calls will fall back to env-var auth",
                    _login_err,
                )

        # Pin a known-good revision so upstream model.py / config tweaks
        # can't shift behavior under us. ba85abe... is the revision the
        # codebase was integrated against (PR #74). Override via env to
        # test newer revisions.
        repo = os.environ.get("INDICF5_MODEL", "ai4bharat/IndicF5")
        revision = os.environ.get(
            "INDICF5_REVISION",
            "ba85abedf18dc479a447eaa0eccbd76ab78a47d5",
        )
        log.info(
            "loading IndicF5 from %s@%s (first call: ~1.5 GB download)",
            repo,
            revision[:8],
        )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise TTSError(
                "huggingface_hub is required for IndicF5 but isn't "
                "importable. Rebuild the backend image. Original error: " + str(e),
            ) from e

        try:
            ckpt_path = hf_hub_download(
                repo_id=repo,
                filename="model.safetensors",
                revision=revision,
            )
            vocab_path = hf_hub_download(
                repo_id=repo,
                filename="checkpoints/vocab.txt",
                revision=revision,
            )
        except Exception as e:
            err_str = str(e)
            is_gated = (
                "gated repo" in err_str.lower()
                or "401" in err_str
                or ("access" in err_str.lower() and "restricted" in err_str.lower())
            )
            if is_gated:
                token_set = bool(token)
                token_hint = (
                    "an HF_TOKEN IS set in this container; the token "
                    "may not have access yet (check approval status at "
                    "https://huggingface.co/ai4bharat/IndicF5)"
                    if token_set
                    else "no HF_TOKEN found in this container's env. "
                    "Set HF_TOKEN=hf_... in .env (and "
                    "`docker compose up -d --force-recreate backend` "
                    "to pick it up)."
                )
                raise TTSError(
                    f"IndicF5 weights are gated on HuggingFace and the "
                    f"current container can't authenticate.\n"
                    f"  Status: {token_hint}\n"
                    f"  Underlying error: {e}\n"
                    f"  To grant access: visit "
                    f"https://huggingface.co/{repo} and click "
                    f"'Request access'. Approval is usually instant."
                ) from e
            raise TTSError(
                f"IndicF5 weight download failed from {repo}@{revision[:8]}: "
                f"{e}. Check that (a) the model repo is reachable, "
                f"(b) disk at $HF_HOME has enough free space (~2 GB), "
                f"(c) the revision exists. Override with INDICF5_MODEL or "
                f"INDICF5_REVISION if needed."
            ) from e

        # IndicF5's model.safetensors was saved from a torch.compile()-
        # wrapped INF5Model containing BOTH ema_model and vocoder.
        # That gives us state_dict keys like:
        #   ema_model._orig_mod.transformer.blocks.0.attn.to_q.weight
        #   vocoder._orig_mod.backbone.convnext.0.dwconv.weight
        # F5-TTS's load_model wants a checkpoint of just the ema_model
        # weights without the _orig_mod wrapper. Repack the safetensors
        # into the shape F5-TTS expects, write a clean copy alongside
        # the original, and feed the clean path to load_model.
        #
        # Cached after first run — the (cleaned) file goes next to the
        # original in HF_HOME and is reused on subsequent calls.
        try:
            from safetensors.torch import load_file, save_file
        except ImportError as e:
            raise TTSError(
                "safetensors is required for IndicF5 weight repack: " + str(e),
            ) from e

        clean_ckpt_path = ckpt_path.replace(
            "model.safetensors",
            "model.f5tts_ready.safetensors",
        )
        if not os.path.exists(clean_ckpt_path):
            log.info("repacking IndicF5 safetensors → F5-TTS-compatible layout")
            raw = load_file(ckpt_path)
            ema_state = {}
            for k, v in raw.items():
                # Keep only ema_model weights, strip both prefixes.
                # vocoder weights are loaded separately via load_vocoder.
                if k.startswith("ema_model._orig_mod."):
                    new_k = k[len("ema_model._orig_mod.") :]
                    ema_state[new_k] = v
                elif k.startswith("ema_model."):
                    # Defensive: in case a future revision drops torch.compile
                    new_k = k[len("ema_model.") :]
                    ema_state[new_k] = v
            if not ema_state:
                raise TTSError(
                    f"IndicF5 safetensors at {ckpt_path} contained no "
                    f"'ema_model.*' keys to repack. Layout may have "
                    f"changed upstream — sample of actual keys: "
                    f"{sorted(raw.keys())[:5]!r}"
                )
            save_file(ema_state, clean_ckpt_path)
            log.info(
                "wrote %d ema_model weights to %s",
                len(ema_state),
                clean_ckpt_path,
            )

        # Now build the model + vocoder via F5-TTS's APIs directly —
        # same calls IndicF5's broken model.py *should* have made.
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
        except ImportError as e:
            raise TTSError(
                "f5-tts package is required for IndicF5 but isn't "
                "importable. Rebuild the backend image (Dockerfile "
                "installs f5-tts with --no-deps). Original error: " + str(e),
            ) from e

        try:
            # IndicF5's DiT architecture parameters from its model.py.
            # Use the repacked safetensors (clean_ckpt_path) so F5-TTS's
            # state-dict loader sees the keys it expects.
            ema_model = load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                ckpt_path=clean_ckpt_path,
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=settings.resolved_device,
            )
            vocoder = load_vocoder(
                vocoder_name="vocos",
                is_local=False,
                device=settings.resolved_device,
            )
        except Exception as e:
            raise TTSError(
                f"IndicF5 model construction failed: {e}. This is a "
                f"f5-tts internals issue. Check that f5-tts version "
                f"in the backend image is compatible with the IndicF5 "
                f"DiT config (dim=1024 depth=22 heads=16)."
            ) from e

        # Wrap as a dict so the synthesis function has typed access.
        # Eval mode + no_grad are applied per-call in the synth function.
        _indicf5 = {
            "ema_model": ema_model,
            "vocoder": vocoder,
            "sample_rate": 24000,
        }
        return _indicf5


def _synthesize_indicf5_single_shot(
    text: str,
    target_language: str,
    reference_audio: Path,
    output_path: Path,
    transcript_segments: list[dict] | None,
) -> TTSResult:
    """Single-shot IndicF5 synthesis.

    Calls F5-TTS's `infer_process` directly with the IndicF5
    ema_model + vocoder loaded by `_get_indicf5()`. This replicates
    what `INF5Model.forward` *should* do — IndicF5's HF model.py
    forward path is fine, we just bypass the broken __init__ that
    couldn't load weights. Same downstream artifact (24 kHz waveform).

    Reference audio: we pass the source-language (e.g. English)
    reference from the pipeline. IndicF5 adapts the voice color to
    the target phoneme set; it does not require a native-language
    reference. If quality suffers on specific speaker-target pairs,
    users can override by supplying a native-language reference clip.
    """
    components = _get_indicf5()
    picked = _select_f5_reference(reference_audio, transcript_segments, output_path.parent)
    if picked is not None:
        reference_audio, ref_text = picked
    else:
        ref_text = _f5tts_reference_text(reference_audio, transcript_segments)
    if not ref_text:
        raise TTSError(
            "IndicF5 requires a reference transcript but none was found. "
            "The pipeline normally provides the Whisper transcript from "
            "Stage 2. Check that transcribe.py ran successfully and that "
            "transcript_segments is populated on the job."
        )

    try:
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
        )
    except ImportError as e:
        raise TTSError(
            "f5-tts inference helpers not importable: " + str(e),
        ) from e

    try:
        import torch as _torch

        # Preprocess (normalize loudness, trim silence, etc.). Returns
        # the cleaned reference + transcript.
        ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(
            str(reference_audio),
            ref_text,
        )
        with _torch.no_grad():
            audio, sr_candidate, _ = infer_process(
                ref_audio_clean,
                ref_text_clean,
                text,
                components["ema_model"],
                components["vocoder"],
                mel_spec_type="vocos",
                speed=1.0,
                device=settings.resolved_device,
            )
    except TypeError as e:
        raise TTSError(
            f"IndicF5 inference signature mismatch ({e}). The f5-tts "
            f"infer_process API may have changed; check f5-tts version."
        ) from e
    except Exception as e:
        raise TTSError(f"IndicF5 inference failed: {e}") from e

    # f5-tts.infer_process returns (audio, sample_rate, spectrogram).
    # We've already destructured the tuple at the call site; sr_candidate
    # is the sample rate it reports. Fall back to IndicF5's documented
    # 24 kHz if the value looks bogus.
    import numpy as _np

    try:
        sr = int(sr_candidate) if sr_candidate else 24000
    except Exception:
        sr = 24000
    if hasattr(audio, "detach"):  # torch tensor
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
            "silence trim failed (%s); keeping untrimmed IndicF5 output",
            e,
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
    picked = _select_f5_reference(reference_audio, transcript_segments, output_path.parent)
    if picked is not None:
        ref_path, ref_text = picked
    else:
        ref_path = reference_audio
        ref_text = _f5tts_reference_text(reference_audio, transcript_segments)

    # F5-TTS's Python API writes directly to `file_wave`.
    f5tts.infer(
        ref_file=str(ref_path),
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

_REF_GAP_SECONDS = 0.30  # word-to-word gap threshold for "contiguous"
_REF_MIN_SPAN_SECONDS = 3.0  # XTTS's documented minimum reference length
_REF_MAX_SPAN_SECONDS = 12.0  # longer doesn't help; cap to keep loads fast


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
        for w in seg.get("words") or []:
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
            "longest clean speech span is only %.2fs — using whole source as XTTS reference",
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
        span_len,
        span_start,
        span_end,
    )
    return trimmed, f"trimmed {span_len:.2f}s"


_F5_REF_MAX_SPAN_SECONDS = 10.0  # F5 clips refs at ~12 s internally; stay under


def _word_text(w: dict) -> str:
    return str(w.get("text") or w.get("word") or "").strip()


def _select_f5_reference(
    reference_audio: Path,
    transcript_segments: list[dict] | None,
    work_dir: Path,
) -> tuple[Path, str] | None:
    """Pick a short reference clip *and the text spoken in it* for F5-TTS.

    F5-style models size the generated audio from the reference: roughly
    ref_seconds / len(ref_text) * len(gen_text). They also clip the
    reference audio to ~12 s internally. Feeding the whole 52 s source
    plus its full transcript therefore gave a per-character duration 4-5x
    too small and the output came out compressed to 9 s on the XE7740.
    The reference text must describe exactly the audio span passed.

    Returns None when word timestamps are missing, so callers can fall
    back to the whole-clip behaviour.
    """
    if not transcript_segments:
        return None
    words: list[dict] = []
    for seg in transcript_segments:
        for w in seg.get("words") or []:
            # transcribe.py serialises each word as {start, end, text}.
            if w.get("start") is not None and w.get("end") is not None and _word_text(w):
                words.append(w)
    if len(words) < 3:
        return None
    span = _longest_contiguous_word_span(words, max_gap=_REF_GAP_SECONDS)
    if span is None:
        return None
    span_start, span_end = span
    if span_end - span_start < _REF_MIN_SPAN_SECONDS:
        return None
    span_end = min(span_end, span_start + _F5_REF_MAX_SPAN_SECONDS)
    # Only words fully inside the (possibly capped) span.
    in_span = [
        w
        for w in sorted(words, key=lambda w: float(w["start"]))
        if float(w["start"]) >= span_start - 1e-3 and float(w["end"]) <= span_end + 1e-3
    ]
    if len(in_span) < 3:
        return None
    # Cut on the last whole word so the text matches the audio exactly.
    span_end = float(in_span[-1]["end"])
    ref_text = " ".join(_word_text(w) for w in in_span).strip()
    trimmed = work_dir / "f5_reference.wav"
    try:
        _ffmpeg_atrim(reference_audio, trimmed, span_start, span_end)
    except Exception as e:
        log.warning("failed to cut F5 reference audio (%s); using whole clip", e)
        return None
    log.info(
        "selected F5 reference: %.2fs span (%.2f–%.2fs), %d words",
        span_end - span_start,
        span_start,
        span_end,
        len(in_span),
    )
    return trimmed, ref_text


def _longest_contiguous_word_span(
    words: list[dict],
    max_gap: float,
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
    *,
    backend: str = "xtts",
    target_language: str | None = None,
    reference_segments: list[dict] | None = None,
    options: dict | None = None,
    source_duration_seconds: float | None = None,
) -> int:
    """Render every utterance, fit its slot, and preserve original onset times.

    Fail explicitly if content cannot fit without excessive speeding up.
    Never replace missing or failed speech with silence.
    """
    import tempfile

    options = options or {}
    target_language = target_language or language
    reference_segments = reference_segments or transcript_segments
    if not translation_segments:
        raise TTSError("no translation segments")
    generate = (
        _synthesize_indicf5_single_shot if backend == "indicf5" else _synthesize_f5tts_single_shot
    )
    timings = []
    cache = output_path.parent / "segments"
    cache.mkdir(exist_ok=True)
    from ..checkpoints import digest

    reference_hash = digest(reference_audio) if reference_audio.exists() else "missing"
    with tempfile.TemporaryDirectory(prefix="tts-", dir=output_path.parent) as temp:
        work = Path(temp)
        ref = reference_audio
        if backend == "xtts":
            selected = _select_reference(reference_audio, reference_segments, work)
            if selected is not None:
                ref = selected[0]
        paths = []
        retakes = []  # per segment: (re-speak at a given XTTS speed, speed of the kept take)
        for i, segment in enumerate(translation_segments):
            text = str(segment.get("text", "")).strip()
            if not text:
                raise TTSError(f"segment {i + 1} has no translation")
            import hashlib

            voice = options.get("speaker_voices", {}).get(segment.get("speaker")) or options.get(
                "voice"
            )
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "text": text,
                        "voice": voice,
                        "speaker": segment.get("speaker"),
                        "backend": backend,
                        "language": language,
                        "reference": reference_hash,
                        "reference_segments": reference_segments,
                        "revision": settings.model_revision,
                        "tts_model": settings.f5tts_model,
                        "cache_format": "raw-take-v2",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if voice and backend != "xtts":
                raise TTSError("Bundled voice selection requires the XTTS backend")
            speaker_segments = [
                s
                for s in reference_segments
                if not segment.get("speaker") or s.get("speaker") == segment.get("speaker")
            ]
            selected = (
                _select_reference(reference_audio, speaker_segments, work)
                if backend == "xtts" and segment.get("speaker") and not voice
                else None
            )
            segment_ref = selected[0] if selected else ref

            # XTTS native speed for retries of an overrunning segment; "take"
            # and "best" are the speeds the current and the best take were made at.
            take_speed = {"value": None, "take": 1.0, "best": 1.0}

            def generate_take(value):
                if backend == "xtts":
                    extra = {"voice": voice} if voice else {}
                    if take_speed["value"]:
                        extra["speed"] = take_speed["value"]
                    _xtts_to_file(value, segment_ref, language, path, **extra)
                    take_speed["take"] = take_speed["value"] or 1.0
                else:
                    generate(value, target_language, reference_audio, path, reference_segments)

            cached = cache / (cache_key + ".wav")
            checksum = cached.with_suffix(".sha256")
            if cached.exists() and (
                not checksum.exists() or checksum.read_text() != digest(cached)
            ):
                cached.unlink()
                checksum.unlink(missing_ok=True)
            path = work / f"segment-{i}.wav"
            if cached.exists():
                import shutil

                shutil.copyfile(cached, path)
            for attempt in range(settings.tts_segment_retries + 1):
                try:
                    if path.exists():
                        break
                    generate_take(text)
                    if not path.exists() or not path.stat().st_size:
                        raise TTSError("empty synthesized audio")
                    break
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    if attempt == settings.tts_segment_retries:
                        raise TTSError(
                            f"segment {i + 1} failed after {attempt + 1} attempts: {exc}"
                        ) from exc
            if not cached.exists():
                import shutil

                shutil.copyfile(path, cached)
                checksum.write_text(digest(cached))
            start, end = float(segment["start"]), float(segment["end"])
            # Borrow the following pause, but never overlap the next sentence.
            if i + 1 < len(translation_segments):
                end = float(translation_segments[i + 1]["start"])
            elif source_duration_seconds is not None:
                # The final utterance may use the source's trailing silence,
                # just as earlier utterances may use the pause before the next.
                import math

                if not math.isfinite(source_duration_seconds) or source_duration_seconds <= 0:
                    raise TTSError("invalid source duration for speech timing")
                end = float(source_duration_seconds)
            available = end - start
            if available <= 0:
                raise TTSError(f"segment {i + 1} has invalid or overlapping timestamps")
            # Short slots get a higher ceiling: at a conversational turn there is
            # no pause to borrow, and Spanish runs longer than English.
            ceiling = (settings.tts_max_speed_short if available < settings.tts_short_slot_seconds
                       else settings.tts_max_speed_hard)
            # A full expected-text match can distinguish clicks/repeated tails
            # from short words. Never discard content based on span duration alone.
            # XTTS takes vary a lot in length for the same sentence; a take that
            # overruns the preferred speed earns extra attempts, and the shortest
            # verified take is the one kept.
            import shutil

            best_take = work / f"segment-{i}.take.wav"
            best_take_duration = None
            attempts = settings.tts_segment_retries + 1
            cleanup_attempt = 0
            while True:
                verified = (
                    _trim_tail_via_whisper(path, target_language, text)
                    if backend == "xtts" or _probe_duration(path) > available
                    else None
                )
                if verified is not False:
                    # Also for whisper-verified takes: XTTS leaves up to a second
                    # of silence after the last word, and fitting the take to
                    # its slot would stretch that silence too, so the speech
                    # ended early and the source mouth showed through at the end.
                    _trim_to_speech(path)
                    _compress_pauses(path)
                duration = _probe_duration(path)
                if verified is not False and (best_take_duration is None or duration < best_take_duration):
                    shutil.copyfile(path, best_take)
                    best_take_duration = duration
                    take_speed["best"] = take_speed["take"]
                if verified is not False and duration <= available * settings.tts_max_speed:
                    break
                if verified is not False and duration > available * settings.tts_max_speed:
                    # Best of N: XTTS take lengths for one sentence ranged from
                    # 8.6 to 22 s on 5 Oct 2026 at every temperature, but the
                    # shortest of five always fitted. Extra attempts, granted
                    # once; the loop stops as soon as a take fits.
                    attempts = max(attempts, settings.tts_segment_retries + 1 + settings.tts_overrun_retries)
                    # Retries speak faster inside the model (more natural than
                    # stretching the waveform afterwards).
                    if backend == "xtts" and settings.tts_retry_speed > 1.0:
                        take_speed["value"] = settings.tts_retry_speed
                cleanup_attempt += 1
                if cleanup_attempt >= attempts:
                    if verified is False and best_take_duration is None:
                        raise TTSError(
                            f"segment {i + 1}: generated speech does not match the complete translation"
                        )
                    break
                # Retry the same words before asking an LLM to shorten them.
                generate_take(text)
            if best_take_duration is not None and best_take_duration < duration:
                shutil.copyfile(best_take, path)
                duration = best_take_duration
                take_speed["take"] = take_speed["best"]
            best_take.unlink(missing_ok=True)
            if duration > available:
                speed = duration / available
                if speed > settings.tts_max_speed and (
                    options.get("rewrite_overruns", settings.rewrite_overruns)
                ):
                    from .quality import rewrite

                    # Rewrites are a gamble: the LLM may not shorten the text, and a
                    # shorter text can still come back as a slower take. Keep the
                    # shortest verified take so a failed gamble never replaces a
                    # usable one with a worse one.
                    best = work / f"segment-{i}.best.wav"
                    best_duration, best_text = duration, text
                    take_speed["best"] = take_speed["take"]
                    import shutil

                    shutil.copyfile(path, best)
                    for fit_attempt in range(settings.tts_fit_retries):
                        try:
                            text = rewrite(
                                text,
                                segment.get("source_text", text),
                                target_language,
                                available,
                                options.get("glossary"),
                            )
                        except ValueError as exc:
                            log.warning(
                                "segment %d: no faithful rewrite for its %.2fs slot (%s); "
                                "keeping the %.2fs take",
                                i + 1,
                                available,
                                exc,
                                best_duration,
                            )
                            break
                        generate_take(text)
                        verified = (
                            _trim_tail_via_whisper(path, target_language, text)
                            if backend == "xtts" or _probe_duration(path) > available
                            else None
                        )
                        if verified is False:
                            continue
                        _trim_to_speech(path)
                        _compress_pauses(path)
                        duration = _probe_duration(path)
                        speed = duration / available
                        if duration < best_duration:
                            shutil.copyfile(path, best)
                            best_duration, best_text = duration, text
                            take_speed["best"] = take_speed["take"]
                        if speed <= settings.tts_max_speed:
                            break
                    # The best take is a verified (or ambiguous) one from before the
                    # rewrites; a rejected rewrite never replaces it.
                    shutil.copyfile(best, path)
                    best.unlink()
                    duration, text = best_duration, best_text
                    take_speed["take"] = take_speed["best"]
                    speed = duration / available
                    if text != segment["text"]:
                        segment["original_text"] = segment.get("original_text", segment["text"])
                        segment["text"] = text
            # Fitting is decided for all segments together below: a line that
            # overruns its own slot may borrow from the pauses that follow it.
            kept = work / f"segment-{i}.kept.wav"
            shutil.copyfile(path, kept)
            paths.append(kept)
            if backend == "xtts":
                def retake(out, speed, text=text, segment_ref=segment_ref, voice=voice):
                    _xtts_to_file(text, segment_ref, language, out, speed=speed,
                                  **({"voice": voice} if voice else {}))
                    if _trim_tail_via_whisper(out, target_language, text) is False:
                        return None
                    _trim_to_speech(out)
                    _compress_pauses(out)
                    return _probe_duration(out)

                retakes.append((retake, take_speed["take"]))
            else:
                retakes.append(None)
            timings.append(
                {
                    "segment": i,
                    "start": start,
                    "slot_end": end,
                    "speech_seconds": duration,
                    "natural_seconds": duration,
                    "ceiling": ceiling,
                    "text": text,
                }
            )
        plan_args = (
            [t["start"] for t in timings],
            [t["natural_seconds"] for t in timings],
            [t["slot_end"] for t in timings],
        )
        plan = _plan_timeline(*plan_args, [t["ceiling"] for t in timings], timings[-1]["slot_end"],
                              max_drift=settings.tts_max_drift_seconds)
        tail = float(options.get("tail_overlap_seconds") or 0.0)
        if plan is None and tail > 0:
            # At a turn the next speaker starts at once. Let this one finish
            # over the next span's first moments, as people do in conversation.
            plan = _plan_timeline(*plan_args, [t["ceiling"] for t in timings], timings[-1]["slot_end"] + tail,
                                  max_drift=settings.tts_max_drift_seconds)
            if plan is not None:
                timings[-1]["tail_overlap"] = True
        if plan is None and settings.tts_max_speed_last_resort > max(t["ceiling"] for t in timings):
            # Faster than we like, but one line must not fail an hour-long job.
            # Every line above its ceiling is marked in the timing file.
            plan = _plan_timeline(*plan_args, [settings.tts_max_speed_last_resort] * len(timings),
                                  timings[-1]["slot_end"] + tail, max_drift=settings.tts_max_drift_seconds)
            if plan is not None:
                for t in timings:
                    t["normal_ceiling"], t["ceiling"] = t["ceiling"], settings.tts_max_speed_last_resort
        if plan is None:
            worst = max(timings, key=lambda t: t["natural_seconds"] / max(t["slot_end"] - t["start"], 1e-3))
            raise TTSError(
                f"segment {worst['segment'] + 1} needs {worst['natural_seconds']:.2f}s in a "
                f"{worst['slot_end'] - worst['start']:.2f}s slot and the following pauses cannot absorb it; "
                "shorten the translation or use a faster TTS voice. No speech was discarded."
            )
        placements, factors = plan
        placed_segments = []
        for t, path, placed, factor, segment, redo in zip(timings, paths, placements, factors,
                                                           translation_segments, retakes):
            if factor > 1.0 + 1e-6:
                if factor > settings.tts_max_speed:
                    log.warning("segment %d: %.2fx (above the preferred %.2fx)", t["segment"] + 1, factor,
                                settings.tts_max_speed)
                target = round(t["natural_seconds"] / factor, 3)
                if redo is not None and factor >= settings.tts_native_speed_min:
                    native = _respeak_faster(path, target, factor, *redo)
                    if native:
                        t["native_speed"] = native
                _maybe_time_stretch(path, target_duration=target, max_speed=t["ceiling"])
                t["speech_seconds"] = _probe_duration(path)
            t["placed_start"] = round(placed, 3)
            t["speed"] = round(factor, 3)
            if factor > t.get("normal_ceiling", t["ceiling"]) + 1e-6:
                t["last_resort"] = True
                log.warning("segment %d spoken at %.2fx (last resort): %s", t["segment"] + 1, factor, t["text"][:80])
            t["drift_seconds"] = round(placed - t["start"], 3)
            placed_segments.append({**segment, "start": placed})
        drifted = [t for t in timings if t["drift_seconds"] > 0.05]
        if drifted:
            log.info("timeline: %d of %d segments start later than the source (max %.2fs) to absorb overruns",
                     len(drifted), len(timings), max(t["drift_seconds"] for t in drifted))
        _assemble_timeline(paths, placed_segments, output_path)
    output_path.with_suffix(".timing.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2)
    )
    return len(paths)


def _respeak_faster(path: Path, target: float, factor: float, retake, base_speed: float) -> float | None:
    """Re-speak a line at XTTS's own speed instead of stretching the waveform.

    The model speaks faster with natural articulation; a stretch of the
    waveform smears it (PESQ 4.1 vs 3.9 at 1.1x and 3.8 vs 3.6 at 1.25x
    against atempo, 8 Oct 2026). Up to `tts_native_speed_takes` verified
    takes; the shortest one that beats the current take replaces it, and the
    caller's stretch covers whatever is still over `target`. Returns the
    XTTS speed of the take kept, or None when the original take stays."""
    speed = min(2.0, round(base_speed * factor, 3))
    current = _probe_duration(path)
    kept = None
    for n in range(settings.tts_native_speed_takes):
        candidate = path.with_name(f"{path.stem}.native{n}.wav")
        try:
            duration = retake(candidate, speed)
        except Exception as exc:
            log.warning("native-speed retake of %s failed: %s", path.name, exc)
            duration = None
        if duration is None or duration >= current:
            candidate.unlink(missing_ok=True)
            continue
        candidate.replace(path)
        current, kept = duration, speed
        if current <= target + 0.05:
            break
    return kept


def _plan_timeline(starts, durations, slot_ends, ceilings, end_limit, *, max_drift=1.5,
                   preferred=None, tolerance=1e-6):
    """Where each utterance starts and how much it is sped up, decided for the
    whole run of segments instead of one slot at a time.

    Each utterance starts at its source onset or right after the previous one,
    whichever is later, and is sped up only as much as it needs for its own slot,
    but at most `g`. The smallest `g` (up to each segment's ceiling) for which
    every start stays within `max_drift` of the source and the last utterance
    ends by `end_limit` wins. So a line that overruns borrows the pauses after
    it, and the overrun is shared out before anything is sped up hard. The lip
    sync follows the audio, so a start that drifts by a second only shifts the
    speech against the speaker's gestures.

    Returns (placements, speed factors), or None when nothing fits."""
    preferred = settings.tts_max_speed if preferred is None else preferred
    n = len(starts)
    if n == 0:
        return [], []

    def layout(g):
        placements, factors, cursor = [], [], None
        for s, d, e, cap in zip(starts, durations, slot_ends, ceilings):
            own = d / max(e - s, 1e-3)
            f = min(cap, max(1.0, min(own, g)))
            p = s if cursor is None else max(s, cursor)
            if p - s > max_drift + 1e-9:
                return None
            placements.append(p)
            factors.append(f)
            cursor = p + d / f
        if cursor > end_limit + tolerance:
            return None
        return placements, factors

    top = max(ceilings)
    if layout(top) is None:
        return None
    unhurried = layout(1.0)
    if unhurried is not None:
        return unhurried
    # Prefer staying at or under the preferred speed when that fits.
    low = 1.0
    if layout(min(preferred, top)) is not None:
        high = min(preferred, top)
    else:
        low, high = min(preferred, top), top
    for _ in range(30):
        mid = (low + high) / 2
        if layout(mid) is None:
            low = mid
        else:
            high = mid
    return layout(high)


def _assemble_timeline(seg_paths, transcript_segments, output_path):
    """Place complete utterances at their original offsets, without overlap."""
    if (
        not seg_paths
        or len(seg_paths) != len(transcript_segments)
        or any(p is None for p in seg_paths)
    ):
        raise TTSError("all translated segments must have audio")
    origin = float(transcript_segments[0]["start"])
    filters = []
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for i, (path, segment) in enumerate(zip(seg_paths, transcript_segments)):
        delay = round((float(segment["start"]) - origin) * 1000)
        if delay < 0:
            raise TTSError("segments must be in chronological order")
        cmd.extend(["-i", str(path)])
        filters.append(
            f"[{i}:a]aresample=24000,aformat=channel_layouts=mono,adelay={delay}:all=1[a{i}]"
        )
    labels = "".join(f"[a{i}]" for i in range(len(seg_paths)))
    duration = float(transcript_segments[-1]["end"]) - origin
    filters.append(
        f"{labels}amix=inputs={len(seg_paths)}:normalize=0:duration=longest,"
        f"apad=whole_dur={duration:.6f}[out]"
    )
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[out]", str(output_path)])
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode:
        raise TTSError(f"timeline assembly failed: {proc.stderr.decode(errors='replace')[-1000:]}")


# --------------------------------------------------------------------------- #
# XTTS call helpers
# --------------------------------------------------------------------------- #


def _xtts_to_file(
    text: str,
    reference_audio: Path,
    language: str,
    output: Path,
    voice: str | None = None,
    speed: float | None = None,
) -> None:
    """Single XTTS call that writes to `output`."""
    tts = _get_xtts()
    tts.tts_to_file(
        text=text,
        **({"speaker": voice} if voice else {"speaker_wav": str(reference_audio)}),
        language=language,
        file_path=str(output),
        **({"speed": float(speed)} if speed else {}),
    )
    if not output.exists() or output.stat().st_size == 0:
        raise TTSError(f"XTTS produced no output at {output}")


def _synthesize_whole(
    text: str,
    reference_audio: Path,
    language: str,
    output_path: Path,
) -> None:
    """Legacy single-shot path. One XTTS call for the entire translation."""
    _xtts_to_file(text, reference_audio, language, output_path)


# --------------------------------------------------------------------------- #
# Silence trimming
# --------------------------------------------------------------------------- #

_SILENCE_THRESHOLD_DB = -40.0  # retain quiet words; text alignment handles edge artifacts
_MIN_SILENCE_SECONDS = 0.10  # how long a quiet stretch needs to be to count
_HEADROOM_SECONDS = 0.10  # pad on either side of the kept span
_MIN_SPEECH_SECONDS = 0.30  # below this, assume detection failed and skip


def _trim_to_speech(audio_path: Path) -> None:
    """Trim `audio_path` in place to [first speech span, last speech span].

    XTTS tends to emit a short click at the start, a long pause, the actual
    speech, more silence, and a trailing blip. `silenceremove` can't handle
    that shape because the click is above any reasonable threshold.

    Instead: find silence boundaries via ffmpeg `silencedetect`, derive the
    non-silent spans and keep everything from the first span to the last.
    Even short edge spans may contain a word and must be preserved.

    This used to keep only the *longest* span. That silently deleted every
    sentence but one whenever the synthesised text had sentence pauses
    longer than the silence threshold — F5-TTS produced 4.5 s of audio for
    nine sentences on the XE7740 before this was caught.
    """
    spans = _non_silent_spans(audio_path)
    speech = [(s, e) for s, e in spans if e > s]  # Preserve short edge words too.
    if not speech:
        log.info("no speech span detected; leaving %s untouched", audio_path.name)
        return

    start = min(s for s, _ in speech)
    end = max(e for _, e in speech)
    total = _probe_duration(audio_path)
    start = max(0.0, start - _HEADROOM_SECONDS)
    end = min(total, end + _HEADROOM_SECONDS)

    _ffmpeg_atrim(audio_path, audio_path, start, end)
    log.info(
        "trimmed %s: %.2fs -> %.2fs (kept %.2f-%.2f)",
        audio_path.name,
        total,
        end - start,
        start,
        end,
    )


def _compress_pauses(audio_path: Path, max_gap: float | None = None) -> float:
    """Cap every pause inside `audio_path` at `max_gap` seconds, in place.

    XTTS copies the reference speaker's rhythm and on 5 Oct 2026 produced
    takes with 8-9 s of silence inside 17-26 s of audio (single gaps of 3-5 s)
    for a sentence that takes 10 s to say. Fitting such a take to its slot
    would have to speed the words up to pay for the silence, so the pauses are
    shortened instead; the words are untouched. Returns the seconds removed."""
    import numpy as np
    import soundfile as sf

    max_gap = settings.tts_max_pause_seconds if max_gap is None else max_gap
    if max_gap <= 0:
        return 0.0
    spans = _non_silent_spans(audio_path)
    if len(spans) < 2:
        return 0.0
    data, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    keep = np.ones(len(data), dtype=bool)
    removed = 0.0
    for (_, end), (start, _) in zip(spans, spans[1:]):
        gap = start - end
        if gap <= max_gap:
            continue
        cut0 = int((end + max_gap / 2) * rate)
        cut1 = int((start - max_gap / 2) * rate)
        if cut1 > cut0:
            keep[cut0:cut1] = False
            removed += (cut1 - cut0) / rate
    if removed <= 0.0:
        return 0.0
    sf.write(str(audio_path), data[keep], rate)
    log.info("compressed pauses in %s: removed %.2fs (gaps capped at %.2fs)", audio_path.name, removed, max_gap)
    return removed


def _ffmpeg_atrim(src: Path, dst: Path, start: float, end: float) -> None:
    """Write `src[start:end]` to `dst`. In-place safe (uses a temp path)."""
    # Keep the real extension (.wav) so ffmpeg can infer the output format.
    # `.with_suffix(suffix + ".trim")` produced "foo.wav.trim", which ffmpeg
    # refuses because .trim isn't a known format.
    tmp = dst.parent / f"{dst.stem}.trim{dst.suffix}"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(src),
        "-af",
        f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atrim failed: {proc.stderr.decode(errors='replace')}")
    tmp.replace(dst)


def _probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        timeout=30,
    )
    return float(out.decode().strip())


def _non_silent_spans(path: Path) -> list[tuple[float, float]]:
    """Return [(start, end), ...] of non-silent spans inside `path`."""
    duration = _probe_duration(path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={_SILENCE_THRESHOLD_DB}dB:duration={_MIN_SILENCE_SECONDS}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
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

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
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
    expected_text: str,
    transcribed_words: list[tuple[float, str]],
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

    ei = 0  # index into expected
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


_PERCENT_WORDS = {"es": "por ciento", "pt": "por cento", "it": "percento", "fr": "pour cent",
                  "de": "prozent", "en": "percent", "nl": "procent", "pl": "procent"}


def _spell_numbers(text: str, language: str | None) -> str:
    """Digits and % as words in `language`, so "88" in the text matches a spoken
    "ochenta y ocho" that the recognizer may write either way."""
    import re

    lang = (language or "").lower()
    text = text.replace("%", f" {_PERCENT_WORDS[lang]} " if lang in _PERCENT_WORDS else " ")
    try:
        from num2words import num2words
    except ImportError:
        return text

    def spell(match):
        raw = match.group(0)
        try:
            value = float(raw.replace(",", ".")) if re.search(r"[.,]\d", raw) else int(raw)
            return f" {num2words(value, lang=lang or 'en')} "
        except Exception:
            return raw

    return re.sub(r"\d+(?:[.,]\d+)?", spell, text)


def _near_match(expected: str, heard: str, threshold: float = 0.92) -> bool | None:
    """A recognition that differs only slightly from the text (an accent, a
    split compound, a number read differently) is ambiguous, not different
    speech: keep the take whole (None) instead of rejecting it (False)."""
    from difflib import SequenceMatcher

    if not expected or not heard:
        return False
    return None if SequenceMatcher(None, expected, heard).ratio() >= threshold else False


def _trim_tail_via_whisper(
    audio_path: Path,
    target_language: str,
    expected_text: str,
    source_duration_seconds: float | None = None,
) -> bool | None:
    """Trim only after complete, confident text alignment.

    True: all expected text recognized; False: confidently recognized different
    text; None: unavailable/ambiguous, leave audio intact. No duration fallback.
    Whitespace/punctuation differences are ignored, but words and repetitions
    must match exactly. This intentionally declines fuzzy ASR matches.
    """
    import unicodedata

    def normalize(text):
        text = _spell_numbers(unicodedata.normalize("NFKC", text), target_language)
        return "".join(c for c in text.casefold() if c.isalnum())

    expected = normalize(expected_text)
    if not expected:
        return None
    try:
        from .transcribe import _get_model

        segments, _ = _get_model().transcribe(
            str(audio_path),
            language=target_language or None,
            beam_size=1,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        words = [
            w for seg in segments for w in (getattr(seg, "words", None) or []) if normalize(w.word)
        ]
    except Exception as exc:
        log.warning("speech validation unavailable; retaining full audio: %s", exc)
        return None
    if not words or any(
        w.start is None or w.end is None or getattr(w, "probability", 0) < 0.8 for w in words
    ):
        return None
    matched = ""
    last = None
    for i, word in enumerate(words):
        matched += normalize(word.word)
        if not expected.startswith(matched):
            return _near_match(expected, "".join(normalize(w.word) for w in words))
        if matched == expected:
            last = i
            break
    if last is None:
        return _near_match(expected, matched)
    total = _probe_duration(audio_path)
    start, end = 0.0, total
    # Only remove a leading isolated click when every expected word is accounted
    # for after it; a short recognized first word is always retained.
    first = float(words[0].start)
    spans = _non_silent_spans(audio_path)
    if spans:
        # Recognized quiet edge words can fall below the energy threshold.
        start = max(0.0, min(first, spans[0][0]) - 0.08)
        end = min(total, max(float(words[last].end), spans[-1][1]) + 0.08)
    if spans and 0 < spans[0][1] - spans[0][0] <= 0.08 and first - spans[0][1] >= 0.2:
        start = max(0.0, first - 0.08)
    if last + 1 < len(words):
        # Timestamp overlap is ambiguous; keep it rather than slicing a word.
        boundary = float(words[last].end)
        if float(words[last + 1].start) >= boundary + 0.08:
            end = min(total, boundary + 0.08)
        else:
            return False  # resynthesize rather than cutting overlapping words
    if start > 0 or total - end >= 0.1:
        _ffmpeg_atrim(audio_path, audio_path, start, end)
    return True


def _maybe_time_stretch(
    audio_path: Path, target_duration: float, max_speed: float | None = None
) -> None:
    """Speed `audio_path` up to ~`target_duration` seconds with ffmpeg's atempo.

    No-op when the current duration is already within the target or when
    the required ratio is too aggressive. atempo replaced rubberband on
    8 Oct 2026: on the same XTTS line rubberband (R2 --formant, and R3)
    scored PESQ 1.8 at 1.1x and 1.4-1.5 at 1.25x, heard as static and
    scratchiness, where atempo scored 3.9 and 3.6.
    """
    current = _probe_duration(audio_path)
    if current <= target_duration + 0.05:
        # Already at or under the target — nothing to do.
        return

    max_speed = max_speed or settings.tts_max_speed
    ratio = target_duration / current
    if ratio < 1.0 / max_speed:
        log.info(
            "time-stretch would need %.2fx (< min %.2fx); retaining complete audio for caller to handle",
            ratio,
            1.0 / max_speed,
        )
        return

    tmp = audio_path.parent / f"{audio_path.stem}.stretch{audio_path.suffix}"
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
         "-filter:a", f"atempo={1.0 / ratio:.4f}", str(tmp)],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"atempo exit {proc.returncode}: {proc.stderr.decode(errors='replace')[-500:]}"
        )
    tmp.replace(audio_path)
    log.info(
        "time-stretched %s: %.2fs -> %.2fs (ratio %.3f)",
        audio_path.name,
        current,
        target_duration,
        ratio,
    )


# --------------------------------------------------------------------------- #
# Source-aligned silence prepend
# --------------------------------------------------------------------------- #


def _prepend_silence(audio_path: Path, seconds: float) -> None:
    """Prepend `seconds` of silence to `audio_path` (in place) via ffmpeg.

    Keeps the input's sample rate, channels and 16-bit depth. The result is an
    audio file whose first spoken frame sits at `seconds` — aligning the
    TTS with the source clip's pre-speech silence.
    """
    if seconds <= 0:
        return

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=nw=1",
            str(audio_path),
        ],
        capture_output=True,
        timeout=30,
    )
    sr = 24000
    for line in probe.stdout.decode(errors="replace").splitlines():
        if line.startswith("sample_rate="):
            sr = int(line.split("=", 1)[1])

    tmp = audio_path.parent / f"{audio_path.stem}.pad{audio_path.suffix}"
    # adelay on the speech itself. This used to concat an anullsrc lead in
    # front of it, and ffmpeg negotiated that graph to unsigned 8-bit: every
    # span with a pause before its first word was quantized to 256 levels,
    # heard as static (PESQ 3.86 -> 3.26, found 8 Oct 2026).
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-af",
        f"adelay=delays={round(seconds * sr)}S:all=1",
        "-c:a",
        "pcm_s16le",
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

_LOUDNORM_TARGET_I = -16.0  # LUFS integrated
_LOUDNORM_TARGET_TP = -1.5  # dBTP true-peak ceiling
_LOUDNORM_TARGET_LRA = 11.0  # loudness range


def _loudnorm(audio_path: Path) -> None:
    """Apply EBU R128 loudnorm to `audio_path` in place."""
    tmp = audio_path.parent / f"{audio_path.stem}.lnorm{audio_path.suffix}"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-af",
        f"loudnorm=I={_LOUDNORM_TARGET_I}:TP={_LOUDNORM_TARGET_TP}:LRA={_LOUDNORM_TARGET_LRA}",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed: {proc.stderr.decode(errors='replace')[-500:]}")
    tmp.replace(audio_path)
    log.info("loudness-normalized %s to %.1f LUFS", audio_path.name, _LOUDNORM_TARGET_I)
