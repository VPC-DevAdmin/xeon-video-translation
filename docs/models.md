# Models

CPU-only choices for the polyglot-demo pipeline. All weights are downloaded
on first use into `MODEL_CACHE_DIR` (default `./models`).

## Stage 2 — Transcription

**`faster-whisper`** (CTranslate2 build of OpenAI Whisper)

| Variant      | Disk    | RAM (int8) | Speed on 16-core Xeon | Quality  |
| ------------ | ------- | ---------- | --------------------- | -------- |
| `tiny`       | ~75 MB  | ~0.5 GB    | ~15× realtime         | poor     |
| `base` ←default | ~140 MB | ~0.8 GB | ~5–8× realtime        | usable   |
| `small`      | ~480 MB | ~1.5 GB    | ~2–3× realtime        | good     |
| `medium`     | ~1.5 GB | ~3 GB      | ~0.7× realtime        | great    |
| `large-v3`   | ~3 GB   | ~5 GB      | ~0.2× realtime        | best — but slow |

`compute_type=int8` is the recommended default on CPU. Use `int8_float32` if
you see numerical artifacts on a particular clip.

## Stage 3 — Translation

**Default: NLLB-200 distilled-600M** (`facebook/nllb-200-distilled-600M`).

- ~2.4 GB on disk, runs comfortably on CPU with 4–6 GB RAM.
- ~2–3 seconds per segment, scales linearly with segment length.
- Supports the languages listed in [`backend/app/pipeline/translate.py`](../backend/app/pipeline/translate.py).
- Quality is solid for major language pairs (EN↔ES/FR/DE/JA/ZH); weaker for
  long-tail languages where the distilled model has less capacity.

**Alternate: Ollama** (`TRANSLATE_BACKEND=ollama`)

- Runs whichever model you've pulled into a local Ollama daemon.
- Llama 3.1 8B at q4 quantization works but takes 10–30 s per segment on CPU —
  too slow for the live demo. Try a 1B–3B model if you really want LLM-style
  translations on CPU.

## Stage 4 — Voice cloning / TTS

**XTTS-v2** via `coqui-tts` (`tts_models/multilingual/multi-dataset/xtts_v2`).

- ~1.8 GB on disk. First `TTS(...)` call downloads into `MODEL_CACHE_DIR/coqui-tts/`.
- ~0.3–0.7× realtime on a 16-core Xeon — a 3-second translation takes
  5–10 seconds to synthesize.
- Needs ≥ 3 seconds of reference audio. We feed the whole Stage 1 WAV.
- Supported languages (subset of NLLB's list): en, es, fr, de, it, pt, pl,
  tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi.
- **License**: the XTTS-v2 weights are released under [CPML](https://coqui.ai/cpml),
  which restricts commercial use. This demo sets `COQUI_TOS_AGREED=1` on first
  load so the library doesn't prompt interactively; do not ship outputs
  commercially without reviewing the license.

### Integration details

- **Smart reference selection.** When Whisper word timestamps are
  available, the XTTS speaker reference is trimmed to the longest
  contiguous span of clean speech (gap threshold: 300 ms, minimum span:
  3 s, cap: 12 s). Falls back to the whole source audio when no clean
  span is long enough. Produces a noticeably cleaner voice clone on
  clips with lead-in noise or mumbled intros.
- **Per-segment synthesis** when the transcript has multiple segments.
  Each Whisper segment is synthesized separately and concatenated with
  inter-segment silences derived from the original source's gaps, so the
  translated clip preserves the speaker's pause structure rather than
  collapsing to a single continuous utterance. Falls back to single-shot
  for one-segment transcripts or when segment counts don't line up.
- **Source-aligned first frame.** The first spoken syllable is padded
  with silence to match the source clip's pre-speech gap
  (`first_speech_seconds` from the transcript), so the speaker's video
  cue "pauses, then speaks" stays in sync.
- **Rubberband time-stretch (formant-preserving)** when assembled TTS
  exceeds the remaining source-video window. Caps at 0.85× ratio; below
  that we prefer freeze-padding the video over chipmunk audio.
- **EBU R128 loudness normalization** (−16 LUFS I / −1.5 dBTP / 11 LU
  LRA) applied as the final audio step. Broadcast-dialog standard.
- **De-essing** is still not applied. Add in a follow-up if needed.

## Stages 5–6 (not yet wired up)

- **Lip sync**: LatentSync needs a GPU realistically. On CPU, the demo will
  likely skip Stage 5 and just produce a video with the new audio dubbed
  over the original mouth.
- **Mux + watermark**: ffmpeg, no model.

## Where models live

```
$MODEL_CACHE_DIR/
├── faster-whisper/        # CTranslate2 weights, by model name
├── huggingface/           # HF transformers cache (NLLB lives here)
└── coqui-tts/             # XTTS-v2 weights (via TTS_HOME)
```

In Docker, `MODEL_CACHE_DIR=/models` and a named volume persists across
restarts so first-run downloads happen once.
