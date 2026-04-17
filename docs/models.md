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

## Stages 4–6 (not yet wired up)

- **TTS / voice clone**: XTTS-v2 is GPU-leaning. CPU works but is ~0.5–1×
  realtime. [`piper`](https://github.com/rhasspy/piper) is a much faster
  CPU-only fallback if voice cloning isn't required for the demo.
- **Lip sync**: LatentSync needs a GPU realistically. On CPU, the demo will
  likely skip Stage 5 and just produce a video with the new audio dubbed over
  the original mouth.
- **Mux + watermark**: ffmpeg, no model.

## Where models live

```
$MODEL_CACHE_DIR/
├── faster-whisper/        # CTranslate2 weights, by model name
└── huggingface/           # HF transformers cache (NLLB lives here)
```

In Docker, `MODEL_CACHE_DIR=/models` and a named volume persists across
restarts so first-run downloads happen once.
