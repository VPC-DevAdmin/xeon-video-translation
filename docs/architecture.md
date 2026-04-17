# Architecture

## High-level

```
┌─────────────────┐
│   Next.js UI    │  React 18, Tailwind, react-dropzone
│  - Uploader     │
│  - LanguagePick │
│  - PipelineView │  ← consumes SSE for live stage updates
└────────┬────────┘
         │ HTTP (multipart upload, SSE for progress)
         ▼
┌─────────────────┐
│   FastAPI app   │
│   /jobs         │  POST upload + spawn pipeline
│   /jobs/{id}    │  GET status (replays from meta.json after restart)
│   /jobs/{id}/   │  GET artifacts (transcript.json, translation.json, …)
│   .../events    │  SSE stream of pipeline events
└────────┬────────┘
         │ background task on the same loop
         ▼
┌─────────────────────────────────────────────┐
│  Pipeline orchestrator                      │
│                                             │
│  Stage 1  audio        ffmpeg → 16 kHz WAV  │
│  Stage 2  transcribe   faster-whisper int8  │
│  Stage 3  translate    NLLB-200 / Ollama    │
│  Stage 4  tts          XTTS-v2 (CPU)        │
│  Stage 5  lipsync      none | wav2lip | …   │
│  Stage 6  mux          ffmpeg + drawtext    │
└─────────────────────────────────────────────┘
```

## CPU-only changes from the original spec

- **Whisper**: faster-whisper with `compute_type=int8` and a smaller default model (`base`). On a 16-core Xeon, `base` runs ~5–8× realtime. Bump to `small` for better accuracy.
- **Translation**: NLLB-200 distilled-600M is the default — fits comfortably in RAM and runs at ~2–3 s per segment on CPU. Ollama is wired as an alternate backend but is impractically slow for 8B models on CPU; only use it if you've quantized further.
- **No CUDA layers** in the Docker image. PyTorch CPU wheel is installed from the PyTorch CPU index.
- **Resource caps** in `docker-compose.yml` keep the backend from saturating all cores on a shared host.

## Job lifecycle

1. `POST /jobs` writes the upload to `jobs/<id>/input.<ext>` and creates `meta.json`.
2. A background task on the FastAPI loop runs the pipeline. Each stage:
   - Updates the `JobState` in memory
   - Persists `meta.json` after every transition (so the job survives a restart)
   - Pushes an event onto the per-job `asyncio.Queue` consumed by SSE
3. `GET /jobs/<id>/events` opens an SSE stream that drains that queue.
4. `GET /jobs/<id>/artifacts/<name>` returns any file written under the job dir.

## Why per-stage files on disk

- Each stage is independently inspectable (`transcript.json`, `translation.json`, eventually `translated_audio.wav`).
- Caching / re-running one stage is trivial.
- The UI can deep-link to artifacts.

## What's intentionally simple in v1

- **In-memory job registry** — fine for a single-machine demo with `MAX_CONCURRENT_JOBS=1`. A multi-worker deployment would need Redis.
- **No auth** — single-user, localhost-only. Don't expose this to the internet without adding auth + rate limits.
- **SSE rather than WebSockets** — one-way is enough; SSE survives proxies better.
