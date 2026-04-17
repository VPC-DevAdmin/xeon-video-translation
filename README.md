# polyglot-demo

> An open-source video translation demo: take a short clip of someone speaking, produce a new clip of the same person saying the same thing in another language — with their own voice and (eventually) lip-synced mouth movements. Designed for live conference demos.
>
> **This repo targets CPU-only inference on Xeon-class machines.** No GPU required.

---

## Responsible Use

This tool exists to demonstrate, transparently, what end-to-end open-source media translation can do. **Read [docs/ethics.md](docs/ethics.md) before using it.**

- All outputs are watermarked as AI-generated. The watermark is on by default and should remain on.
- Only use this on people who have explicitly consented to be recorded and translated.
- Do not use it to impersonate anyone, public or private.
- Always disclose AI generation when sharing outputs.

If you find yourself wanting to disable the watermark for any non-internal-testing reason, stop and reconsider.

---

## Status

Milestones implemented:

- **M1** — Repo skeleton, Docker Compose, FastAPI backend, Next.js frontend, ffmpeg audio extraction.
- **M2** — Whisper transcription (`faster-whisper`, int8 CPU build) + NLLB-200 translation.
- **M3** — Voice cloning with XTTS-v2; produces `translated_audio.wav` in the speaker's own voice.
- **M4** — Selectable lipsync backend (`none` / `wav2lip` / `musetalk`‑stub / `latentsync`‑stub), final mux + watermark, per-stage ETA + live progress over SSE. End-to-end producing a watermarked `final.mp4`.

Still to come: MuseTalk and LatentSync proper integrations (separate PRs); conference-polish milestone (M5).

---

## Quick start

### Option A — Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build
# Frontend: http://localhost:3030     (override with FRONTEND_PORT in .env)
# Backend:  http://localhost:8088/docs (override with BACKEND_PORT in .env)
```

Both ports are configurable via `.env` to avoid clashes with other local apps:

- `FRONTEND_PORT` — default `3030`. If you change it, also update `CORS_ORIGINS`.
- `BACKEND_PORT` — default `8088`. The frontend image bakes this URL into its
  client bundle, so **changing `BACKEND_PORT` requires a rebuild**
  (`docker compose up --build`).

First run downloads model weights into the `models` volume — expect a few minutes for Whisper `base` + NLLB-600M (~3 GB total).

### Option B — Local dev (Python + Node directly)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

You will need `ffmpeg` available on `PATH`.

### Smoke test

Drop a short single-speaker clip at `backend/tests/fixtures/sample_5s.mp4`, then:

```bash
./scripts/smoke_test.sh
```

This uploads the clip, polls until done, and prints the transcript and translation.

---

## Architecture (M1 + M2)

```
Video upload ──► Stage 1: audio extract (ffmpeg, 16kHz mono WAV)
              └► Stage 2: transcribe   (faster-whisper int8, CPU)
              └► Stage 3: translate    (NLLB-200 distilled-600M, CPU)
              └► [stages 4–6: TTS, lipsync, mux — not yet implemented]
```

Each stage writes its output to disk under `jobs/<job_id>/`, so artifacts can be inspected directly and the UI can show intermediate results.

For the full design see [docs/architecture.md](docs/architecture.md).

---

## Why CPU-only?

This fork targets booth/laptop demos on Xeon machines without GPUs. Trade-offs:

| Stage         | CPU choice                          | Approx. speed on 16-core Xeon           |
| ------------- | ----------------------------------- | --------------------------------------- |
| Transcription | `faster-whisper base`, int8         | ~5–8× realtime                          |
| Translation   | NLLB-200 distilled-600M             | ~2–3 sec per segment                    |
| TTS           | XTTS-v2 (Coqui)                     | ~0.3–0.7× realtime                      |
| Lipsync       | `none` (default) / `wav2lip`        | 0s · ~15× source duration               |
| Mux+wm        | ffmpeg drawtext                     | <1 s                                    |

See [docs/lipsync.md](docs/lipsync.md) for why MuseTalk and LatentSync are stubbed — TL;DR: measured-to-projected CPU latency is impractical.

See [docs/limitations.md](docs/limitations.md) for honest expectations.

---

## Repo layout

```
backend/    FastAPI app, pipeline stages, model wrappers
frontend/   Next.js 14 demo UI
scripts/    download_models.sh, smoke_test.sh, benchmark.py
docs/       architecture, models, limitations, ethics
```

---

## License

[Apache 2.0](LICENSE).
