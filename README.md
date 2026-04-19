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

```bash
cp .env.example .env             # port config + optional CLI defaults
make up                           # build & start backend + frontend + lipsync-musetalk
make models                       # Whisper + NLLB + XTTS  (~3 GB, one-time)
# Drop a clip in artifacts/inputs/, then:
make run-none                     # fast dub-over path; sanity check
```

Run `make help` for the full menu. Most common targets:

| Target | What it does |
|---|---|
| `make up` / `make down` | Start / stop all containers |
| `make restart` | Restart backend + lipsync-musetalk (picks up bind-mounted code) |
| `make health` | `/health` on backend + lipsync-musetalk |
| `make logs` / `make logs-musetalk` | Tail service logs |
| `make models` | Pre-fetch Whisper + NLLB + XTTS |
| `make models-wav2lip` | Adds the Wav2Lip checkpoint |
| `make models-musetalk` | MuseTalk weights (~1.4 GB) |
| `make models-all` | All of the above |
| `make run-none` / `run-wav2lip` / `run-musetalk` | Smoke test with the named backend |
| `make run-musetalk QUALITY=N` | Quality ladder (1=fastest, 3=default/best). See `docs/lipsync.md`. |
| `make list` | Recent jobs on the backend |
| `make progress` | One-shot status on the latest job (use `JOB=prefix` for specific) |
| `make watch` | Poll the latest job until it finishes (good for `run-musetalk`) |
| `make fetch` | Download latest job to `artifacts/jobs/<short>/` |
| `make fetch JOB=b0965` | Specific job by id prefix |
| `make inputs` | List fixtures under `artifacts/inputs/` |

### Config via `.env`

The Makefile reads a `.env` at the repo root. Set your defaults once to skip CLI flags:

```bash
# uncomment the relevant lines in .env
FIXTURE=artifacts/inputs/my_clip.mov
TARGET=ja
LIPSYNC=musetalk
```

Per-invocation overrides still work: `make run TARGET=fr LIPSYNC=wav2lip`.

### Ports

Both default to non-common ports to avoid clashing with other local apps:

- `FRONTEND_PORT` — default `3030`. If you change it, also update `CORS_ORIGINS`.
- `BACKEND_PORT` — default `8088`. The frontend image bakes this URL in at build time, so **changing it requires `make rebuild`**.

After `make up`: frontend at `http://localhost:3030`, backend docs at `http://localhost:8088/docs`.

### Artifacts layout

```
artifacts/
├── inputs/     your source clips (tracked in git; drop files here)
├── jobs/       fetched pipeline outputs (gitignored)
│   └── <short_id>/
│       ├── input.*
│       ├── audio.wav
│       ├── transcript.json
│       ├── translation.json
│       ├── translated_audio.wav
│       ├── lipsynced.mp4
│       └── final.mp4
└── review/     occasional committed samples for design review
```

### Local dev without Docker

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
