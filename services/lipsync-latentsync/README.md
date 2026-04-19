# lipsync-latentsync

CPU LatentSync lipsync microservice. Dep-isolated from the main backend
and from `lipsync-musetalk` — different diffusers/transformers pinning.

## Staging

| PR | What lands | `/lipsync` behavior |
|----|-----------|---------------------|
| **PR-LS-1a** (this one) | FastAPI scaffold, Compose wiring, backend HTTP client | 501 with structured body |
| **PR-LS-1b** (next) | torch 2.x CPU wheels + diffusers/transformers pins + `scripts/download_models.sh` pulls weights into `/models/latentsync/` | still 501 — container has everything on disk |
| **PR-LS-1c** (later) | Vendor LatentSync inference, first runnable frame, per-request quality knobs wired up | real lipsync |

## CPU realism

LatentSync is SD 1.5 latent diffusion, 20 denoising steps per frame at
24 fps. Budget ~10 minutes of wall-clock per second of source video on
a 16-core Xeon. This service is for **batch workflows** (overnight
translations for archival / research), not live demos.

See `docs/lipsync.md` at the repo root for the full architecture note.

## Runtime introspection

```bash
curl http://localhost:${LATENTSYNC_PORT:-8090}/health
curl http://localhost:${LATENTSYNC_PORT:-8090}/ready
curl http://localhost:${LATENTSYNC_PORT:-8090}/weights
```

In PR-LS-1a `/ready` intentionally reports every dep missing (no torch
yet) and `/weights` reports every file missing (no download script yet).
Those flip green in PR-LS-1b.
