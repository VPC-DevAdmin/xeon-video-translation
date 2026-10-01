# GPU track

The CPU build was optimised as far as it goes and still missed the minimum
bar for quality and turnaround. This track targets a **Dell PowerEdge
XE7740 with 8x RTX PRO 6000 Blackwell (96 GB each)** and three user-facing
modes. The CPU stack stays runnable from the same repo; everything here is
additive and opt-in via `docker-compose.gpu.yml`.

| Mode | Doc | Input | Target | Lipsync | Status |
|---|---|---|---|---|---|
| 1. Real-time translation | [mode1-realtime.md](mode1-realtime.md) | webcam (WebRTC) | result within a few minutes of stopping; quality traded for speed | MuseTalk | scaffolded |
| 2. Batch translation | [mode2-batch.md](mode2-batch.md) | webcam or upload | best achievable quality; time unconstrained | LatentSync 512, sharded over GPUs | scaffolded |
| 3. Real-time voice avatar | [mode3-avatar.md](mode3-avatar.md) | mic + one still image | conversational latency, continuous | image-driven talking head | design only |

"Scaffolded" means: the ingest path, mode→parameter mapping, GPU compose
overlay, backend device plumbing and the CUDA backend Dockerfile exist on
this branch; the lipsync services have not yet been ported and nothing
has been run on the target hardware.

## What is on this branch

- `backend/app/config.py` — `DEVICE=cpu|cuda|auto`, resolved once and used
  by whisper, NLLB, XTTS, F5-TTS, IndicF5 and Wav2Lip. CPU remains the
  default so the existing stack is unchanged. NLLB loads fp16 on CUDA.
- `backend/Dockerfile.gpu` — CUDA 12.8 + torch cu128. Blackwell needs this;
  the CPU image's torch 2.6 has no sm_120 kernels.
- `docker-compose.gpu.yml` — overlay with per-service GPU assignment, GPU
  model defaults (whisper large-v3 fp16, NLLB 3.3B), and the ingest service.
- `services/ingest-webrtc/` — aiortc service: browser offer → recording →
  job submission with the mode's parameters.
- `frontend/app/live/page.tsx` — webcam capture page with a mode selector.
- `make up-gpu`, `make health-gpu`.

## Hardware assumptions

- 8 GPUs, 96 GB each, CUDA 12.8+ driver, NVIDIA Container Toolkit.
- Verify before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## GPU assignment

| GPU | Service | Why |
|---|---|---|
| 0 | backend (whisper large-v3, NLLB 3.3B, TTS) | all three fit in <40 GB; keeps the chatty stages co-located |
| 1 | lipsync-musetalk | mode 1 needs it warm and alone |
| 2–7 | lipsync-latentsync workers | mode 2 shards 16-frame chunks across them |

Until chunk sharding lands (PR-G5) only GPU 2 runs LatentSync. Mode 3 will
take a GPU from the LatentSync pool when it exists.

## Roadmap (PR series)

Each PR is independently mergeable. Order matters for the first four.

| PR | Scope | Unblocks |
|---|---|---|
| **G0** (this branch) | device plumbing, GPU Dockerfile, compose overlay, ingest service, live page, docs | everything |
| **G1** | Bring up backend on GPU 0. Verify XTTS on torch 2.7 (known risk), faster-whisper float16, NLLB 3.3B fp16. Record real per-stage timings into `orchestrator.py` ETA table. Add a warmup hook in `main.py` so the first job doesn't pay load time. | modes 1, 2 |
| **G2** | MuseTalk CUDA port: Dockerfile.gpu, drop IPEX/tcmalloc/KMP, `DEVICE` env, `onnxruntime-gpu` + CUDA provider for SCRFD, batch VAE encode and CodeFormer. Target: ≥ real-time on one GPU. | mode 1 |
| **G3** | LatentSync CUDA port: same treatment, fp16 weights, model singleton (today it rebuilds per request), keep DeepCache opt-in. Re-run `stability_metric.py` to confirm the CPU bf16 jitter does not reappear. | mode 2 |
| **G4** | Job queue: replace the in-memory registry and `MAX_CONCURRENT_JOBS` semaphore with Redis + per-stage workers so modes 1 and 2 run concurrently without starving each other. | throughput |
| **G5** | LatentSync chunk sharding across GPUs 2–7 (16-frame chunks are independent given audio features + shared noise). Near-linear speedup. | mode 2 turnaround |
| **G6** | Per-utterance duration fitting: length-aware translation prompt (LLM backend) + TTS speed parameter, replacing the whole-file rubberband stretch. Biggest remaining quality lever for all modes. | modes 1, 3 |
| **G7** | Mode 3 avatar: live consumer on the ingest track, streaming ASR, LLM turn, streaming TTS, image-driven talking head, WebRTC return path. See mode3-avatar.md. | mode 3 |
| **G8** | Translation backend upgrade: LLM (70B class fits one card) with context window across segments and length control; keep NLLB as the fast fallback. | quality |

## Decisions already made

- **Separate Dockerfiles, not build args.** Base image, torch index and
  optional deps all differ; a branchy single Dockerfile was more fragile.
- **Record-then-submit for modes 1 and 2.** The existing job model is "a
  file appeared". Sub-utterance streaming is only needed for mode 3 and
  gets its own consumer on the same WebRTC track.
- **Ingest re-uploads over loopback** instead of adding a path-adopt
  endpoint to the backend. Simpler; revisit if clips exceed a few minutes.
- **CPU stack stays green.** `DEVICE` defaults to `cpu`; the CPU compose
  file is untouched.

## Confirmed by the owner (2026-10-01)

- **Hardware:** RTX PRO 6000 Blackwell. sm_120, so CUDA 12.8+ and torch
  2.7+ are required; the Dockerfile assumptions hold.
- **Licensing:** this is a demo. Open-source models are the requirement;
  non-commercial weights (XTTS CPML, F5-TTS and Wav2Lip CC-BY-NC) and
  gated repos (IndicF5) are acceptable. No need to swap TTS for licensing
  reasons; model choice is on quality and speed alone.
