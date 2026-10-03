# GPU testing handoff

## Status

The local implementation spans all six planned phases. It is ready for GPU integration testing, subject to the hardware checks below. No new GPU inference, generated-face quality, CUDA compatibility, or latency claims have been validated in this pass.

### What changed

| Phase | Implemented locally |
| --- | --- |
| Reliability | SQLite job registry, owner isolation, single-dispatcher lock, execution leases, restart handling, checksum checkpoints, immutable revisions, targeted retries, cancellation that drains native work. |
| Measurement | Stage and queue timing with p50/p95, process peak RSS, disk/GPU samples, provenance, model manifest tool, benchmark comparisons and evaluation cases. |
| Execution | Overlapping bounded render windows, direct media seeking, resumable window checksums, priority scheduling with aging, quotas, periodic retention, avatar file cleanup and storage limits. |
| Translation/audio | Context and glossary validation, number checks, editable transcript/translation, speaker assignments and XTTS voice previews, speech cache, optional duration rewrites, isolated alignment/diarization/separation service, background ducking/remix. |
| Avatar | CPU Silero ONNX VAD integration, configurable endpointing, bounded generation/render queues, renderer timeout with audio fallback, audio context overlap, played-sentence acknowledgments, interruption fencing, provisional microphone captions and telemetry. |
| Product/operations | Studio history/queue/editor, cancel/retry/delete, original/result playback, subtitles and bundles, capture timer and connection diagnostics, token accounts, HttpOnly cookies, expiring TURN credentials, readiness and CI/browser coverage. |

Webcam translation records through WebRTC and starts the job on Stop. Captions during recording are provisional. Avatar replies stream during the conversation. The portrait renderer animates lips; it does not generate head movement.

### Local evidence

- **82 backend/worker tests**: includes real ffmpeg extraction, watermark/mux, background remix, window duration/tail handling, cached rendering and corruption recovery; API ownership, revision, scheduler, cache and model-adapter contracts.
- **15 ingest/avatar tests**: includes actual local ICE/DTLS/SRTP audio/video transport, real Silero ONNX inference, fallback/cleanup, interruption acknowledgments, TURN credentials and bounded buffers.
- **6 Chromium workflow tests**: real Next.js proxy with deterministic API fixtures; editing, glossary/voice settings, sign-in/out, camera denial, submission retry and avatar controls.
- Production frontend build and Linux frontend container build pass. Linux ingest container build/import checks pass. Compose validates with GPU, avatar and optional audio profiles.
- The optional WhisperX/Demucs service dependency graph resolves for Linux x86_64/Python 3.11; `requirements.lock` records the selected versions.
- Frontend dependency audit reports **zero vulnerabilities** with the checked lockfile. Updated to Next 15.5.27, React 19.3.0, Tailwind 4.3.3 and PostCSS 8.5.28.

Model fixtures in tests verify application contracts, not translation, speech or face quality. Silero silence inference does not establish microphone accuracy. The local host's Homebrew ffmpeg lacks drawtext; watermark tests used imageio-ffmpeg 0.6.0's ffmpeg 7.1 binary. The Linux images install ffmpeg independently.

## Reproduce local checks

Use separate Python environments: backend tests avoid downloading speech models; ingest needs aiortc, PyAV and ONNX Runtime.

```sh
python3.11 -m venv .venv-test
.venv-test/bin/pip install -r backend/requirements-test.txt
# ffmpeg and ffprobe must be on PATH; ffmpeg must include drawtext.
PYTHONPATH=backend .venv-test/bin/python -m pytest backend/tests services/lipsync-latentsync/tests -q

python3.11 -m venv .venv-ingest
.venv-ingest/bin/pip install -c services/ingest-webrtc/requirements.lock -e ./services/ingest-webrtc pytest pytest-asyncio
python3 scripts/fetch_vad.py /tmp/silero.onnx
SILERO_TEST_MODEL=/tmp/silero.onnx PYTHONPATH=services/ingest-webrtc .venv-ingest/bin/python -m pytest services/ingest-webrtc/tests -q

cd frontend
npm ci
npm run build
npx playwright install chromium
npm run test:e2e
npm audit --audit-level=high
```

Real transport tests need permission to open local UDP sockets. The browser suite uses fixed localhost ports 3188 and 18088, synthetic API data, and a simulated peer for the submission-retry test. Python tests separately establish real WebRTC media transport.

## Bring up on the GPU host

1. Back up the jobs volume **with services stopped**. Include `.state.sqlite3`, `-wal` and `-shm` files if present, media, and JSON metadata. Keep the current source tree and image digests for rollback.
2. Check `nvidia-smi` and set nonoverlapping GPU assignments. Suggested dedicated layout: backend 0, MuseTalk 1, LatentSync 2–4, avatar speech 5, avatar rendering 6, optional audio quality 7. Override occupied devices; these are examples, not discovered availability.
3. Preserve `.env`, run `python3 scripts/setup_local.py` if the internal proxy key is absent, and configure `AUTH_TOKENS_JSON` for shared access. Keep backend/renderer ports local. Configure HTTPS and TURN for remote webcams.
4. Build and inspect every CUDA image. CUDA builds, optional WhisperX/Demucs dependencies, gated diarization models, and actual model imports remain hardware-host gates. The new audio-quality service has a fully resolved Linux x86_64/Python 3.11 dependency lock, including the pinned PyTorch family. Dependency resolution passed locally; its full CUDA environment has not been built locally.

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.avatar.yml --profile quality-audio config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.avatar.yml --profile quality-audio build
# Start only after reviewing GPU assignments and model access.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.avatar.yml --profile quality-audio up -d
```

5. Install/check model assets. For trained VAD, place the checksum-verified file from `scripts/fetch_vad.py` at `/models/silero/silero_vad.onnx` in the shared models volume, then set `AVATAR_VAD_BACKEND=silero`. Energy VAD remains the default until the model is present. Diarization requires the model provider's gated access and HF token file.
6. Create a model manifest using `scripts/model_manifest.py MODEL_DIRECTORY MANIFEST.json` and record its printed `MODEL_MANIFEST_SHA256` plus `RELEASE_ID` in `.env`. Store manifests outside the model tree. Capture `pip freeze`, image digests, driver and GPU inventory with benchmark results.
7. Run preflight. `API_TOKEN` may be supplied in the environment; it is never written to reports.

```sh
python3 scripts/gpu_acceptance.py
```

A passing preflight only confirms packages, model presence and service contracts. Run actual inference next.

## GPU acceptance matrix

Use the consented recordings described in `tests/evaluation/cases.json`. Keep cold-start runs separate from warm runs; warm the models before collecting the warm baseline. Do at least three warm repetitions per case, then increase repetitions for a meaningful p95.

```sh
python3 scripts/benchmark_modes.py artifacts/inputs/short-words.mp4 --target es --tag baseline --repeats 3
python3 scripts/benchmark_modes.py artifacts/inputs/short-words.mp4 --target es --tag windowed --repeats 3 --options '{"windowed_lipsync":true}'
python3 scripts/compare_benchmarks.py artifacts/bench/modes/results.jsonl --output artifacts/bench/comparison.json
```

| Test | Required evidence |
| --- | --- |
| Fast translation | Webcam input → complete translated file; no missing speech. Proposed target: warm p95 ≤180 s for 30 s input. |
| Quality batch | Blind comparison against fast mode, preserved identity, accurate meaning, stable mouth/face and clean audio. Quality takes precedence over runtime. |
| Windowing | Compare 30 s and 120 s clips; inspect every boundary, frame continuity, A/V sync, peak VRAM and restart reuse. Oversized inputs now use windows automatically; validate seams before production use. |
| Speech quality | Short/quiet words, numbers/names, fast speech, non-Latin text, long translations, speaker changes and overlapping speech. Rewrites require human review of meaning. |
| Optional audio | Verify alignment timing, diarization labels, selected voice per speaker, source-vocal removal and full-length background remix. |
| Avatar | Real portrait/mic, first audio, full-duplex interruption, heard-history accuracy, render timeout fallback, continuity across chunks and a 30-minute session. Proposed targets: first reply p95 ≤3 s and interruption p95 ≤300 ms. |
| Concurrent load | Batch + fast + avatar; capture queue time, starvation behavior, RAM/VRAM, OOM recovery and disk growth. Scheduling is nonpreemptive within a native inference call. |
| Optimizations | A/B one flag at a time: precision, compilation, process/thread sharding, replica count, restoration and encoder. Promote only when quality and failure tests pass. |
| Recovery | Cancel each stage, restart backend during work, crash a renderer, retry from checkpoint, corrupt one cached artifact, fill quota and interrupt network submission. |
| External network | Real desktop/mobile cameras, HTTPS, TURN relay-only test, NAT, permission denial, reconnect and packet loss. |

Score translation accuracy, voice identity, lip sync and flicker from 1–5; proposed median minimum 4. Mark missing speech separately; any missing speech fails acceptance. Benchmark reports deliberately leave quality scores blank until reviewed. Proposed latency thresholds are targets, not achieved results.

## Operational limits and rollback

- One translation dispatcher owns a jobs volume. SQLite provides local transactional state; this is not a distributed multi-host scheduler. A dedicated avatar backend uses `RECOVER_JOBS=false`.
- Account storage quotas are admission checks using SQLite byte totals. Totals refresh after uploads, revisions and stages, at completion, and every minute for active jobs in the background; generated artifacts can grow during an accepted job. Window limits and avatar storage caps bound individual operations. Watch free disk; retention is disabled unless explicitly configured.
- Graceful backend shutdown drains native calls, preserves queued jobs, and requeues interrupted work for checkpoint recovery. Compose allows ten minutes to drain. SIGKILL, host loss, or expiry of that grace period still marks interrupted work failed on startup for explicit retry.
- Cancellation waits for running native work. A timed-out remote renderer may retain files until aged cleanup, because it can still hold them. Avatar fallback prioritizes audio and keeps the portrait static for the affected chunk.
- Browser playback acknowledgments commit complete heard sentences. The unacknowledged tail of an interrupted sentence is omitted. Real browser A/V clock behavior still needs target testing.
- Alignment/diarization/separation and duration rewriting remain opt-in. Oversized clips automatically use render windows; small clips can opt in per job. Bundled voice names are supported by XTTS; other TTS backends need their own voice/reference controls.
- Head/gesture synthesis, distributed scheduling and live translated video before Stop are future extensions beyond these three implemented workflows. They are not represented as working features.
- New frontend CSS requires modern browsers (Safari 16.4+, Chrome 111+, Firefox 128+). FastAPI startup hooks still emit deprecation warnings; they pass lifecycle tests.
- To roll back: stop services, restore the backed-up source/images and jobs volume together, preserve secrets, then restart. Do not run old/new dispatchers on the same jobs volume. No destructive schema migration or remote deployment was performed by this update.

## Sources used for integration and upgrades

- [WhisperX](https://github.com/m-bain/whisperX), [Demucs](https://github.com/facebookresearch/demucs), [Silero VAD](https://github.com/snakers4/silero-vad).
- [Next 15 migration](https://nextjs.org/docs/app/guides/upgrading/version-15) and [Tailwind 4 migration](https://tailwindcss.com/docs/upgrade-guide).

## Review follow-up

See `review-remediation.md` for all ten fixes and their regression coverage. LatentSync defaults to process sharding. `LATENTSYNC_REALTIME_FACTOR=5.0` retains the conservative historical ETA until new measurements include windowing and speech validation. Automatic windows reserve 20% of the configured decoded-frame budget for rounding; this is not a bound on total model RAM or VRAM. XTTS cleanup requires complete high-confidence recognized text; ambiguous recognition keeps the audio intact, with bounded resynthesis before an overrun can fail. GPU speech tests must measure both false rejection and audible artifacts.
