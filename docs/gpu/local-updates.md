# Three-mode implementation and validation

## Current status

The local code now exposes three workflows:

| Workflow | Entry point | Pipeline | Current limit |
|---|---|---|---|
| Fast translation | `/live` or upload page, Fast | WebRTC recording → ASR → translation → timed TTS → MuseTalk | Processing begins after Stop; completion within minutes is a target, not a live dubbing guarantee |
| Quality translation | `/live` or upload page, Quality | Same capture → ASR → translation → timed TTS → LatentSync | Batch processing; compare real outputs before calling it the highest-quality configuration |
| Voice avatar | `/avatar` | Portrait + live microphone → utterance ASR → streamed LLM text (OpenAI-compatible, vLLM on the host) → streamed XTTS → prepared-image MuseTalk → WebRTC A/V | First implementation; target-GPU latency and visual quality are not yet validated |

The audio-only preset skips lip synchronization for minimum turnaround. Fast capture requests 640×360 at 25 fps; quality capture requests 1280×720 at 25 fps. Actual camera constraints may differ. Avatar output is at most 512 pixels on its longest side and 25 fps. It animates the mouth on a still portrait; it does not synthesize natural head gestures.

The earlier ~52-second synthetic-loop benchmarks (about 121 seconds for MuseTalk and 271 seconds for LatentSync end to end) describe previous code and settings. They are not measurements of this update. Existing historical experiment documents remain useful provenance.

## Start on the GPU host

```sh
python3 scripts/setup_local.py
# Check .env GPU assignments against nvidia-smi before starting.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Use the existing model-download instructions in `docs/gpu/README.md`. For a new `.env`, the setup script selects GPU defaults (Whisper large-v3 FP16, NLLB 3.3B and a 120-second limit). For an existing `.env`, it preserves model/GPU values. It generates an internal proxy key if absent, and changes legacy browser localhost URLs to same-origin routes. It never prints the key. The frontend, backend and ingest containers must receive the same configuration.

For a browser on the server, open `http://localhost:3030`. For a remote browser, use an SSH tunnel to port 3030 or configure HTTPS. Camera and microphone access require a secure browser context. Signaling travels through the frontend's `/api` and `/ingest` proxies; remote clients no longer call their own localhost by mistake.

For HTTPS, set `PUBLIC_HOST`, `DEMO_USER` and `DEMO_PASSWORD_HASH` in `.env`, point DNS at the host, and enable the `web` profile. Generate a bcrypt hash interactively with `docker run --rm -it caddy:2.10.2-alpine caddy hash-password`; quote its value in `.env`. Then:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile web up -d
```

Only the optional Caddy entrypoint should be exposed publicly. API, frontend and lipsync HTTP ports bind to loopback; ingest signaling requires the internal proxy key. WebRTC media still needs reachable ICE candidates. Configure `ICE_SERVERS_JSON` with your STUN/TURN servers and test across the actual firewall. Example schema:

```json
[{"urls":["stun:stun.example.com:3478"]},{"urls":["turn:turn.example.com:3478"],"username":"demo","credential":"your-turn-credential"}]
```

The same ICE settings reach both peers. Host networking is designed for the Linux GPU host. `INGEST_PUBLIC_IP` is no longer used; setting a hostname alone does not configure ICE or TURN.

## Avatar setup and capacity

The chat server must be reachable at `LLM_BASE_URL` (OpenAI-compatible `/v1`; the XE7740 runs vLLM with `Qwen/Qwen3-30B-A3B-Instruct-2507`, which is the default `LLM_MODEL`). XTTS uses a bundled speaker, optionally selected through `AVATAR_SPEAKER`; the portrait does not supply a voice clone. Required XTTS and MuseTalk weights must already be cached. Start with one avatar session, a front-facing portrait and a headset. Tune `AVATAR_VAD_RMS` on the real microphone; the initial energy detector has a 600 ms silence endpoint and is not a trained voice activity detector.

The default stack shares speech and MuseTalk services. Their inference locks prevent corruption but can make avatar replies wait behind translation. Use the optional dedicated-capacity overlay for simultaneous workloads:

```sh
# Exclude these cards from LATENTSYNC_GPUS and other workloads.
# .env: AVATAR_BACKEND_GPU=5, AVATAR_MUSETALK_GPU=6
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.avatar.yml up -d --build
```

This adds a speech backend on port 8092 and an avatar MuseTalk service on 8093. The avatar backend does not recover translation jobs from the shared volume. The LLM server needs capacity of its own; the overlay does not place or launch it. A dedicated speech backend defaults to Whisper small to reduce response latency; set `AVATAR_WHISPER_MODEL` to change it.

Avatar sessions expire after 30 minutes. Incoming speech or the Interrupt button clears queued audio and video together. Native GPU work already in flight drains safely. Cancelled turn artifacts are retained until the 24-hour orphan cleanup rather than deleted underneath inference. Normal completed chunks are removed after loading into the bounded playback queue.

## Reliability and quality changes

- GPU services serialize model/scheduler use. LatentSync requests get isolated temporary directories; workers tag commands/results with job IDs and discard stale results.
- Worker queues are bounded; completed chunks move to CPU and conditioning is released after collection. Source clips are still loaded in memory. Frame budgets reject excessive inputs; this is not a streaming implementation for arbitrary-length videos.
- Fast and batch jobs have separate lanes. Shared speech inference is serialized. Keep one backend API worker; replicas would require an external queue and ownership leases.
- Job metadata writes are atomic. Queued jobs recover after restart. Graceful shutdown drains native calls and requeues interrupted work; hard-crash interruptions are marked failed for an explicit checkpoint retry. Terminal in-memory state is released. SSE subscribers have independent cursors and receive durable snapshots.
- Cancellation waits for active native inference before releasing capacity. A queued cancellation exits without waiting for that job's own inference.
- WebRTC verifies both incoming tracks. Recording limits, session cleanup and retryable, idempotent submission prevent lost recordings and duplicate jobs. Limits default to 120 seconds, 100 MB and four simultaneous recordings.
- Every nonempty translated utterance is synthesized, including short replies. Failures retry and then fail visibly. Speech is aligned by original utterance onset. Small overruns can accelerate up to `TTS_MAX_SPEED` (default 1.15); larger overruns fail explicitly rather than silently dropping speech. Such failures need a shorter translation or a better duration-aware voice; automatic rewriting is not yet implemented.
- NLLB mutable tokenizer use is locked. The GPU compose file defaults both lanes to `TRANSLATE_BACKEND=llm` (contextual, glossary-aware translation through `LLM_BASE_URL`); set `nllb` to compare fidelity per target language.
- MuseTalk runs inference without gradients; CUDA Whisper weights and features use matching dtypes. FP16 remains opt-in pending parity checks. LatentSync defaults to process sharding; compilation remains opt-in pending GPU validation.
- The frontend is patched to Next.js 15.5.27, React 19.3.0, Tailwind 4.3.3 and PostCSS 8.5.28. Key CUDA framework packages are pinned with pip constraints so later installs cannot silently replace the chosen versions. The full ML dependency graph and model revisions are not yet completely locked; capture built-image digests and model hashes when validating a release.

## Validate before claiming speed or quality improvements

```sh
# Install httpx in the host's benchmark environment, then use representative speech.
python3 scripts/benchmark_modes.py artifacts/inputs/real-speaker.mp4 \
  --target es --repeats 3 --tag baseline \
  --config MUSETALK_DTYPE=fp32 --config LATENTSYNC_SHARD_MODE=process
```

The harness stores the commit, tracked diff hash, untracked file list, input hash, GPU/driver information, explicit experiment settings, stage times, output metadata and downloaded artifacts. Quality fields are intentionally blank for human review. It does not certify quality automatically.

Compare cold/warm runs, 15/60/120-second real clips, multiple languages, glasses/facial hair, head turns and interruptions. Change one variable at a time: MuseTalk FP16 vs FP32; LatentSync thread vs process; one vs several GPUs; compile off vs on. Inspect translation accuracy, omitted speech, voice identity, lip alignment, face texture and temporal flicker. Measure queue time and total time separately; avatar tests need end-of-speech to first audible/visible reply and interruption latency under simultaneous batch load.

CPU validation covers submission retries, cancellation drain, SSE fanout, short speech, segment timing, worker result identity, avatar playback/backpressure and a real local WebRTC audio/video recording. TypeScript and a frontend production build pass locally. GPU inference and real-camera/NAT operation require target-host validation. One existing watermark test requires an ffmpeg build with `drawtext`; the review machine's Homebrew ffmpeg lacks it, while the container installs the required libraries.

## Remaining engineering work

1. Run the GPU and human quality matrix; select language-specific models from evidence.
2. Validate automatic bounded render windows on the GPU, including window seams, A/V timing and host RAM use, before increasing recording duration or resolution limits.
3. Add duration-aware translation/TTS regeneration, forced alignment and speaker separation for multi-speaker content.
4. Optimize the avatar's audio feature windows and overlap chunk boundaries; evaluate richer portrait animation if natural head motion is required.
5. Replace energy VAD with a measured speech detector and add browser tests across TURN, disconnects and camera changes.
6. Lock the full dependency/model graph from validated images and add a GPU CI runner. Current CPU CI cannot establish numerical parity or throughput.

Upstream references: [Next.js security patch](https://nextjs.org/blog/security-update-2025-12-11), [XTTS streaming API](https://docs.coqui.ai/en/latest/models/xtts.html), [OpenAI-compatible chat completions (vLLM)](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html), [aiortc media helpers](https://aiortc.readthedocs.io/en/latest/helpers.html).

## Reviewer remediation

The ten review findings are addressed in `review-remediation.md`. GPU quality and performance acceptance remains pending. The current dependency versions and validation commands are in `gpu-testing-handoff.md`.
