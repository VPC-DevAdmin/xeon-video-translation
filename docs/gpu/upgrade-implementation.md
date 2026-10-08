# Eight-GPU update: implementation and qualification

> Hardware update: the project has now been deployed and tested on XE7740. See the [measured results and remaining acceptance gates](xe7740-validation-2026-10-03.md). Earlier local-only status below describes the pre-deployment snapshot.

Date: 3 October 2026. Based on checkout `025b01e`.

## Implemented locally

- Explicit CUDA requests fail if CUDA is unavailable. Loaded InsightFace sessions must actually select CUDA; whole-model ONNX fallback is disabled. Small ONNX control/shape operations can still execute on CPU.
- GPU video encoding failures fail the job instead of permanently switching the process to libx264. CPU deployments still select software codecs explicitly.
- Readiness executes FP16 matrix operations on every visible GPU and a real NVENC/NVDEC round trip. Results include GPU identity, VRAM, capability and runtime versions, with a 60-second cache. These checks do not certify model placement or quality.
- MuseTalk and LatentSync use a bounded NVDEC reader. LatentSync no longer encodes an intermediate MP4 just to normalize its frame rate. Window cuts and filtered final muxing explicitly request CUDA decoding. Unsupported CUDA formats fail rather than silently using software decoding. Current supported input surfaces are 8/10-bit 4:2:0; unusual pixel formats need an explicit conversion policy.
- The decoder enforces retained-frame limits, handles rotation metadata, preserves RGB/BGR order, rejects truncated output, and terminates timed-out/oversized subprocesses. RGB conversion, frame-rate selection and rotation still run on the host.
- Avatar VAE output stays on the GPU through resize, feathering, composition and the disclosure badge. Portrait/mask/badge tensors are cached. The existing NumPy transport still requires a final host transfer. `AVATAR_GPU_COMPOSITE=0` provides an explicit comparison path.
- WebRTC uses NVENC for H.264 transmission/recording and NVDEC for incoming H.264 in the GPU overlay. Startup exercises the actual PyAV codec libraries. H.264-only negotiation prevents a VP8 software fallback. CPU mode retains the portable codecs. The adapter is fenced to aiortc 1.15.0 / PyAV 17.1.0 because recorder codec selection uses a private upstream interface.
- LLM requests have bounded concurrency and queue depth per backend, queue deadlines, input/output limits and bounded response reads. Streaming cancellation releases its admission slot. Server-wide KV-cache and scheduling limits remain the external vLLM operator's responsibility.
- Structured execution logs cover media decode/encode, renderer requests, avatar compositing and LLM queue/inference. `GPU_PROFILE=1` enables current-stream CUDA timings for avatar compositing; it synchronizes the stream and is a diagnostic setting, not a latency benchmark setting.
- Shared/dedicated allocation generation rejects duplicate assignments and optionally maps indices to UUIDs. With `--quality-audio`, GPU 7 is removed from the batch pool entirely. This conservative static allocation avoids cross-service contention without claiming a dynamic lease scheduler.

## Bring-up commands

Run from the project root. Preserve the existing `.env` and secrets.

```sh
# Existing vLLM on GPUs 1–2; select --profile dedicated only on a dedicated host.
python3 scripts/gpu_layout.py --profile shared --output gpu-layout.env
# For optional audio quality, generate with --quality-audio instead.
# Add --inventory-json inventory.json to map indices to stable GPU UUIDs.
# Inventory format: [{"index": 0, "uuid": "GPU-..."}, ... eight entries ...].

docker compose --env-file .env --env-file gpu-layout.env \
  -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.avatar.yml config

docker compose --env-file .env --env-file gpu-layout.env \
  -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.avatar.yml up -d --build

# Requires API_TOKEN / INTERNAL_API_KEY when those routes require authentication.
python3 scripts/gpu_acceptance.py --output artifacts/bench/gpu-preflight.json
```

The layout generator refuses to overwrite existing files. Review live GPU processes before starting services. It does not reconfigure the external LLM. WebRTC codec engines intentionally share the avatar-renderer GPU; `INGEST_GPU` can override this only after reviewing the assignment.

Shared-host defaults are speech 0, external LLM 1–2, fast rendering 3, batch 4/7, avatar speech 5 and avatar rendering/media codecs 6. Existing `.env` assignments take precedence. The avatar overlay is needed to start its dedicated services. Do not enable the `quality-audio` Compose profile unless the layout reserves GPU 7 for it.

## Build and development changes

The three inference images install the local `shared/` runtime through the `gpu_shared` named build context. Compose config supplies it automatically. A direct Docker build now needs:

```sh
docker build --build-context gpu_shared=./shared -f backend/Dockerfile.gpu backend
# For a local backend checkout:
python -m pip install -e ./shared
```

CI runs the shared/backend/worker tests, portable WebRTC tests, and a CPU tensor/media suite. Tensor math tests do not verify CUDA kernel support or speed.

## Qualification status

Local evidence:

- 104 shared/backend/worker tests passed.
- 21 WebRTC/avatar tests passed, including real local peer connections and codec packetization with a software test encoder.
- 13 tensor/media tests passed using actual CPU PyTorch and FFmpeg. Rotation, resampling, buffer ownership, overflow cleanup and compositing were exercised. Compositing matches the existing CPU reference within three intensity levels in the tested cases.
- WebRTC image builds passed for the local architecture and Linux x86-64. The x86-64 PyAV image exposes both `h264_nvenc` and `h264_cuvid`; execution requires the NVIDIA host.
- Shared-runtime installation through Docker's named context passed. Combined GPU/avatar Compose configuration resolves successfully.

GPU model images, CUDA inference, codec execution, eight-card contention, visual quality and latency are **not qualified** on this local machine. Run the runtime probes first, then model fixtures, actual browser sessions, interruption tests and mixed-load/soak benchmarks. Compare avatar compositing enabled/disabled, one versus multiple LatentSync workers, and cold versus warm timings. Capture end-to-end p50/p95, VRAM peaks, dropped frames and audiovisual skew. Do not accept an FPS-only improvement that loses speech, worsens identity or increases visible jitter.

## Remaining work in the larger plan

The full upgrade plan remains open. This update supplies concrete GPU paths and checks; it does not complete every phase.

1. Replace host-frame boundaries with a GPU-resident decoder → crop/landmark/warp → renderer → composite → encoder path. Translation compositing, legacy CPU filters and some preprocessing still use host memory.
2. Replace avatar WAV/NumPy artifact exchange with bounded streaming buffers, explicit ownership and interruption fencing across the media worker.
3. Add comprehensive per-job transfer/memory/queue traces, persistent benchmark comparisons and quality scoring. Existing execution spans are a starting point.
4. Add measured model-specific conditioning/verification caches and evaluate compilation/TensorRT only after reference outputs pass quality gates.
5. Add dynamic GPU borrowing/leases only if static profiles leave useful capacity unused. The current generator prevents allocation conflicts; it is not a runtime allocator.
6. Qualify all model images and CUDA/ONNX kernels on the actual GPU architecture, then complete the mixed-load and browser acceptance matrix.
