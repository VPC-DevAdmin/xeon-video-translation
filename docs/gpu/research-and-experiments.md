# GPU quality and performance review — 3 October 2026

> Hardware update: the project has now been deployed and tested on XE7740. See the [measured results and remaining acceptance gates](xe7740-validation-2026-10-03.md). Earlier local-only status below describes the pre-deployment snapshot.

## Recommendation

Prioritize the video data path and avatar scheduling, then compare new models in isolated environments. Keep three separate winners: fast translation, faithful high-quality dubbing, and a responsive portrait avatar. One model is unlikely to win all three.

The project has useful engineering foundations: bounded work, cancellation fences, explicit GPU readiness, reusable portrait state, service boundaries and local regression coverage. It is still a GPU qualification candidate. The available evidence does not establish production quality, sustained concurrency, or the fastest configuration on the eight-card host. CPU tests and successful image builds cannot establish those properties.

**Status of this review:** source inspected; primary-source research completed; 24 experiments cataloged; measurement tooling improved; local tests completed. New model adapters and actual GPU trials remain outstanding. The new NVDEC/DLPack smoke probe is implemented but hardware-unqualified. No serving model or GPU dependency was replaced by this research update.

## 1. What the measurements actually support

The recorded fixture is 52.067 seconds long and combines looped webcam video with synthesized narration. It is useful for repeatable throughput checks but weak evidence for natural conversation, identity retention or difficult speech. These are individual historical observations, not current p95 measurements:

| Historical run | Total | Lip rendering | Share spent rendering | ASR |
| --- | ---: | ---: | ---: | ---: |
| MuseTalk opt2, warm | 120.600 s | 95.858 s | 79.5% | 1.892 s |
| LatentSync opt3, four GPUs, warm | 271.183 s | 247.163 s | 91.1% | 1.879 s |

Source: `artifacts/bench/xe7740-2026-10-03.jsonl`; context and visual observations: `docs/gpu/bring-up-2026-10.md`. Rebaseline the current uncommitted source before comparing. Four-GPU historical results do not describe the current shared-host two-GPU allocation.

**Arithmetic projection, not a promised speedup:** halving only rendering would reduce these totals to approximately 72.67 and 147.60 seconds. Halving ASR would save less than one second. Rendering and transfers therefore deserve the first performance work. Speech replacements are more interesting for accuracy, voice quality and interactive latency.

The bring-up notes also record lower-face smoothing/stubble loss with MuseTalk, plastic-looking restoration with CodeFormer, and better local facial detail with LatentSync. Those observations merit a real evaluation corpus; a few sampled frames do not establish temporal quality. An optional CPU stabilization run took about 170 seconds for the 52-second clip without a clear sampled-frame benefit. Leave it outside the fast baseline until it earns its cost.

## 2. Remaining code and architecture opportunities

| Priority | Finding in current source | Experiment and expected benefit |
| --- | --- | --- |
| 1 | Hardware codecs still download frames into host memory. `shared/gpu_runtime/media.py` includes host conversion/filtering. | Decode into CUDA surfaces, import with DLPack, and perform crop/resize/blending on device. Measure transfer bytes as well as FPS. |
| 1 | `backend/app/pipeline/windowed.py` and final watermark processing can produce several decode/encode passes. | Reuse bounded frame windows, compose once, encode final frames once. This can improve both detail and speed. Preserve timestamps, resume and disclosure. |
| 1 | MuseTalk translation still has CPU NumPy decode/composite work; LatentSync worker boundaries return substantial frame tensors through host memory. | Extend the GPU tensor path to translation. Compare same-device processing, measured peer transfers and independent whole-job workers. |
| 1 | Avatar text waits for punctuation/length; audio collects roughly 500 ms and retains one chunk to determine the final chunk. The ingest path waits for rendering before queuing playback. | Compare 160/240/320/500 ms chunks, prompt first-chunk delivery, an explicit end marker and bounded A/V buffering. Preserve heard-speech acknowledgments and interruption behavior. Shorter chunks can hurt prosody and GPU efficiency. |
| 2 | Segment TTS passes reference audio for each synthesis call. | Instrument actual conditioning cost first, then cache by reference content, model revision and dtype if upstream repeats the work. Do not infer recomputation solely from the API call. |
| 2 | MuseTalk samples VAE latents per frame. | Compare seeded sampling with posterior mean and stable masks. Mean latents alter the model input distribution and may reduce quality; this is not a safe default change. |
| 2 | Fixed encoder settings and earlier lossy intermediates constrain quality mode. | Test NVENC p4/p6/p7, CQ16/18/20 and fewer transcodes. Evaluate teeth/stubble/mouth detail; whole-frame similarity can conceal local damage. |
| 2 | Prior benchmark aggregation could pool different languages, fixtures, allocations and cache states, and reported p95 from three runs. | Implemented: separate these cohorts, include source fingerprints, show unsuccessful runs, and suppress p95 below 20 completed samples. Labels still require operator verification. |

The avatar code references are `backend/app/api/avatar.py`, `services/ingest-webrtc/app/avatar.py` and `services/musetalk/app/avatar.py`. Rendering references are `services/musetalk/app/musetalk/inference.py`, its VAE module, and the LatentSync inference/worker implementation.

“GPU first” should mean moving expensive inference and repeated image work onto GPUs. Signaling, small control buffers and inexpensive VAD can reasonably remain on CPU when measurement shows that transfers and GPU launch overhead cost more. Strict GPU configuration prevents some silent fallbacks; it does not prove every operation is GPU-resident or fast.

## 3. Hardware compatibility: qualify SM120 explicitly

The documented target is eight RTX PRO 6000 Blackwell Server Edition cards with 96 GB each. NVIDIA lists this family as compute capability **12.0**. It is a different kernel target from B200/SM100 and Hopper/SM90. Published “Blackwell” or H100 results alone are insufficient compatibility evidence. Each 96 GB card is a separate memory domain; inspect PCIe topology, NUMA placement and peer access before choosing sharding. [NVIDIA CUDA GPU list](https://developer.nvidia.com/cuda/gpus), [hardware specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).

The recorded host has driver 580.178.04 and reports CUDA 13.0. A driver's reported CUDA capability does not identify the CUDA libraries in each container. Use separate, pinned candidate environments, record the actual PyTorch/CUDA/kernel versions, and run a real operator test. Do not replace the production stack globally to satisfy one experimental model.

Before trials, capture GPU UUIDs, compute capability, driver, `nvidia-smi topo -m`, active processes, power limits, NUMA placement and memory headroom. Historical notes place unrelated vLLM services on cards 1–2; confirm current occupancy. Use explicitly allocated UUIDs and leave unrelated services alone. A five-GPU experiment should run in a designated test window rather than compete with interactive service.

## 4. Model shortlist

All upstream speed claims below are authors' results, **not measurements on this server**. “Plausible fit” means worth attempting; it does not mean qualified or guaranteed to fit at every resolution/batch size.

| Candidate | Best role | Why test it | Important limit |
| --- | --- | --- | --- |
| SoulX-FlashHead Lite / Pro | Live portrait avatar | Released 1.3B streaming models. Authors report Lite at 96 FPS on one 4090 and Pro at 25+ FPS on two 5090s. Start Lite on one card; compare Pro on two. | Verify SM120 kernels, chunk continuity, identity and first-frame latency. Pro's published real-time setup uses SageAttention. [Source](https://github.com/Soul-AILab/SoulX-FlashHead) |
| VoxCPM2 | Multilingual voice; quality and speed | 2B, 30 languages, 48 kHz output, cloning and streaming. Authors report RTF about 0.3 on 4090, or 0.13 with an accelerated serving backend. Apache-2.0 code/weights are advertised. | Preserve native sample rate through explicit resampling boundaries. Includes Hindi, unlike several alternatives. Test serving support, pronunciation and voice identity. [Source](https://github.com/OpenBMB/VoxCPM) |
| Qwen3-TTS 0.6B / 1.7B Base | Streaming cloned voice | Compare smaller and larger models for the quality/latency tradeoff. | Base, CustomVoice and VoiceDesign are distinct checkpoints. Ten listed languages exclude Hindi. Published first-packet latency is not assistant response latency. [Source](https://github.com/QwenLM/Qwen3-TTS) |
| CosyVoice3 | Responsive voice assistant | Bidirectional text/audio streaming can reduce waiting for complete sentences. | Nine listed languages exclude Hindi. The repository's TensorRT-LLM speed claim for CosyVoice2 does not establish the same gain for v3. [Source](https://github.com/QwenAudio/CosyVoice) |
| IndexTTS 2.5 | Controlled dubbing | Newer release with rate and pronunciation controls; useful when translated speech must fit a time window. | `duration_factor` controls speaking rate, not exact output seconds. Listed languages are Chinese, English, Japanese, Spanish and Arabic. Custom model-use license. [Source](https://github.com/index-tts/index-tts) |
| Qwen3-ASR | Difficult multilingual input | Test recognition accuracy on noise, accents, names and short utterances. | Streaming is a vLLM path without timestamps/batching; the separate forced aligner's language coverage differs and excludes Hindi. [Source](https://github.com/QwenLM/Qwen3-ASR) |
| Voxtral Mini Realtime 4B | Streaming ASR and turn-taking | Causal transcription with configurable delay. | Different objective from the already-fast offline ASR stage; assess endpoint-to-response latency and final transcript accuracy. [Model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) |
| TranslateGemma 12B / 27B | Translation quality | Specialized translation models covering 55 languages; compare against current NLLB/Qwen for meaning, numbers and concise phrasing. | Requires its translation template and Gemma access/license terms. Measure full weights, KV cache and runtime headroom rather than estimating fit from parameter count alone. [Model card](https://huggingface.co/google/translategemma-27b-it), [announcement](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) |

**Existing controls:** the project already downloads MuseTalk 1.5 and LatentSync 1.6. Calling those versions an upgrade would be misleading. Optimize and retain them as controls while testing replacements. [MuseTalk](https://github.com/TMElyralab/MuseTalk), [LatentSync](https://github.com/bytedance/LatentSync).

### Higher-risk trials worth keeping

- **InfiniteTalk with supported distilled checkpoints:** try for animated avatars and video-to-video quality. Its repository documents identity/color drift risks with FusionX, and older installation instructions need adaptation for this hardware. Model-specific LightX2V acceleration is not a drop-in LatentSync optimization. [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk), [LightX2V](https://github.com/ModelTC/LightX2V).
- **LiveAvatar 14B:** ambitious long-running avatar quality. The published TPP setup requires five GPUs with at least 80 GB each; its multi-H800 performance does not prove RTX/PCIe performance. Separate a one-card functionality trial from a five-card throughput trial. [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar).
- **Fish Audio S2 Pro:** expressive multilingual speech candidate. Its Research License is a deployment constraint; evaluation success alone does not authorize production use. [Fish Speech](https://github.com/fishaudio/fish-speech), [technical report](https://arxiv.org/abs/2603.08823).

Full-body or head-motion generators can look attractive while reducing faithful reproduction of a webcam recording. Score that separately from image-to-avatar expressiveness. Do not choose a dubbing renderer solely from a promotional avatar demo.

## 5. Acceleration tools worth trying

1. **PyNvVideoCodec + CUDA tensor operations / CV-CUDA.** First choice for removing host round trips. NVIDIA documents GPU decode surfaces and DLPack tensor import; CV-CUDA supplies batched image operations. The new smoke probe checks the initial decode/import/resize chain. Production integration still needs timestamp, color, rotation, ownership and encoder tests. [NVIDIA API guide](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html), [CV-CUDA](https://cvcuda.github.io/CV-CUDA/).
2. **torch.compile and CUDA graphs for stable shapes.** Test bounded shape/batch buckets around existing models. Record compile time separately, then assess steady-state latency and concurrent throughput. Dynamic shapes, graph breaks and memory duplication can erase gains.
3. **FlashAttention 4 CuTe implementation.** Current source has an SM120 dispatch path, so it deserves a compatibility trial. Check actual head sizes, dtypes and model integration; FA3's Hopper optimization is not equivalent support. [Repository](https://github.com/Dao-AILab/flash-attention), [dispatch implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py).
4. **SageAttention2, then experimental SageAttention3/FP4.** Quantized attention can accelerate video models, but lip/identity/flicker quality must pass. Upstream recommends Sage2 for precision-sensitive use. A kernel speedup is not an end-to-end speedup. [Repository](https://github.com/thu-ml/SageAttention).
5. **TensorRT / TensorRT-LLM selectively.** Consider stable detector/encoder graphs and supported language/speech models after profiling. Support varies by model, precision and hardware; do not treat an SM120 support-table entry as proof every FP4 kernel works. [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html), [TensorRT-LLM matrix](https://nvidia.github.io/TensorRT-LLM/1.2.0rc1/legacy/reference/support-matrix.html).

CPU-offloaded large models are useful as compatibility experiments but weak candidates for latency-critical serving. Prefer an accurately measured smaller resident model when offload dominates.

## 6. Evaluation protocol and proposed acceptance gates

These thresholds are initial engineering targets, not observed capabilities or universal perceptual standards. Refine them after the first measured baseline.

**Corpus:** collect 24–40 consented, real webcam clips, 10–90 seconds each, plus a five-minute soak fixture. Start with English↔Spanish and English→Japanese/Hindi/French. Include silence, two-word utterances, numbers, names, negations, code-switching, fast speech, noise, head turns, glasses, facial hair, occlusion, low light, portrait/landscape and variable frame rate. Keep real source recordings; the looped synthetic fixture remains a separate performance control.

**Compare fairly:** pin source and weight revisions, precision, codec settings, prompts, seeds, resolution and face crop. Preserve input/output artifacts and all failures. Use separate cold, warm-model and warm-artifact groups. Randomize A/B order; measure isolated and mixed interactive/batch load. A requested config label is not proof it reached the running service—save observed service configuration and container/model manifests too.

**Measurements:** completion latency, queue time, stage timings, first heard audio, first displayed moving frame, continuous generated FPS, underruns, interruption latency, A/V skew, per-process VRAM, CPU/RAM, transfer bytes and GPU-hours per completed output. Use CUDA events for GPU timing and representative Nsight traces for copies/serialization. Do not add instrumentation that synchronizes every frame to the production hot path.

**Quality:** blind paired listening/viewing, reviewed by speakers of the target language. Score meaning/omissions, numbers/names/negation, intelligibility, voice identity, timing, lip motion, teeth/skin detail and temporal stability. Independent ASR can catch spoken-text errors but does not replace listening or semantic evaluation. Avoid using the same recognizer both to generate and judge transcripts. Record baseline/candidate preference and ties per clip.

**Promotion targets:**

- Fast translation: aim for at least 20% lower median completion time with unchanged semantic accuracy and no missing utterances. On the standard 60-second fixture, first aim for completion within two minutes after Stop, then stretch toward one minute. Any accepted visual tradeoff must be explicitly scored.
- Batch quality: a clear blind preference on held-out clips, no new material omissions/identity defects, and acceptable GPU-hours per output. Equal-quality slower candidates do not win simply because they are larger.
- Avatar: initial target p95 end-of-user-speech to heard response ≤1.5 seconds, first moving frame ≤1.8 seconds, sustained 25 FPS where video is active, A/V skew ≤80 ms and interruption ≤250 ms. Measure over WebRTC, including TURN and simultaneous batch load. These are end-to-end targets, not model first-token claims.
- Both quality and speed: seek ≥20% lower median latency plus a positive blind preference. Use paired intervals and failure rates; do not promote on a handful of favorable clips.
- Start with 3–5 repetitions for screening, then at least 30 paired runs across the corpus. Use 100+ representative sessions/jobs for a meaningful p95 qualification report, with sample count and uncertainty. The comparator's 20-sample threshold is only a guard against obviously misleading tiny-sample p95 claims.

## 7. Ordered todo list

- [x] Audit current source and historical evidence; identify remaining CPU image work and avoid claiming unmeasured GPU speedups.
- [x] Verify primary model/tool sources and build the 24-entry catalog with explicit rejection criteria.
- [x] Add a runner that records command failures/timeouts, keeps logs and continues independent trials.
- [x] Improve benchmark cohort separation, source fingerprints and small-sample reporting.
- [x] Implement a bounded NVDEC → DLPack → CUDA resize smoke probe and test local failure behavior.
- [ ] Wave 0: inventory host, record actual model/container manifests, collect real clips, reproduce current controls and quality review.
- [ ] Wave 1: trace rendering, prototype GPU-resident frames and one final encode, tune avatar chunk scheduling and conditioning cache; compare replicas with sharding.
- [ ] Wave 2: implement isolated service adapters for FlashHead, shortlisted TTS, ASR and TranslateGemma. Test one model at a time before combining winners.
- [ ] Wave 3: attempt quantized attention, distilled InfiniteTalk, LiveAvatar and Fish S2 Pro where earlier results justify their cost.
- [ ] Final: validate the best per-mode combinations under concurrent load, interruption, long sessions and browser/network variation; publish results including rejected candidates.

An installation failure, unsupported kernel, language gap, OOM, quality loss or slowdown is a valid recorded result. Limit each first compatibility attempt to a defined time budget. Repair tractable integration issues once; park candidates that require broad upstream rewrites unless their likely benefit justifies it. Avoid stacking several speculative optimizations before learning which one helped.

## 8. What is implemented locally

- `experiments/gpu-candidates.json`: 24 ranked experiments; explicit integration readiness and rejection criteria.
- `scripts/experiment_lab.py`: plan/execute modes, explicit GPU UUID normalization, minimum GPU counts, isolated local process groups, timeout/cancellation cleanup and retained logs/results. A successful exit is labeled completed, never quality-approved.
- `scripts/probe_gpu_frames.py`: hardware smoke probe only; bounded batches and fail-closed CUDA checks. PyTorch peak memory excludes decoder/external allocations and is labeled accordingly.
- `scripts/benchmark_modes.py`: language/cache/hardware labels, observed device details and hashes of untracked source files in addition to tracked changes.
- `scripts/compare_benchmarks.py`: prevents incompatible cohorts and source revisions from being pooled; reports failed runs, missing quality review and insufficient p95 samples.
- `tests/research`: regression coverage wired into CPU CI; existing tiny-sample p95 test updated to match the safer reporting contract.

See `experiments/README.md` for commands, artifact contracts and remaining adapter work. Final source-tree verification is recorded below. No new model's GPU performance or perceptual quality has been established by these CPU tests.

### Local verification results

| Check | Result | Scope |
| --- | --- | --- |
| Shared/backend/worker plus experiment tests | 128 passed | Includes 24 new research-tool tests |
| WebRTC/avatar tests | 21 passed | Real local peer connections; CPU test codec |
| Tensor/media tests | 13 passed | CPU PyTorch and real FFmpeg fixtures |
| Static checks | Passed | Ruff correctness checks and `git diff --check` |
| Experiment catalog | 24 entries validated | Plan mode works without GPU libraries |
| GPU frame probe on local host | Blocked: CUDA unavailable | Correct fail-closed result; no hardware claim |

**Total: 162 passing local tests.** The first backend run used the system FFmpeg, which lacks `drawtext`, and failed two watermark-dependent tests. Re-running with the existing FFmpeg 7.1 test binary passed all 128 tests. No application behavior was weakened to make those tests pass. Existing FastAPI lifecycle/test-client deprecation warnings remain.

The source update is local and uncommitted. It preserves the preceding GPU-path changes. CUDA codec execution, candidate model installation/inference, SM120 kernel compatibility, perceptual quality and eight-GPU load behavior remain hardware-side work.
