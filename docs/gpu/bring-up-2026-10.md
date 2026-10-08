# XE7740 bring-up, 2026-10-02/03

First run of the GPU track on the target hardware. Everything below was
measured, not projected. Raw per-job numbers are in
[`artifacts/bench/xe7740-2026-10-03.jsonl`](../../artifacts/bench/xe7740-2026-10-03.jsonl)
(produced by `scripts/gpu_bench.sh`).

## Setup that worked

- Dell XE7740, 8x RTX PRO 6000 Blackwell Server Edition (96 GB), driver
  580.178.04 / CUDA 13.0, Docker 29 with the NVIDIA runtime.
- **The box is shared.** vLLM containers hold ~90 GB on GPUs 1 and 2. The
  stack runs on GPUs 0 (backend), 3 (MuseTalk) and 4 (LatentSync) via
  `BACKEND_GPU` / `MUSETALK_GPU` / `LATENTSYNC_GPU` in `.env`.
- Hugging Face anonymously rate-limits the box's egress IP (429 on every
  call). `HF_TOKEN_FILE` in `.env` mounts the host's token into the
  containers; see [README.md](README.md#hugging-face-access).
- `.env` on the box: `WHISPER_MODEL=large-v3`, `WHISPER_COMPUTE_TYPE=float16`,
  `NLLB_MODEL=facebook/nllb-200-3.3B`, `MAX_CONCURRENT_JOBS=1`.

## Test fixture

The repo's only clip is 1.8 s ("Good morning"). A 17x loop of it collapses
to one whisper segment, so it is useless for timing TTS or lipsync.
`clip_speech.mov` (52 s, on the box under `artifacts/inputs/`, not
committed) is the fixture video looped under 51 s of continuous English
narration synthesised with XTTS in the speaker's own voice. Whisper
transcribes it as 9 accurate segments.

## Performance, 52 s clip, warm models

| Stage | CPU build (docs) | GPU measured | Notes |
|---|---|---|---|
| transcribe (whisper) | base int8, ~6x RT | **2.0 s** (large-v3 fp16, 26x RT) | |
| translate (NLLB) | 600M, ~3 s/segment | **2.5 s** (3.3B fp16, 9 segments) | |
| tts XTTS-v2 | ~0.5x RT | **14 s** (3.6x RT) | per-segment path |
| tts F5-TTS (en) | ~0.6x RT | **21 s** | after the two fixes below |
| tts IndicF5 (hi) | never ran | **14 s** | first successful end-to-end run |
| lipsync Wav2Lip | ~15 s / src-s | **76 s** (1.5 s / src-s) | |
| lipsync MuseTalk, fast | ~200 s / src-s | **168 s** (3.2 s / src-s) | after batching; was 630 s |
| lipsync MuseTalk + CodeFormer | — | 978 s | before batching; see quality |
| lipsync LatentSync 512 fp16 20 steps | ~5400 s / src-s | **422 s** (8.1 s / src-s) | one GPU |
| post-stabilise (vidstab) | ~2x src | 170 s | CPU ffmpeg, two-pass |
| mux + watermark | <1 s / 10 s | 4 s | NVENC; decode + drawtext dominate |

End to end for the 52 s clip, no lipsync: **24 s** wall. With MuseTalk fast:
~3.3 min. With LatentSync + stabilisation: ~10.5 min.

Cold model loads are now paid at container start (`WARMUP_MODELS`):
whisper 3 s, NLLB 6 s, XTTS 26 s.

MuseTalk fast, per phase (1562 frames in, 1211 audio frames out), after
the batching change: read frames 12 s, audio features 11 s, SCRFD on GPU
16 s (cached on rerun), VAE encode 18 s (was 382 s), UNet 16 s,
compositing 68 s (was 150 s), cv2 write 27 s, mux 3 s.

## Quality

Frames at t=12 s, mouth region:

- **MuseTalk (fast and with CodeFormer).** Lips track the audio, but the
  regenerated lower face is a smooth, lighter patch: stubble is gone and
  there is a visible seam at the jaw. CodeFormer does not bring stubble
  back; it makes skin plastic and shifts the eyes across the whole face.
  This is the 256 px VAE ceiling the CPU-era docs predicted. Acceptable
  for mode 1 only if the subject is clean-shaven; try `mouth` blend mode.
- **LatentSync 512 fp16.** Stubble preserved, natural mouth, no seam.
  Clearly demo quality. The stabilised output looks the same at this
  level. The CPU bf16 "quantised bouncing" jitter did not visibly recur in
  fp16 on GPU; a `stability_metric.py` run is still owed.
- **Transcript / translation.** Whisper large-v3: 9/9 segments correct.
  NLLB 3.3B Spanish: fluent. One hallucination seen on a one-sentence
  input ("Good morning." → "Buen día. ¿Cómo estás?").
- **TTS.** XTTS Spanish 50 s for 51 s source. IndicF5 Hindi 52.6 s. F5
  English 37 s (F5 paces off the reference and this speaker's chosen
  span is quick). Listening checks are still owed.

## Bugs found and fixed on the hardware

| Symptom | Cause | Fix |
|---|---|---|
| backend image build failed | Ubuntu's Debian-managed pip refuses to upgrade | venv in both GPU Dockerfiles |
| every transcribe: `open() got an unexpected keyword argument 'metadata_errors'` | unpinned PyAV 19; faster-whisper needs <15 | `av==14.2.0` |
| F5-TTS prefetch: config `F5-TTS_v1.yaml` missing | f5-tts 1.1 renamed configs | default `F5TTS_v1_Base` |
| MuseTalk `/weights` never ready | download script wrote CodeFormer under `musetalk/codeformer/`, runner reads `codeformer/` | script writes the right place and migrates |
| MuseTalk face detection 13.5 min for 30 s | onnxruntime-gpu 1.30 links CUDA 13; provider failed to load, fell back to CPU | pin `<1.23`; LatentSync image also gets the pip CUDA libs on `LD_LIBRARY_PATH` |
| MuseTalk onnxruntime-gpu shadowed | `insightface>=0.7.3` resolved to 2.x which depends on CPU onnxruntime | pin `insightface==0.7.3` |
| MuseTalk VAE encode 382 s | per-frame CPU preprocessing; torch 172 threads in an 8-CPU cgroup | batched GPU encode, `OMP_NUM_THREADS=16`, CPU limit 32 |
| MuseTalk CUDA OOM | GPU 1 occupied by someone else's vLLM | per-service GPU ids in `.env` |
| F5 / IndicF5 output 4-5x too short | whole 52 s source + full transcript as reference; F5 clips refs to ~12 s so the text/duration ratio was wrong | 3-10 s clean word span + exactly its words |
| F5 output then 4.5 s | silence trim kept only the *longest* non-silent span, dropping 8 of 9 sentences | keep first-to-last speech span (affects XTTS too) |
| LatentSync re-downloaded buffalo_l on every container | relative insightface root | `MODEL_CACHE_DIR/insightface` |

## Functional coverage

- Backend unit tests: 24/24 pass in the GPU container.
- Pipeline end to end: XTTS/es, F5/en, IndicF5/hi; lipsync none,
  Wav2Lip, MuseTalk fast, MuseTalk + CodeFormer, LatentSync + stabilise.
- WebRTC ingest: `scripts/e2e_client.py` streams the fixture over a real
  peer connection (ICE connected in 5 s, audio + video tracks, recorder,
  `/stop`, backend job 201, job completed with MuseTalk). The `/live` page
  itself has not been exercised in a browser with a webcam yet.
- Not run: LatentSync `stability_metric.py`; listening tests on the TTS
  outputs; the CPU compose stack after these changes (device defaults to
  `cpu` and the CPU Dockerfiles only changed their extras).

## Second pass (2026-10-03): sharding and the CPU bookends

Sampled GPU utilization during the first-pass run was the motivation:
over the 12-minute run our three cards averaged 2%, 5% and 38%; the
kernels pegged at ~99% when running, everything around them was CPU.

| | first pass | second pass | what changed |
|---|---|---|---|
| MuseTalk fast, lipsync stage | 168 s | **96 s** | BiSeNet batched on GPU (20 s for 1538 crops); blend is a crop-local numpy composite across 16 threads (105 s → 4.7 s); output written by one ffmpeg NVENC pass over a pipe (34 s → 7 s) |
| LatentSync, lipsync stage, warm | 422 s (1 GPU) | **247 s** (4 GPUs) | denoise sharded over one UNet + VAE replica per GPU with overlapped conditioning/decode; pipeline and face detector reused across requests (18 s + 43 s per job gone); checkpoint save off on CUDA (30 s); NVENC for the 25 fps re-encode and the final write |
| LatentSync, lipsync stage, cold | 422 s | 306 s | first request after a restart pays the ~12 s pipeline build and ONNX session creation |

End to end for the 52 s clip: MuseTalk ~2.0 min, LatentSync ~4.5 min
(stabilisation off). Output quality unchanged in both cases
(same-frame comparisons against the first-pass outputs).

### Where the remaining LatentSync time goes (warm, 247 s)

| Phase | Wall | GPUs |
|---|---|---|
| read + 25 fps re-encode + detect + warp | 30 s | mostly CPU; detect now ~8 s |
| conditioning + sharded denoise + decode (overlapped) | 182 s | gpu4 81%, gpu5–7 ~58% |
| restore + write | 35 s | CPU warp-back per frame; NVENC write |

The denoise phase did not improve between the "overlap" and "worker-side
decode" iterations, and per-replica utilization stays near 58%: four
Python threads launching ~hundreds of kernels per UNet step contend for
the interpreter lock. The replicas are launch-bound, not compute-bound.
Threads were the cheap first step; the next one is **one process per
GPU** (torch.multiprocessing with CUDA IPC for the latents), or CUDA
graphs / `torch.compile` to cut launches per step. Either should get the
denoise phase near 70 s, where the GPU work actually is.

### Knobs that moved

| Env | Default | Meaning |
|---|---|---|
| `LATENTSYNC_GPUS` | `2` | comma list of host GPUs for the LatentSync container (`4,5,6,7` on the XE7740) |
| `LATENTSYNC_UNET_REPLICAS` | `0` = all visible | cap replica count |
| `LATENTSYNC_DENOISE_CACHE` | off on CUDA | re-enable the 2 GB resume checkpoint |
| `LATENTSYNC_VIDEO_ENCODER` | `h264_nvenc` on CUDA | intermediate re-encode and final write; libx264 fallback |
| `MUSETALK_COMPOSITE_THREADS` | `16` | blend thread pool |
| `MUSETALK_VIDEO_ENCODER` | `h264_nvenc` on CUDA | final encode; libx264 fallback |

DeepCache is disabled automatically when more than one replica is in
use (its patched cache state is per-UNet).

## Follow-ups, in priority order

1. **LatentSync denoise workers as processes, not threads** (see second
   pass). The sharding is in place and correct; the interpreter lock caps
   it at ~2x on 4 GPUs. Processes or CUDA graphs should reach ~4x.
2. **LatentSync warp and restore** (~15 s + ~28 s): per-frame kornia
   warps at 1080p. Batch them with batched affine matrices.
3. **Post-stabilisation** is 170 s of CPU. Either drop it for LatentSync
   on GPU (jitter not observed) or move vidstab to a wider CPU quota.
4. **Mode 1 quality.** Evaluate `mouth` blend mode and MuseTalk fp16;
   CodeFormer should default off for mode 1.
5. **Translation hallucination** on very short inputs: add
   `no_repeat_ngram_size`/length penalty or fall back to 600M for
   one-liners; G8 (LLM translation) supersedes this.
6. **Job queue (G4)** so mode 1 and mode 2 do not serialise behind each
   other (`MAX_CONCURRENT_JOBS=1` was needed for clean timings here).
