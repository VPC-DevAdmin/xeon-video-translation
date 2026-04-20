# Lip sync on a Xeon CPU — what to expect

This repository ships with a **pluggable** lipsync stage. Four backends are
defined; only `none` and `wav2lip` are wired up in this build. MuseTalk and
LatentSync are intentionally stubbed — the UI dropdown exposes them so the
shape is there, but they raise a clear error if you pick them.

| Backend     | Status   | Wall-clock for a 3 s clip | Quality   | License of weights |
| ----------- | -------- | ------------------------- | --------- | ------------------ |
| `none`      | shipping | <1 s                      | dub-over  | n/a                |
| `wav2lip`   | shipping | ~30–60 s                  | mediocre  | CC-BY-NC 4.0       |
| `musetalk`  | stubbed  | projected 6–18 min        | good      | MIT (code)         |
| `latentsync`| stubbed  | projected 30–60+ min      | best      | Apache 2.0         |

"Shipping" in this context means "implemented and callable". Quality is
honestly assessed — all three real lipsync backends produce visible artifacts
at some level, and on a CPU none of them are production-grade.

## `none` — dub over

The fastest and most reliable option. Stage 5 copies the original video
bytes; Stage 6 replaces the audio track and burns in the watermark. The
resulting `final.mp4` shows the speaker's lips still moving in the source
language while you hear the translated voice. Think foreign-film dub.

Default for a reason. Always works.

## `wav2lip` — implemented, mediocre

Vendored from [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) (IIIT
Hyderabad, 2020). Architecture is a GAN with a face encoder, audio encoder,
and face decoder. We run it on CPU with float32 weights.

**First run** downloads `wav2lip_gan.pth` (~415 MB) into
`$MODEL_CACHE_DIR/wav2lip/wav2lip_gan.pth`. The default URL points at a
GitHub release asset (immutable). Override with `WAV2LIP_CHECKPOINT_URL` if
it ever breaks:

```
# Default (GitHub release, justinjohn0306/Wav2Lip)
WAV2LIP_CHECKPOINT_URL=https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth

# Alternate HuggingFace mirrors (all 435,801,865 bytes, identical content)
WAV2LIP_CHECKPOINT_URL=https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth
WAV2LIP_CHECKPOINT_URL=https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth
```

### Measured limitations on CPU

- **Speed**: scales linearly with frame count. A 3 s @ 24 fps clip has 72
  frames; Wav2Lip runs in batches of 16 on a 16-core Xeon at roughly
  1–2 FPS including face detection. Budget 30–60 s.
- **Face detection**: we use OpenCV's Haar frontal cascade. It's fast but
  fragile — profile shots are not detected, and fast head motion drops
  frames. Dropped frames get left untouched (speaker's original mouth shows
  through for that frame).
- **Quality**: Wav2Lip regenerates a 96×96 mouth patch and upsamples back
  into the frame. Expect softness / slight color mismatch at the seam. Not
  subtle, especially on high-resolution source footage.
- **Phoneme distance**: EN→ES looks passable; EN→JA/ZH looks worse because
  the mouth shape library the model learned is heavily English.

### When wav2lip will refuse

- No face detected in *any* frame → raises `LipsyncError`. Switch to `none`.
- Silent audio → raises (mel spectrogram gets NaNs).
- Video has no frames / is corrupt.

## `musetalk` — microservice, staged rollout

MuseTalk upstream pins `transformers==4.39.2` which conflicts with the main
backend's `>=4.44` (coqui-tts). To keep both alive we run MuseTalk in its own
container (`lipsync-musetalk`) with its own Python env. The backend calls it
over HTTP and passes **file paths** through the shared `/jobs` volume — no
bytes cross the wire.

### Known quality issues on the current build

- **Mouth softness** is MuseTalk's inherent 256×256 VAE ceiling. Not fixable
  without retrained weights at higher resolution; no public release exists.
- **Face detection gaps**: dropped detections would previously leak the
  original frame (passthrough), flipping visibly against MuseTalk-regenerated
  neighbors. Now: the nearest valid bbox is forward-filled through gaps so
  every frame gets inference.
- **Bbox jitter**: SCRFD returns accurate but not stable boxes — typical
  jitter is ±few pixels frame-to-frame on a static face. A 5-frame moving
  average on `(x1, y1, x2, y2)` dampens the jitter enough that the blend
  seam doesn't wobble between frames. Shorter window = more responsive to
  real head motion; longer = smoother but laggier. 5 is the current default.
- **Anatomical coupling** — three blend modes exposed via
  `MUSETALK_BLEND_MODE`:
  - `raw`: full lower face replaced (original v1.0 behavior; lots of
    texture loss)
  - `jaw` (default): chin + cheeks + mouth together. Chin moves with the
    lips, which is how real jaws work. Some stubble in the chin/cheek
    region is regenerated and softened; a downstream face-restoration
    pass (CodeFormer, separate PR) brings that detail back.
  - `mouth`: lips + teeth only. Best stubble preservation but freezes
    the chin while the lips move, producing a visible "sticker" effect
    when the speaker opens their mouth. Mask is dilated by ~7 px inside
    `face_parsing` to avoid a secondary ghosting issue (the Gaussian
    feather at composite time would otherwise erode such a small mask's
    opaque center).
  Feather is controlled by `MUSETALK_BLEND_FEATHER` (fraction of crop
  width; default 0.04). The absolute kernel is capped at 31 px regardless
  of crop size — larger kernels look smoother on big masks but erode
  small ones, so we cap pre-emptively.
- **Quality ceiling**: even with those fixes, MuseTalk's output is "mouth
  is regenerated, fuzzy but continuous" rather than "indistinguishable from
  source". For demo, a reasonable tradeoff; for production, a higher-res
  successor model is the real answer.

### Detector: InsightFace SCRFD (not face-alignment)

Earlier builds used `face-alignment` (SFD + FAN). On the demo 53-frame clip,
face detection alone took 9m23s — nearly half the total MuseTalk runtime.
SCRFD through ONNX Runtime runs the same workload in seconds while producing
comparable boxes for front-facing talking heads. The `buffalo_l` pack is
~285 MB; pre-fetched by `scripts/download_models.sh` alongside the MuseTalk
weights.

### Detection cache

Raw SCRFD output (per-frame bbox + score) is cached under
`${MODEL_CACHE_DIR}/cache/face_detections/<sig>.json`, keyed on a cheap
signature of the input video (size + first 1 MB + detector version). A
second run on the same clip skips the detection pass entirely. This is
purely a dev-iteration optimization; on a first run it costs nothing.

### Face restoration — CodeFormer (post-blend)

MuseTalk's 256×256 VAE round-trip softens skin detail (pores, stubble,
fine wrinkles). The lipsync service runs **CodeFormer** (Shangchen Zhou et
al., 2022, vendored under `services/musetalk/app/musetalk/_codeformer/`)
on every output frame to recover that detail. CodeFormer is a face-only
restoration model trained to preserve identity while sharpening texture.

Pipeline placement: **after** the MuseTalk blend. Frames flow as:

```
SCRFD → UNet → VAE decode → BiSeNet blend (jaw/mouth mode) → CodeFormer restore → mux
```

Per-frame cost on Xeon is ~3–8 s. For a 53-frame clip that's ~3–7 min
added on top of MuseTalk's own run time.

Knobs (all via env):

```
MUSETALK_FACE_RESTORE          codeformer | none   (default: codeformer)
MUSETALK_FACE_RESTORE_FIDELITY 0.0–1.0             (default: 0.7)
MUSETALK_FACE_RESTORE_BLEND    0.0–1.0             (default: 0.6)
```

`FIDELITY` is CodeFormer's internal `w` parameter. 0.0 = maximum visual
quality but may morph identity; 1.0 = strict identity preservation with
less detail recovery. 0.7 is the commonly cited balance.

`BLEND` is how much of the restored face to alpha-blend on top of
MuseTalk's output. A 5-point affine transform (using SCRFD's keypoints)
aligns the face to CodeFormer's 512×512 canonical positions and inverse-
warps the result back. The blend mask is a Gaussian-feathered ellipse so
the seam at the face boundary doesn't show.

Weight: `codeformer.pth` (~376 MB), official GitHub release
(`sczhou/CodeFormer`). Pre-fetched by `scripts/download_models.sh`; if
missing at inference time the stage is skipped with a warning.

Output quality: the regenerated lower face keeps its anatomy (lips track
jaw from the `jaw` blend mode) but CodeFormer brings the stubble/pore
detail back. Identity preservation is usually good at fidelity=0.7.

### `QUALITY=N` ladder

Progressive dial over the per-request knobs above. The Makefile passes the
corresponding `MUSETALK_*` values as form fields on `POST /jobs`, which the
backend forwards to the lipsync service in the `/lipsync` request body.
**No container restart needed between invocations** — the already-loaded
models just read the per-request overrides in `inference.run()`.

| `QUALITY` | blend_mode | blend_feather | face_restore | Notes |
|---|---|---|---|---|
| `1` | `raw` | `0.06` | `none` | Minimum. Whole lower face replaced, no restoration. Fastest. |
| `2` | `jaw` | `0.04` | `none` | Balanced. Anatomically correct jaw motion, skip restoration. |
| `3` ← default | `jaw` | `0.04` | `codeformer` (0.7 / 0.6) | Full quality. Jaw + CodeFormer restoration. |

Numeric today; will likely migrate to `low | medium | high` once more
features land. F5-TTS landed as a separate `TTS` knob rather than a
`QUALITY` level (orthogonal axis: voice quality vs. lipsync quality);
LatentSync, once wired up, would claim levels 4+ on this dial.

Use:
```bash
make run-musetalk              # uses QUALITY=3 (default)
make run-musetalk QUALITY=2    # skip face restore for ~30 min less wall time
make run-musetalk QUALITY=1    # fastest; visibly lower quality
```

### CPU acceleration — IPEX, TCMalloc, Intel OpenMP

The lipsync service image ships with Intel's performance tooling wired in by
default:

- **Intel Extension for PyTorch (IPEX)** — `ipex.optimize()` is applied to
  the Whisper encoder, SD-VAE, and MuseTalk UNet at load time. On Xeon
  Sapphire Rapids+ this gives a material speedup on Conv/Linear kernels.
- **TCMalloc** (`libtcmalloc.so` from `libgoogle-perftools4`) — preloaded
  before Python starts; reduces allocator contention under heavy ML load.
- **Intel OpenMP** (`libiomp5.so` from the `intel-openmp` pip wheel) —
  preferred over libgomp for math-heavy parallel loops. `KMP_AFFINITY=
  granularity=fine,compact,1,0` and `KMP_BLOCKTIME=1` are set in
  docker-compose.yml to keep threads pinned and avoid spin-wait penalties.

#### bf16 mode

bf16 autocast on CPU is opt-in behind the `MUSETALK_IPEX_DTYPE` env var:

```
MUSETALK_IPEX_DTYPE=fp32   # default; safe
MUSETALK_IPEX_DTYPE=bf16   # ~1.5-2x faster on AMX-capable Xeon; try it
```

bf16 uses `torch.autocast(device_type="cpu")` around the UNet forward. The
autocast allowlist keeps BatchNorm/LayerNorm fp32 so stability is usually
fine, but output can drift subtly (mouth texture, color). Compare before
shipping.

#### Disabling

All three are non-fatal — if IPEX fails to import, or the preload libraries
aren't present in the image, the service logs a warning and runs vanilla
PyTorch.

### Roadmap

| Phase | What ships | Behavior when you pick `musetalk` |
|-------|------------|-----------------------------------|
| **PR 1a** | FastAPI scaffold, Compose wiring, HTTP client | Fails fast with a clean `LipsyncError` pointing here |
| **PR 1b** | torch/diffusers/transformers pinned; face-alignment (replaces mmpose); weight downloads; `/ready` + `/weights` introspection | Same 501 but the container has everything on disk |
| **PR 1c** (current) | Vendor MuseTalk V1.5 model code, AudioProcessor (HF Whisper with `output_hidden_states=True`), face-alignment preprocessing, UNet+VAE forward, BiSeNet-aware blending | **First runnable pass.** Expect initial bugs; iterate. |

### Pre-fetch weights

```bash
docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh
```

Downloads ~1.4 GB into the shared `/models/musetalk/` volume:

| Weight | Source | Size | License |
|---|---|---|---|
| MuseTalk V1.5 UNet | `TMElyralab/MuseTalk` (HF) | ~900 MB | CC-BY-NC 4.0 |
| SD 1.5 VAE (ft-MSE) | `stabilityai/sd-vae-ft-mse` (HF) | ~330 MB | CreativeML Open RAIL-M |
| Whisper tiny | `openai/whisper-tiny` (HF) | ~75 MB | MIT |
| BiSeNet face parser | `zllrunning/face-parsing.PyTorch` (Google Drive) | ~52 MB | MIT |
| ResNet18 backbone | `pytorch.org` | ~46 MB | BSD |

If the Google Drive fetch fails, download `79999_iter.pth` manually from
https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view and
drop it in `/models/musetalk/face-parse-bisent/`.

### Introspection

```bash
curl http://localhost:8089/health      # top-level: is the service up + weights on disk?
curl http://localhost:8089/ready       # are all Python ML deps importable?
curl http://localhost:8089/weights     # per-file presence and size check
```

### Why a separate service?

- **Dependency isolation.** MuseTalk + coqui-tts can't share a Python env.
- **Rebuild blast radius.** Inference changes in the lipsync service don't
  invalidate the backend image's layers (and vice-versa).
- **Future-proof.** Swapping in a GPU variant later is trivial — just change
  the compose service definition.

### HTTP contract

```
POST /lipsync
{
  "video_path":  "/jobs/<job_id>/input.mp4",
  "audio_path":  "/jobs/<job_id>/translated_audio.wav",
  "output_path": "/jobs/<job_id>/lipsynced.mp4"
}
```

Both services mount `/jobs` and `/models`. Backend reaches the service via
Docker DNS at `http://lipsync-musetalk:8000`; the host can hit it for
debugging at `localhost:${MUSETALK_PORT:-8089}`.

## `latentsync` — live inference (PR-LS-1c)

SD 1.5 latent-diffusion based. Best open-source lipsync quality as of
writing, and designed as a **batch workflow** on CPU: the architecture
(per-frame latent denoising at video frame rate) isn't optimizable into
live-demo territory on any current-gen CPU.

**CPU budget:** ~10 minutes of wall-clock per second of source video. A
10 s clip takes 1.5–2 hours. For translation workflows where batch
overnight is acceptable (archival, research, accessibility), that's
reasonable; for a demo booth, use `LIPSYNC=musetalk` or `LIPSYNC=none`.

### State as of PR-LS-1c (this PR)

**Inference is live.** The vendored LatentSync package
(`services/lipsync-latentsync/app/latentsync/`) is callable end-to-end
on CPU. `make run-latentsync` kicks a real job.

**Performance stack** (IPEX bf16 + DeepCache, both default-on):

| Config | Per source-second | For a 3 s clip |
|---|---|---|
| fp32, no DeepCache | ~90 min | ~4.5 h |
| bf16 IPEX, no DeepCache | ~25 min | ~1.25 h |
| bf16 IPEX + DeepCache (default) | **~18 min** | **~55 min** |

The bf16 path requires an AMX-capable Xeon (Sapphire Rapids or newer);
older cores fall through to fp32 without error. Flip to fp32 via
`LATENTSYNC_IPEX_DTYPE=fp32` in `.env` if you're chasing a quality
regression; flip DeepCache off with `LATENTSYNC_ENABLE_DEEPCACHE=0`
for the same reason. Both stack multiplicatively.

**Dry-run mode for fast iteration.** Set `LATENTSYNC_DRY_RUN=1` in your
`.env` and rebuild — the driver collapses the denoising loop to 1 step
so you can verify the pipeline wires together in minutes instead of
hours. Output quality is unusable (single-step diffusion ≠ a real take)
but face detection, VAE encode/decode, and ffmpeg mux all run, so
wiring bugs surface fast.

**Per-request knobs** (all accept `null` for env-default fallback):

| Field | Range | Default | Purpose |
|---|---|---|---|
| `num_inference_steps` | 1–100 | `LATENTSYNC_STEPS=20` | Denoising steps. Halving roughly halves wall time; below 10 visible quality drops |
| `guidance_scale` | 0–15 | `LATENTSYNC_GUIDANCE=1.5` | Classifier-free guidance. Higher pushes mouth shapes harder at identity cost |
| `seed` | int | random | Set for reproducibility between runs |

**Container-level knobs** (env only, not per-request):

| Env | Default | Purpose |
|---|---|---|
| `LATENTSYNC_IPEX_DTYPE` | `bf16` | `bf16` = 2–4× faster on AMX Xeon; `fp32` = vanilla path |
| `LATENTSYNC_ENABLE_DEEPCACHE` | `1` | Cache intermediate UNet features (~1.3× on top of IPEX) |
| `LATENTSYNC_DRY_RUN` | `0` | Collapse diffusion to 1 step; smoke-test wiring only |

**CPU adaptations** (inline in the vendored tree, marked with
`CPU patch:` comments — see `services/lipsync-latentsync/NOTICE` for
the full list):

- `ImageProcessor` now respects `self._execution_device` instead of
  hard-coding `"cuda"`.
- `FaceDetector` routes through `onnxruntime` CPU provider + InsightFace
  `ctx_id=-1` when device isn't `cuda:*`.
- `AlignRestore` default dtype changed from `float16` to `float32` —
  many CPU kernels don't have fp16 paths.
- `torch.cuda.empty_cache()` calls gated behind
  `torch.cuda.is_available()`.
- Dropped the upstream `NotImplementedError("Using the CPU for face
  detection is not supported")` guard that refused CPU face detection
  outright.

**What didn't change.** The UNet, the diffusion pipeline math, the
Audio2Feature whisper wrapper, the loss functions, the overall
architecture — all upstream verbatim.

Exercising it:

```bash
make rebuild                       # rebuilds image with vendored code + driver
make models-latentsync             # pulls ~6.6 GB of weights (once)
make health                        # all three services
curl http://localhost:8090/health  # inference_implemented: true, phase: PR-LS-1c

# Smoke test: dry-run (~2 min)
echo LATENTSYNC_DRY_RUN=1 >> .env
make rebuild
make run-latentsync FIXTURE=artifacts/inputs/short.mov   # ~2 min, unusable output but proves wiring

# Real run (~10 min per second of source)
sed -i '/LATENTSYNC_DRY_RUN/d' .env
make rebuild
make run-latentsync FIXTURE=artifacts/inputs/short.mov   # real output
```

### Roadmap

| Phase | What ships | Status |
|-------|------------|--------|
| **PR 35 (scaffold error)** | `make run-latentsync` target, inline dispatcher error pointing here | shipped |
| **PR-LS-1a** | Dep-isolated `lipsync-latentsync` microservice, Compose wiring, HTTP client, structured 501 end-to-end, introspection endpoints (`/health`, `/ready`, `/weights`) | shipped |
| **PR-LS-1b** | torch 2.5.1 CPU + diffusers 0.32.2 + transformers 4.48.0 + the rest of LatentSync's `requirements.txt` (CPU-adapted) + real `scripts/download_models.sh` pulling UNet + SyncNet + Whisper tiny from `ByteDance/LatentSync-1.6` | shipped |
| **PR-LS-1c (this PR)** | Vendored LatentSync inference tree + CPU adaptation patches + driver + per-request quality knobs (`num_inference_steps`, `guidance_scale`, `seed`) + `LATENTSYNC_DRY_RUN` mode | shipping |
| **GPU path (later)** | Optional CUDA image variant gated behind a compose profile — intentionally not prioritized for the CPU demo | not planned |

### Per-request knobs (forward-compat in PR-LS-1a, wired in PR-LS-1c)

The `POST /lipsync` payload schema already accepts LatentSync-specific
fields so clients can be written against them now:

| Field | Range | Default | Purpose |
|---|---|---|---|
| `num_inference_steps` | 1–100 | `LATENTSYNC_STEPS=20` | Denoising steps per frame. Halving this ~halves wall time; quality drops visibly below 10. |
| `guidance_scale` | 0.0–15.0 | `LATENTSYNC_GUIDANCE=1.5` | Classifier-free guidance. Higher = stronger mouth shaping, at cost of identity drift. |
| `seed` | int | random | Reproducibility. Fix when comparing runs. |

In PR-LS-1a the service accepts and ignores these; in PR-LS-1c they gate
the actual diffusion loop.

## ETA and progress events

The orchestrator publishes two SSE events the UI consumes:

- `pipeline_etas` — emitted once after Stage 1 runs, carrying
  `{source_duration_seconds, lipsync_backend, etas: {...}}` with ETA estimates
  per remaining stage.
- `stage_progress` — emitted throughout the lipsync stage (Wav2Lip only;
  stubs never reach it). Payload is `{stage: "lipsync", percent: 0.0..1.0}`.

Only Wav2Lip publishes real progress — for transcribe/translate/tts the UI
shows an indeterminate spinner with the pre-computed ETA.

## Responsible use

All three lipsync backends can produce convincing synthetic video of a real
person saying something they didn't say. That's why:

- `ENABLE_WATERMARK=true` is the default and should stay on.
- `docs/ethics.md` is required reading.
- The `wav2lip` weights are **CC-BY-NC 4.0** (non-commercial research). If
  you ship anything based on this, you're on your own for the license.
