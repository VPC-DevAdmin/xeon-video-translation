# LatentSync pipeline — internals, settings, and the 2026 jitter bisection

This is the definitive reference for the `lipsync-latentsync` service.
It covers the full end-to-end pipeline, every tunable, the engines
involved, the 2026 jitter investigation and production fix, and a
debugging playbook for recovering quality regressions.

`docs/lipsync.md` is still the top-level "which backend to use and why"
document; read it first if you're choosing between `wav2lip`,
`musetalk`, and `latentsync`. This doc is the deep-dive for the third.

---

## 1. What LatentSync is, in one paragraph

A latent-diffusion lipsync model built on Stable Diffusion 1.5. The
UNet is conditioned on a canonical face crop, a binary mouth mask, an
audio embedding from Whisper, and a fresh noise sample; at inference
it denoises the mouth region into the shape that matches the audio.
The rest of the face comes from the original frame. Output quality
is the best available open-source as of writing. CPU inference is
**not** a live workflow — budget ~10 minutes of wall-clock per
second of source video.

---

## 2. End-to-end pipeline — stages, engines, data flow

What happens from the `POST /jobs` call to the final watermarked
mp4. Each stage runs in sequence in `backend/app/pipeline/orchestrator.py`
and reports progress via a shared JSON file on the `/jobs` volume.

```
upload.mov
   │
   ▼ [audio]          ffmpeg: extract mono 16 kHz WAV
audio.wav
   │
   ▼ [stabilize]      ffmpeg vidstab 2-pass (optional; off by default)
stabilized.mp4                  libx264 -crf 18
   │
   ▼ [transcribe]     faster-whisper base, int8, word timestamps
transcript.json                 ~6× realtime on Xeon
   │
   ▼ [translate]      NLLB-600M distilled, int8
translation.json                ~3 s per segment
   │
   ▼ [tts]            XTTS-v2 (Coqui) per-segment with smart reference,
translated_audio.wav            optional F5-TTS backend, Whisper-based
                                tail trim + loudness normalization
   │
   ▼ [lipsync]        LatentSync — see §3
lipsynced.mp4                   ~10 min per source-second on CPU
   │
   ▼ [poststabilize]  ffmpeg vidstab (optional; off by default)
   │
   ▼ [mux]            ffmpeg: watermark + final encode
final.mp4
```

Only the lipsync stage is LatentSync-specific. The rest is shared
with the other backends; see `docs/architecture.md` for the broader
pipeline.

---

## 3. LatentSync lipsync stage — internals

The lipsync stage is itself a multi-phase pipeline inside the
`lipsync-latentsync` service. Understanding the phases matters because
every performance knob and quality-tuning env var acts on exactly one
of them.

### 3.1 Phase map

```
input.mp4 + translated_audio.wav
   │
   ▼ [1. read_video]           ffmpeg re-encode to 25 fps H.264
   │                           (lossy by default; see LATENTSYNC_LOSSLESS_READ)
   ▼ [2. audio2feature]        Whisper encoder → per-frame audio embedding
   ▼ [3. affine_transform_video]
   │   ├─ face_detector        InsightFace SCRFD + 2d106det (ONNX, CPU)
   │   ├─ extract_landmarks3   pick 3 anchors: left-eye, right-eye, nose
   │   ├─ smooth_landmarks     ← PRIMARY JITTER FIX (PR #64)
   │   │                       SG filter across time on each landmark
   │   │                       Window = LATENTSYNC_LANDMARK_SMOOTH_WINDOW
   │   ├─ transformation_from_points
   │   │                       SVD on 3-point correspondence → similarity matrix
   │   │                       (smooth=False — upstream IIR bias disabled, PR #62)
   │   └─ kornia.warp_affine   crop face to canonical 512×512
   ▼ [4. prepare_masks]        binary mouth mask + encode to latents via VAE
   ▼ [5. denoising loop]       UNet × num_inference_steps
   │                           conditioned on audio + ref face + mask
   │                           DeepCache wraps this (optional)
   │                           IPEX autocast (optional)
   ▼ [6. decode_latents]       VAE decode back to pixels
   ▼ [7. paste_surrounding_pixels_back]
   │                           keep non-face pixels from the ref frame
   ▼ [8. smooth_affine_sequence]
   │                           ← SECONDARY smoother, off by default (PR #67)
   │                           decompose each affine into (tx, ty, θ, scale),
   │                           SG-smooth each component, recompose
   ▼ [9. restore_video]        kornia.warp_affine inverse + soft-mask composite
   ▼ [10. write_video]         ffmpeg encode lipsynced.mp4 (libx264 -crf 13)
```

### 3.2 Engines

| Phase | Engine | Role | Notes |
|---|---|---|---|
| 1. read_video | ffmpeg + libx264 | 25 fps re-encode | `-crf 18` (lossy) default; `LATENTSYNC_LOSSLESS_READ=1` switches to `-qp 0` |
| 2. audio2feature | Whisper tiny | audio → per-frame embedding | Runs at 50 fps effective |
| 3. face detection | InsightFace SCRFD + 2d106det | bbox + 106 landmarks | ONNX Runtime CPU; bit-deterministic on identical pixels |
| 3. warp-to-canonical | kornia | SVD-derived similarity transform | Deterministic at 64 threads |
| 4. VAE encode | SD 1.5 VAE | pixel → latent | fp32 by default (`LATENTSYNC_IPEX_DTYPE`) |
| 5. UNet | SD 1.5 UNet (custom checkpoint) | denoising × num_inference_steps | ~90% of wall-clock |
| 5. DeepCache | `DeepCacheSDHelper` | reuse cached UNet features | ~1.3× speedup, optional |
| 5. IPEX | Intel Extension for PyTorch | bf16 autocast, AMX kernels | Requires torch + triton version match (PR #59) |
| 6. VAE decode | SD 1.5 VAE | latent → pixel | same dtype as encode |
| 9. restore_video | kornia + cv2 | inverse warp + soft mask composite | fp32, deterministic |
| 10. write_video | ffmpeg + libx264 | lossy encode | `-crf 13`; not currently env-toggleable |

### 3.3 Two-pass affine (the production jitter fix)

Added in PR #64. The key architectural change from upstream.

**Upstream** ran detection + affine in a single per-frame call. Raw
landmarks — with natural sub-pixel noise from lossy H.264 or
detection variance — fed the SVD directly. The
`(s2 / s1) × R` scale in `transformation_from_points` amplified
fractional-pixel jitter into pixel-scale drift in the affine, which
became visible "bouncy face" on playback.

**Our pipeline** does it in two passes:

1. **Extract landmarks for every frame** and collect them into a
   trajectory.
2. **Temporally smooth each landmark's (x, y) independently** with a
   Savitzky-Golay filter (window = `LATENTSYNC_LANDMARK_SMOOTH_WINDOW`,
   polyorder 2). This damps sub-pixel noise before it reaches the SVD.
3. **Compute the per-frame affine** from the smoothed landmarks.

The upstream in-line IIR bias smoother inside `transformation_from_points`
is forced off (PR #62, `smooth=False` at the single caller), so the
landmark smoother is the single explicit smoothing stage.

---

## 4. Settings reference

Every environment variable the service recognizes. Defaults reflect
the production config after PRs #55 through #67.

### 4.1 Inference quality / speed

| Env | Default | Range | What it does |
|---|---|---|---|
| `LATENTSYNC_STEPS` | `20` | 1–100 | Denoising steps. Halving roughly halves wall time; below 10 is visibly worse. Per-request override also supported. |
| `LATENTSYNC_GUIDANCE` | `1.5` | 0–15 | Classifier-free guidance. Higher pushes mouth shapes harder at the cost of subject identity. Per-request override. |
| `LATENTSYNC_IPEX_DTYPE` | `fp32` | `fp32` / `bf16` | UNet + VAE compute precision. See §4.4. |
| `LATENTSYNC_ENABLE_DEEPCACHE` | `1` | 0 / 1 | Reuse cached UNet features across denoising steps. ~1.3× faster at negligible quality cost on SD 1.5. |
| `LATENTSYNC_DRY_RUN` | `0` | 0 / 1 | Collapse denoising to 1 step; smoke-test wiring only. Output is unusable. |

### 4.2 Jitter / smoothing

| Env | Default | What it does |
|---|---|---|
| `LATENTSYNC_LANDMARK_SMOOTH_WINDOW` | `5` | **Primary** smoother. SG-filter over per-landmark (x, y) trajectories. Window at 25 fps: 5 = 0.2 s, 9 = 0.36 s, 15 = 0.6 s. `1` disables. |
| `LATENTSYNC_AFFINE_SMOOTH_WINDOW` | `1` | **Secondary** smoother, off by default. Decomposes the per-frame affine into (tx, ty, θ, scale) and SG-smooths each. Useful belt-and-suspenders on aggressive motion where the landmark smoother alone lags. Typical opt-in: 9. |

Cascade warning: these two SG filters composed produce a larger
effective support that can add lagged-oscillatory step responses.
Unless you need the secondary smoother, leave it off.

### 4.3 Diagnostics (all default off)

These exist to aid future debugging. None affect production quality
when unset.

| Env | What it does |
|---|---|
| `LATENTSYNC_BYPASS_UNET=1` | Skip denoise + VAE decode; pass reference face crops straight through. Isolates geometric pipeline jitter from UNet/VAE numerical jitter. |
| `LATENTSYNC_BYPASS_MASK=1` | Use the hard erosion mask for compositing instead of the Gaussian-blurred soft mask. Tests whether frame-to-frame soft-mask-radius fluctuation contributes to jitter. |
| `LATENTSYNC_LOSSLESS_READ=1` | Swap `read_video`'s `-crf 18` for `-qp 0 -preset veryslow`. Eliminates input-side compression noise entirely at the cost of a ~4–8× larger temp file. |
| `LATENTSYNC_DUMP_AFFINES=1` | Write pre- and post-smooth affine matrices + boxes to `/jobs/affine_debug/`. Paired with `scripts/latentsync_debug/analyze_affines.py`. |
| `LATENTSYNC_DEBUG_DUMP=1` | Save intermediate PNGs (face crop, warped mask, restore composite) to `/jobs/latentsync_debug/`. Rate-limited by `LATENTSYNC_DEBUG_FRAME_LIMIT`. |
| `LATENTSYNC_DETERMINISTIC=1` | `torch.use_deterministic_algorithms(True)` + fixed seed. For bit-reproducible runs. Requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`. |
| `LATENTSYNC_IGNORE_DENOISE_CACHE=1` | Bypass the post-denoise resume cache. Forces a full rerun. |
| `LATENTSYNC_DEBUG_FRAME_LIMIT` | Per-stage frame cap when `LATENTSYNC_DEBUG_DUMP=1`. Default 3. |

### 4.4 fp32 vs bf16 — the precision knob

The big trade-off. Measured on `IMG_7228.MOV` (real source, 1.8 s):

| Config | mean px | p95 | max | **std** | Wall clock |
|---|---|---|---|---|---|
| bf16 | 2.43 | 5.73 | 9.81 | **1.89** | 50 min |
| fp32 | 2.05 | 4.44 | 8.05 | **1.29** | 154 min |

The 31% drop in `std` is the critical number. bf16's 8-bit mantissa
snaps intermediate UNet sums to a small set of representable values;
on near-static subjects this manifests as visible "quantized bouncing
between ~4 X/Y values". fp32 removes the artifact entirely. Trade is
~3× wall-clock for clean output.

Default is `fp32`. Override to `bf16` for draft / preview workloads
where the artifact is acceptable and the speedup matters.

---

## 5. The 2026 jitter bisection (case study)

A chronological walk-through of how the residual jitter was
diagnosed and eliminated. Preserved because the investigation is more
educational than the fix itself — the pipeline has many places that
could plausibly introduce frame-to-frame drift, and ruling them out
one at a time required real instrumentation.

### 5.1 Tools built

`scripts/latentsync_debug/`:

- `stability_metric.py` — the measurement. Runs InsightFace on every
  output frame, computes Euclidean per-landmark-pair displacement
  over a fixed set of 7 anchors (eye corners, nose tip, mouth
  corners). Reports mean / p50 / p95 / p99 / max / std.
- `environment_audit.py` — SHA-sums checkpoints, captures dependency
  versions, prints torch config. Diffed across runs to detect silent
  stack drift.
- `analyze_affines.py` — loads dumped pre/post-smooth affine matrices
  and boxes; reports bit-identicality and per-frame diff vs frame 0.
- `BYTEDANCE_AUDIT.md` — file-by-file diff against upstream
  `bytedance/LatentSync@a229c39`. Zero unintentional divergence.
- `DEBUG_PLAN.md` — the bisection plan that drove the investigation.

### 5.2 The investigation

Each entry is one data point on `IMG_7228.MOV` (1.8 s real source)
unless noted.

| Result | Config | Diagnosis |
|---|---|---|
| 6.23 px mean | Baseline | "Bouncy face", visibly broken |
| 0.00 px | `static_lossless.mp4`, in isolation | Noise floor of the landmark detector |
| 0.10 px | `static_test.mp4` (libx264 -crf 18) | Baseline libx264 noise contribution |
| 0.16 px | after `stabilize` stage | vidstab re-encode adds 0.06 px |
| Face detection tested 10× on same image | std = 0.0001 px | Detector is bit-deterministic |
| `torch.svd` tested 10× on same input | max diff = 0.0 | SVD is deterministic |
| `kornia.warp_affine` (fwd, inverse, round-trip) | 0.000e+00 diff | Deterministic at 64 threads |
| `LATENTSYNC_BYPASS_UNET=1` on static | 2.37 px mean | UNet contributes only to max-spikes, not mean |
| `LATENTSYNC_BYPASS_MASK=1` on static | 2.59 px mean | Soft mask is innocent |
| IPEX import at service startup | `ImportError: AttrsDescriptor` | Dead since last triton upgrade |
| `LATENTSYNC_DUMP_AFFINES=1` | Pre-smooth affines differed by up to 8 px | Smoking gun |

The last line localized the bug to `transformation_from_points` —
upstream's in-line IIR bias smoother was persisting state on the
`AlignRestore` instance and amplifying sub-pixel landmark noise into
pixel-scale affine drift. First fix: disable the IIR (PR #62).

Final diagnosis of the remaining residual:

- Sub-pixel landmark noise from `libx264 -crf 18` (unavoidable on real
  input) was amplified through the SVD's `(s2 / s1) × R`.
  **Fix: smooth landmarks upstream of the SVD** (PR #64).
- bf16 mantissa precision snapping UNet sums to discrete values
  produced a quantized bouncing artifact. **Fix: default to fp32** (PR #67).

### 5.3 Final measurements

| Stage | mean | p95 | max | std | Notes |
|---|---|---|---|---|---|
| Baseline | 6.23 | 12.93 | 18.50 | 3.60 | IIR bug, bf16 fake, no landmark smoothing |
| + IIR fix (#62) | ~4 est | | | | bf16 still broken at this point |
| + triton pin (#59) | ~3.5 est | | | | bf16 actually running |
| + landmark smoothing (#64) | 2.43 | 5.73 | 9.81 | 1.89 | Big win on mean + max |
| + fp32 default (#67) | **2.05** | **4.44** | **8.05** | **1.29** | Visible quantization gone |

From 6.23 "bouncy face" to 2.05 "clean drift with slight noise."

### 5.4 Related PRs

| PR | Focus |
|---|---|
| #55 | Debug tooling (`stability_metric.py`, `environment_audit.py`, `DEBUG_PLAN.md`) + upstream audit |
| #56 | Whisper-based TTS tail trim (different jitter: `loop_video` firing on over-long TTS) |
| #57 | ffmpeg temp-file extension fix — `_trim_to_speech` had been silently failing for unknown duration |
| #58 | `LATENTSYNC_BYPASS_UNET` + `LATENTSYNC_BYPASS_MASK` |
| #59 | Pin `triton` 3.1.x so IPEX imports (bf16 had been silently disabled) |
| #60 | `LATENTSYNC_DUMP_AFFINES` + `analyze_affines.py` |
| #61 | Compose-forward the dump env var |
| #62 | Disable upstream IIR bias — **the root-cause fix** |
| #63 | `LATENTSYNC_LOSSLESS_READ` diagnostic flag |
| #64 | Landmark temporal smoothing — **the production fix** |
| #65, #66 | `make watch` ETA — observed-rate extrapolation + bash-quote bugfix |
| #67 | Default fp32; default secondary smoother off |

---

## 6. Debugging playbook — if jitter comes back

If output jitter ever regresses (visible "bouncy face", or mean > 3 px
on a stable source):

### 6.1 Measure

```bash
# Always measure first. Don't chase perception alone.
JOB=<job-id>
docker cp "artifacts/jobs/$JOB/final.mp4" \
    polyglot-lipsync-latentsync:/tmp/final.mp4
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/stability_metric.py /tmp/final.mp4
```

Compare against the "+fp32" row in §5.3. If mean ≥ 2.5 and p95 ≥ 5, something
regressed.

### 6.2 Establish what changed

```bash
# Capture current state
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/environment_audit.py > /tmp/env_after.json

# Diff against a baseline
diff /tmp/env_before.json /tmp/env_after.json
```

Model SHA256s, torch version, IPEX version, thread counts, env flags —
one of these drifted.

### 6.3 Bisect

Standard sequence, cheapest to most expensive:

1. **Static-input test** — repeat one source frame N times and run.
   If that's clean (≤ 1 px mean) but real source is jittery, the
   source is contributing; check whether `ENABLE_VIDEO_STABILIZATION`
   changed, whether the input is unusually noisy, whether the smoothing
   window needs to go up.
2. **`LATENTSYNC_DUMP_AFFINES=1` + `analyze_affines.py`** — are the
   per-frame affines bit-identical? If not, the smoother regressed.
3. **`LATENTSYNC_BYPASS_UNET=1`** — ~5× faster test run. If static-
   input output goes to ~0 px with bypass, UNet path regressed
   (DeepCache interaction, dtype mismatch, seed drift).
4. **`LATENTSYNC_LOSSLESS_READ=1` + `ENABLE_VIDEO_STABILIZATION=false`** —
   gold-standard lossless baseline. Output should be within ~1 px of
   the fp32 measured row in §5.3.

### 6.4 Common regressions

| Symptom | Likely cause | Fix |
|---|---|---|
| Quantized bouncing between ~N discrete positions | bf16 active | Set `LATENTSYNC_IPEX_DTYPE=fp32` |
| Affines differ by ~4–8 px across frames on static input | Upstream IIR re-enabled (someone flipped `smooth=True`) | Re-apply PR #62 — `smooth=False` at the caller |
| Output shorter than input, face loops backwards | TTS overran source duration, `loop_video` fired | Check TTS post-processing; see PR #56 |
| `IPEX import failed` in logs | triton version drift | `pip install triton==3.1.0`; confirm via `environment_audit.py` |
| `make watch` ETA stuck at a constant | Shell script regression | Check `scripts/progress.sh`; observed-rate logic in PR #65 |

---

## 7. Performance vs quality matrix

Everything you can reasonably tune, with measured or estimated impact
on a 1.8 s source clip. Use this for capacity planning.

### 7.1 By axis

| Knob | Speed | Quality | Notes |
|---|---|---|---|
| `LATENTSYNC_STEPS=20` | baseline | baseline | Default. Below 10 quality drops visibly. |
| `LATENTSYNC_STEPS=10` | 2× faster | noticeable quality loss | For previews. |
| `LATENTSYNC_IPEX_DTYPE=fp32` | baseline | baseline | Default after PR #67. |
| `LATENTSYNC_IPEX_DTYPE=bf16` | ~3× faster | quantized bouncing artifact | Use only for drafts. |
| `LATENTSYNC_ENABLE_DEEPCACHE=1` | ~1.3× faster | negligible loss | Default on. Disable if chasing mysterious artifacts. |
| `LATENTSYNC_LANDMARK_SMOOTH_WINDOW=5` | 0 | baseline | Primary jitter fix. |
| `LATENTSYNC_LANDMARK_SMOOTH_WINDOW=15` | 0 | slightly laggy on fast head turns | Use on longer clips with low-frequency jitter. |
| `LATENTSYNC_AFFINE_SMOOTH_WINDOW=9` | 0 | potential cascaded-SG oscillation | Only enable if landmark smoother alone leaves residue. |
| `ENABLE_VIDEO_STABILIZATION=true` | -25 s | removes handheld shake | Re-encodes lossy — adds ~0.06 px jitter floor. Worth it on shaky input. |

### 7.2 Recommended presets

**Production** (best quality, accept batch timing):
```
LATENTSYNC_IPEX_DTYPE=fp32
LATENTSYNC_ENABLE_DEEPCACHE=1
LATENTSYNC_LANDMARK_SMOOTH_WINDOW=5
LATENTSYNC_AFFINE_SMOOTH_WINDOW=1
LATENTSYNC_STEPS=20
```
≈ 154 min per 1.8 s clip. mean 2.05 px, clean drift.

**Preview** (draft timing, accept quantization):
```
LATENTSYNC_IPEX_DTYPE=bf16
LATENTSYNC_STEPS=20
```
Rest at defaults. ≈ 50 min per 1.8 s clip. Visible bouncing on static subjects.

**Lightning smoke-test** (wiring check only, output unusable):
```
LATENTSYNC_DRY_RUN=1
```
Collapses denoise to 1 step. ≈ 5 min per clip; confirms VAE/ffmpeg/mask pipeline holds.

**Quality-lossless reference** (for jitter A/B work, huge temp files):
```
ENABLE_VIDEO_STABILIZATION=false
LATENTSYNC_LOSSLESS_READ=1
LATENTSYNC_IPEX_DTYPE=fp32
```
Provably pipeline-internal jitter isolated from input-side noise.

### 7.3 Memory

At 1080p, peak RSS grows linearly with clip length because
`video_frames` and `restore_img` intermediates are held in RAM:

| Clip length | Measured peak RSS |
|---|---|
| 1.77 s (62 frames) | ~15 GB |
| 30 s (750 frames) | ~20 GB |
| 60 s (1500 frames) | ~25 GB |
| 4K (anything) | multiply by ~4 |

Docker memory cap is set to 128 GB in `docker-compose.yml` to stay
comfortably above any realistic clip. If you're running on a smaller
host, cap input duration rather than reducing memory — the pipeline
doesn't have a streaming path.

---

## 8. Upstream diff

We run `bytedance/LatentSync@a229c39` with six intentional patches.
Full audit in `scripts/latentsync_debug/BYTEDANCE_AUDIT.md`. In brief:

1. `models/unet.py` — CUDA guard on `empty_cache()` (no-op on CPU)
2. `pipelines/lipsync_pipeline.py` — progress callbacks, resume cache,
   landmark smoother, affine smoother, UNet bypass hook, affine dump
3. `utils/affine_transform.py` — dtype default fp32; `smooth=False`
   at the caller; mask-bypass hook; debug dumps
4. `utils/face_detector.py` — CPU provider routing for InsightFace
5. `utils/image_processor.py` — CPU face-detection path (upstream raised
   `NotImplementedError`); accept pre-computed landmarks
6. `utils/util.py` — optional `LATENTSYNC_LOSSLESS_READ` re-encode path
7. `whisper/whisper/__init__.py` — CUDA guard on `empty_cache()`

Data-path and training-only code (`data/*`, `trepa/*`) are correctly
excluded from the vendoring.

---

## 9. See also

- `docs/lipsync.md` — top-level "which backend" doc, covers wav2lip and MuseTalk too
- `docs/architecture.md` — full pipeline architecture, all stages
- `docs/models.md` — model provenance, licenses, SHA-sums
- `scripts/latentsync_debug/DEBUG_PLAN.md` — the bisection plan (history)
- `scripts/latentsync_debug/BYTEDANCE_AUDIT.md` — upstream diff audit
