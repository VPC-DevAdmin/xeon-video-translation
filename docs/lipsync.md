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
- **Stubble/detail preservation** — three blend modes exposed via
  `MUSETALK_BLEND_MODE`:
  - `raw`: full lower face replaced (original v1.0 behavior; lots of
    texture loss)
  - `jaw`: chin + cheeks + mouth (moderate; some stubble replacement)
  - `mouth`: lips + teeth only (default; best stubble preservation)
  Combined with a Gaussian feather controlled by `MUSETALK_BLEND_FEATHER`
  (fraction of crop width; default 0.08 ≈ 21 px on a 256-wide crop), the
  predicted mouth fades into original skin rather than replacing it.
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

## `latentsync` — stubbed

SD 1.5 latent-diffusion based. Best open-source quality as of writing. CPU
inference is **impractical**: estimated 30–60 minutes for a 3 s clip based on
per-frame latent diffusion step counts × typical CPU iteration times.

Selecting `latentsync` today raises:
```
LipsyncError: LatentSync is not yet wired up in this build.
```

Even after it's wired, you should not expect usable live-demo performance on
CPU. The stub exists so the UI and API are ready if and when we add a GPU
path.

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
