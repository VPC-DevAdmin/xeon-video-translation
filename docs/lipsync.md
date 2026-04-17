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

## `musetalk` — stubbed

Would be the right quality/speed tradeoff if it weren't for installation
weight. Pulling in MuseTalk involves:
- Whisper encoder (we already have it)
- Face parsing via BiSeNet
- VAE + diffusion UNet (~2 GB of weights)
- Its own audio-visual alignment code

Selecting `musetalk` today raises:
```
LipsyncError: MuseTalk is not yet wired up in this build.
```

Roadmap: dedicated follow-up PR. Open an issue if you want to prioritize it.

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
