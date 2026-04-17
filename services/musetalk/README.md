# lipsync-musetalk

CPU MuseTalk lipsync microservice. Dep-isolated from the main backend because
MuseTalk upstream pins `transformers==4.39.2` while the main backend needs
`>=4.44` for coqui-tts.

## PR roadmap

- **PR 1a** — FastAPI scaffold. `/lipsync` returns `501 Not Implemented`
  so the HTTP + shared-volume wiring can be verified before heavy ML deps
  land.
- **PR 1b (this build)** — torch CPU + torchaudio + torchcodec,
  diffusers/transformers/accelerate pinned to MuseTalk's expected versions,
  face-alignment (replaces `mmpose`), librosa/soundfile, huggingface_hub,
  gdown for BiSeNet. Includes `scripts/download_models.sh` that fetches
  all required weights (~1.4 GB). `/ready` and `/weights` endpoints added
  for introspection. `/lipsync` still returns 501.
- **PR 1c** — vendor MuseTalk inference code; wire UNet forward + VAE
  decode + face-region blending; flip `INFERENCE_IMPLEMENTED = True`.

## Pre-fetching weights

```bash
docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh
```

Weights land under the shared `/models/musetalk/` volume:

| Path                                            | Source                            | Size    | License           |
|-------------------------------------------------|-----------------------------------|---------|-------------------|
| `musetalkV15/unet.pth`                          | TMElyralab/MuseTalk (HF)          | ~900 MB | CC-BY-NC 4.0      |
| `sd-vae/diffusion_pytorch_model.bin`            | stabilityai/sd-vae-ft-mse (HF)    | ~330 MB | CreativeML Open RAIL-M |
| `whisper/pytorch_model.bin`                     | openai/whisper-tiny (HF)          | ~75 MB  | MIT               |
| `face-parse-bisent/79999_iter.pth`              | zllrunning/face-parsing (GDrive)  | ~52 MB  | MIT               |
| `face-parse-bisent/resnet18-5c106cde.pth`       | pytorch.org                       | ~46 MB  | BSD               |

## Introspection

```bash
curl http://localhost:8089/ready     # are all Python ML deps importable?
curl http://localhost:8089/weights   # are all weight files present & plausibly sized?
```

## Contract

```
POST /lipsync
{
  "video_path":  "/jobs/<job_id>/input.mp4",
  "audio_path":  "/jobs/<job_id>/translated_audio.wav",
  "output_path": "/jobs/<job_id>/lipsynced.mp4"
}

→ 200 { "status": "ok", "output_path": "...", "frames_processed": N, "duration_ms": N }
→ 501 { "phase": "not-implemented", ... }         # PR 1a/1b
→ 400 { "detail": "..." }                         # missing inputs
→ 500 { "detail": "..." }                         # inference failure
```

Paths must live under the shared `/jobs` volume (also mounted by the main
backend). Bytes don't travel over HTTP.
