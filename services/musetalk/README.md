# lipsync-musetalk

CPU MuseTalk lipsync microservice. Dep-isolated from the main backend because
MuseTalk upstream pins `transformers==4.39.2` while the main backend needs
`>=4.44` for coqui-tts.

## PR roadmap

- **PR 1a (this build)** — FastAPI scaffold. `/lipsync` returns `501 Not
  Implemented` so the HTTP + shared-volume wiring can be verified before
  heavy ML deps land.
- **PR 1b** — torch/diffusers/einops/face-alignment. Weight downloads
  (MuseTalk UNet, SD 1.5 VAE, BiSeNet, Whisper encoder). Preprocessing
  (face detect → landmarks → crop) ported off `mmpose` to `face-alignment`.
- **PR 1c** — actual inference: UNet forward, VAE decode, face-region
  blending back into the source frames.

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
