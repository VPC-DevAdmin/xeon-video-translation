# Video assistant pipeline (live, WebRTC)

Date: 4 October 2026. Code: `gpu/track` from `333e869` (pipeline) and `79dfb41`
(personas). Measurements: `artifacts/bench/assistant-ab-2026-10-04/assistant-e2e-*.json`.

## Try it live

On your machine, tunnel the frontend (the box binds it to localhost only, and the
browser needs a secure origin for the microphone and camera):

```bash
ssh -L 3030:localhost:3030 user@xe7740
```

Then open http://localhost:3030/assistant. Signalling goes through the tunnel;
audio and video flow directly between the browser and the box over the LAN.

- **Portrait upload + bundled voice**: pick an image, Start conversation, speak.
- **Use this person…**: the guided capture (consent, portrait, 8 s idle clip,
  scripted voice recording, checks, voice preview), then Start.

Switch the box into this mode with `scripts/mode.sh assistant` (stops LatentSync
and MuseTalk; starts backend, LLM, frontend, ingest; reuses a running renderer).

## What happens on a turn

```
mic ──WebRTC──▶ ingest (Silero VAD, 0.5 s endpoint)
   t0  ├─ acknowledgement clip (pre-rendered at session start) plays at once
       ├─ backend /assistant/respond: whisper transcript → whole reply text (vLLM) → XTTS audio as PCM events
       ├─ FlashHead renders 28-frame chunks as audio arrives (1.12 s of video per ~1.3 s)
       ├─ reply start chosen once: max(default head start 10 s, first chunk + L·(1−r) + 1 s), ≤ 20 s
       ├─ idle motion (rendered from silence) fills the gap on the same timeline
   t0+H└─ reply plays at 25 fps; stalls counted; speaking again or Interrupt clears all and resets the renderer
```

Components:

| Piece | Where | Notes |
| --- | --- | --- |
| Render service | `services/flashhead/app` | FlashHead Pro compiled, one GPU; sessions hold motion state; PCM in, raw RGB frames out; `/health` reports chunk spec and timings |
| Turn endpoints | `backend/app/api/assistant.py` | plan-then-speak; `/assistant/speak` for the acknowledgement and previews; `persona_id` selects a cloned voice |
| Session | `services/ingest-webrtc/app/assistant.py`, `timeline.py` | timeline with future clips, idle loop, promises and stall counting; deadline rule in `head_start_required` |
| Personas | `backend/app/api/personas.py`, `frontend/app/components/PersonaWizard.tsx` | capture, checks, XTTS conditioning, preview |
| Mode | `docker-compose.assistant.yml`, `scripts/mode.sh assistant` | renderer runs in the FlashHead lab container today (see below) |

## Measured (ingest-side WebRTC client, same box, Oct 4)

Session preparation (renderer session, idle loop from 2.2 s of silence, acknowledgement TTS and render): 6.3 to 7.6 s.

| From the endpoint | Bundled voice, turn 1 | Bundled voice, turn 2 (interrupted) | Cloned persona, turn 1 | Cloned persona, turn 2 (interrupted) |
| --- | ---: | ---: | ---: | ---: |
| acknowledgement audible | 0.19 s | | 0.20 s | |
| transcript | 0.43 s | 0.42 s | 0.40 s | 0.39 s |
| full reply text | 0.59 s | 0.55 s | 0.54 s | 0.53 s |
| first reply chunk rendered | 3.8 s | 3.7 s | 3.8 s | 3.8 s |
| reply playing | 10.0 s (default head start) | 10.0 s | 10.0 s | 10.0 s |
| reply video received | | | 25.0 fps, 95 frames | |
| reply length | 3.3 s | 3.2 s | 3.7 s | 3.6 s |
| stalled frames | 0 | | 0 | |
| interrupt → "listening" | | 1 ms | | 4 ms |
| interrupt → last audio packet | | 88 ms | | 99 ms |

Render rate measured in-session: 0.83 to 0.84 (1.31 to 1.35 s per 1.12 s chunk,
first chunk 1.5 s), so a 3 s reply needs a 3.1 s head start and the default
10 s is what the user sees. Video sent at 23.5 fps average over the whole session
(idle included), 3 skipped frames in 185 s.

First turn after a backend restart previously paid 9.5 s in the transcript;
the assistant overlay now runs the speech warm-up at start (`AVATAR_WARMUP_INFERENCE`)
and the first turn is as fast as the others.

Persona build from fixtures (66 s recording, 512 px portrait): 1.6 s including
ASR of the recording and the XTTS conditioning; all checks passed; voice preview
4.2 s of audio in 1.6 s.

## Persona capture protocol

1. **Consent**: fixed text, version recorded with the persona.
2. **Portrait**: webcam, face inside an oval guide, even light, neutral
   expression. Server checks with OpenCV's frontal-face cascade: exactly one face,
   face height 22 to 75 % of the frame, centred within 20 %, Laplacian sharpness
   ≥ 40, mean brightness 55 to 215.
3. **Idle clip** (optional): 8 s of sitting still, looking at the camera, not
   talking; stored as 25 fps MP4 for the footage-based renderer later.
4. **Voice**: a fixed per-language script (~25 s, pangram, numbers, a
   conversational line). Server checks: ≥ 15 s, level −38 to −6 dBFS, clipping
   < 0.5 %, ≥ 35 % voiced blocks, and whisper's transcript must match ≥ 60 % of
   the script's words in order.
5. **Build**: XTTS `get_conditioning_latents` on the 24 kHz master → `voice.pt`;
   `persona.json` with checks, consent and file paths under
   `/jobs/personas/<id>/`. Rejected captures are deleted and the wizard shows
   which part to retake.
6. **Preview**: a fixed sentence in the cloned voice; then "Use this persona".

## Known limits and next steps

- The render service runs inside the FlashHead lab container (`docker exec`),
  reachable from ingest at the lab's bridge address (`ASSISTANT_RENDERER_URL`
  in the box `.env`). `docker commit` of the lab fails on a missing base-layer
  digest, so a reproducible image needs a Dockerfile built from the lab's
  recorded requirements; `docker-compose.assistant.yml` already expects it.
- FlashHead renders 512×512 from the portrait; expression follows the audio,
  texture is softer than a recording. A restoration or upscale pass is the
  quality lever; SageAttention or FlashAttention the speed lever (r 0.84 today).
- Acknowledgement, idle and reply are separate generations, so there is a cut
  between them. Cross-fading or carrying motion state across them is the next
  smoothness item.
- One assistant session per box (`ASSISTANT_MAX_SESSIONS`); the renderer holds
  one portrait at a time and re-prepares on switch (0.2 s).
- The idle clip is captured and stored but not yet used by any renderer.
- Head start is fixed at 10 s minimum by design; replies longer than about
  60 s at r 0.84 would push it toward the 20 s cap.
