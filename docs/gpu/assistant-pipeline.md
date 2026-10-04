# Video assistant pipeline (live, WebRTC)

Date: 4 October 2026. Code: `gpu/track` from `333e869` (pipeline) and `79dfb41`
(personas). Measurements: `artifacts/bench/assistant-ab-2026-10-04/assistant-e2e-*.json`.

## Try it live

On your machine, tunnel the frontend (the box binds it to localhost only, and the
browser needs a secure origin for the microphone and camera):

```bash
ssh -L 3030:localhost:3030 user@xe7740
```

Open http://localhost:3030/assistant. The page shows who will answer (the most
recent persona is preselected) and one button, **Start chat**. It turns on your
microphone (and camera for a small self-view; the camera is not sent anywhere),
and the person's face appears in the stage with a 12 s idle loop as soon as the
call connects, in about 3 s the first time a persona is used and under a second
after that. Speak; the assistant acknowledges at once, shows "Thinking… answer in
N s", and the reply video starts after the head start. Speak again or press
Interrupt to cut it off. **Use this person…** opens the guided capture for a new
face and voice; "or upload a portrait" keeps the older portrait-only path.

Switch the box into this mode with `scripts/mode.sh assistant` (stops LatentSync
and MuseTalk; starts backend, LLM, frontend, ingest, TURN relay; reuses a running
renderer).

Media path: browsers reach the box over the overlay network, where direct UDP
between host candidates is not guaranteed and Chrome hides its own addresses
behind mDNS, so assistant mode runs a coturn relay on the box (port 3478 UDP and
TCP, relay ports 49160 to 49200). Ingest hands both peers STUN and TURN servers
with credentials derived from `TURN_SHARED_SECRET`. The `.env` values are JSON
and must be single-quoted so Compose keeps the quotes:

```
TURN_SHARED_SECRET=<random>
TURN_PUBLIC_IP=100.67.151.209
TURN_URLS_JSON='["turn:100.67.151.209:3478?transport=udp","turn:100.67.151.209:3478?transport=tcp"]'
ICE_SERVERS_JSON='[{"urls":["stun:100.67.151.209:3478"]}]'
```

The page sends its offer after at most 4 s of candidate gathering, so an
unreachable server entry cannot stall the call. If UDP to the box is blocked
entirely, add `"turn:localhost:3478?transport=tcp"` to `TURN_URLS_JSON`, add
`-L 3478:localhost:3478` to the SSH command, and the browser relays over TCP
through the tunnel.

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
| Render service | `services/flashhead/app` | FlashHead Pro compiled, one GPU; sessions hold motion state; PCM in, raw RGB frames out; `/health` reports chunk spec and timings; `/portrait/enhance` (CodeFormer) and `/portrait/pose` (LivePortrait head pose, gaze, head-turn frames) |
| Turn endpoints | `backend/app/api/assistant.py` | plan-then-speak; `/assistant/speak` for the acknowledgement and previews; `persona_id` selects a cloned voice |
| Session | `services/ingest-webrtc/app/assistant.py`, `timeline.py` | timeline with future clips, idle loop, promises and stall counting; deadline rule in `head_start_required` |
| Personas | `backend/app/api/personas.py`, `frontend/app/components/PersonaWizard.tsx` | capture, checks, XTTS conditioning, preview |
| Mode | `docker-compose.assistant.yml`, `scripts/mode.sh assistant` | renderer runs in the FlashHead lab container today (see below) |

## Filling the wait

The gap between the end of speech and the reply is planned the way a person
looking something up behaves (`Assistant.fill_gap` in the ingest service):

1. An opener plays at once ("Hold on, let me find that for you."; five in
   English, rotated so the same one is not used twice in a row).
2. The head turns to a tablet (LivePortrait edit of the portrait: pitch 14,
   yaw -14, gaze lowered; `ASSISTANT_WORKING_POSE`, where a negative eyes_y
   lowers the gaze), and the working idle
   loop shows the persona reading. Short progress utterances ("Hmm, let me
   see.", "Okay, almost there.") and the occasional longer bridge ("Bear with
   me, I want to make sure I get this right.") play with 0.7 to 1.5 s pauses,
   scheduled just in time so the plan can adapt.
3. The reply start is decided as soon as the first reply chunk is rendered: the
   earliest moment the reply can play without stalling given the measured
   render rate, not before `ASSISTANT_HEAD_START` (7 s) and not after the cap.
   Whatever filler is still being said is cut off with an 80 ms fade, the head
   turns back, a short closer plays facing the user ("Okay, so."), and the
   reply starts. The reply prompt tells the model the listener heard a look-up
   phrase, so it begins with the answer.

Phrases live in `FILLERS` (English, Spanish, French, German; other languages
have openers only) and can be overridden per kind and language with
`ASSISTANT_<OPENER|BEAT|BRIDGE|CLOSER>_TEXT_<LANG>` ("|"-separated). Every
phrase has at least three words because XTTS babbles on one-word prompts; the
verifier aligns on canonical tokens so "I'm" for "I am" is not a retake.

## Measured (ingest-side WebRTC client, same box, Oct 4)

Session start: the renderer session opens in 0.2 s; the first time a persona is
used, two chunks of idle motion are rendered before the session answers (3.0 to
3.5 s). In the background the first acknowledgement is synthesized and rendered
(about 5 s), the idle loop is grown to 12.3 s, then two more acknowledgements
are rendered (everything ready 40 to 42 s after start when a turn happens in
between); all of it is cached under the persona, and later sessions answer in
0.4 to 0.9 s with everything ready. The idle loop plays forward over continuous
segments and dissolves across every boundary and the wrap.

Cold-session turn at 8.7 s after start (uploaded-photo persona, stock voice,
`assistant-e2e-persona-cold-ack-first.json`): acknowledgement audible 0.20 s after
the end of speech, reply playing at 10.0 s, 122 frames at 25.0 fps, 0 stalls, 3
idle segments. Idle frame-to-frame change (mean absolute pixel difference) before
the turn: median 0.34, max 2.6; after the reply, while the loop grew from 5 s to
12.3 s: median 0.77, max 3.0, no jump above 6 (the first version of the growing
loop showed jumps of 8 to 11 when footage was appended). Warm session
(`assistant-e2e-persona-warm-segments.json`): ready at channel open,
acknowledgement 0.17 s, 25.0 fps, 0 stalls, max idle change 2.7 after the reply.

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
- Filler clips and the reply are separate generations joined through a common
  anchor: each starts from the renderer's rest pose (every render begins with
  a reset) and, after its trailing silence, settles into that rest frame over
  10 frames, so clips, the idle loop and the reply's end join with cuts, not
  dissolves between two poses. A cut-off filler settles before the head turns
  back. Only joins that cannot be anchored (an opener landing mid-idle, the
  head-turn footage) get a 4-frame dissolve.
- The front idle loop is one continuous take (12 s) cut at the frame that best
  matches its first frame, so the wrap is a natural continuation with a
  two-frame blend and the loop never returns to centre. While the take is
  still growing (first use of a persona) the wrap crossfades. The working idle
  is a LivePortrait reading loop of the posed portrait (eyes scanning the
  page, a line down and back, slight head drift, two blinks), periodic by
  construction; frame 0 is the pose the 12-frame head turn arrives at.
- Reply sentences end with a 20 ms fade and a 0.45 s pause (0.65 s after a
  question), so separately verified sentences sound like one stream.
- Background preparation order is the first opener (about 5 s after the face
  appears), the front idle loop to 12 s, the posed portrait plus turn footage,
  the working idle loop to 6 s, then beats, closers and bridges, then the rest
  of the repertoire (20 clips in English). Cold, with one turn in between, this
  takes about 100 s; everything is cached per persona. All renderer calls of a
  session go through one lock, and the renderer records which footage its
  motion state continues, so idle growth extends the current segment only when
  nothing else rendered in between.
- Cached footage is keyed by the uploaded portrait, the restoration settings,
  the renderer model and chunk spec, the frame rate, the working pose and (for
  clips) the voice conditioning; one idle loop per pose, one turn sequence and
  one clip set are kept per persona, and a changed setting evicts the rest.
- Speech verification keeps a take only when at least 90 percent of the
  sentence's words are recognized; a cloned voice that fails three takes is
  replaced by the stock voice for that sentence, and the UI shows a note.
- One assistant session per box (`ASSISTANT_MAX_SESSIONS`); the renderer holds
  one portrait at a time and re-prepares on switch (0.2 s); an assistant session
  uses two renderer sessions (front and working pose).
- LivePortrait lives in the lab container (`/experiment/LivePortrait`, weights
  under `/experiment-models/liveportrait`, a `gradio` stand-in module in
  `/experiment/liveportrait_shim`); it belongs in the FlashHead Dockerfile.
- The idle clip is captured and stored but not yet used by any renderer.
- The minimum head start is 7 s (`ASSISTANT_HEAD_START`, compose); the filler
  plan fills whatever the wait is, up to the 20 s cap.
