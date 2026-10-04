# Video assistant at 24 fps: architecture review and pathways

Date: 4 October 2026. Code reviewed: `gpu/track` at `403fefd`. Numbers are
the measurements recorded in `profile-2026-10-04.md`,
`xe7740-validation-2026-10-03.md` and `experiments/flashhead/`; anything
not measured on the box is marked as such.

## The new contract

- `t0`: the user stops speaking.
- A pre-rendered acknowledgement (face and voice) starts within about a second.
- The reply video starts at `t0 + H`, with `H` between 10 and 20 s, plays at
  24 fps, and never stalls until the reply ends.
- Interruption still cuts audio and video together.

## What the head start changes

The renderer no longer has to be faster than real time. A reply of `L`
seconds whose rendering starts at `t0` and plays from `t0 + H` never stalls
when the render keeps up with playout: `L * (1 - r) <= H`, where `r` is
rendered fps divided by 24. So `r >= L / (L + H)`.

| Reply length `L` | `H = 10 s` | `H = 20 s` |
| --- | ---: | ---: |
| 15 s | r ≥ 0.60 (14.4 fps) | r ≥ 0.43 (10.3 fps) |
| 30 s | r ≥ 0.75 (18.0 fps) | r ≥ 0.60 (14.4 fps) |
| 60 s | r ≥ 0.86 (20.6 fps) | r ≥ 0.75 (18.0 fps) |

Two further consequences:

- **First-chunk latency stops mattering.** A chunk may take several seconds
  as long as the first one lands before `H`.
- **Plan, then speak.** The whole reply text can be generated before any
  audio is synthesized (160 tokens from vLLM is a second or two, to be
  measured). TTS then works sentence by sentence with the full text in hand,
  and the scheduler knows `L` before rendering starts, so it can choose
  quality per reply instead of per fixed setting.

Replies are bounded today: `backend/app/api/avatar.py` caps the LLM at
160 tokens, roughly 25 to 30 s of speech.

## The current path, measured against that contract

The chain is `services/ingest-webrtc/app/avatar.py` (WebRTC, VAD, turn
loop, playout) calling `backend/app/api/avatar.py` (`/avatar/respond`:
whisper, LLM stream, XTTS stream) and `services/musetalk/app/avatar.py`
(`/avatar/render` per audio chunk), with `services/ingest-webrtc/app/playback.py`
pacing audio and video on one clock.

1. **The unit of work is a 0.5 s audio file.** XTTS emits 12 000 samples,
   the backend writes a WAV, ingest reads it, resamples to 48 kHz, writes a
   second WAV with a 0.2 s tail for context, posts a path to MuseTalk, which
   recomputes whisper features from that file, renders, and writes an `.npy`
   that ingest loads. Four file handoffs and one HTTP round trip per half
   second. The ingest loop is serial: it awaits each render before touching
   the next chunk, with a 5 s timeout that falls back to a still image. Any
   renderer slower than real time stalls this loop by construction.
2. **The buffer policy is the opposite of what we need.** `Playback.enqueue`
   blocks while more than 2 s of media is queued. That was right for a 1 s
   latency target; it forbids the 10 to 20 s of render-ahead that the new
   contract is built on. Output is hard-coded to 25 fps and frames are
   chosen by wall clock, which is fine, but the idle state is a frozen
   portrait.
3. **The renderer has no state between calls.** MuseTalk is single-step and
   well over 100 fps, which is why the design above works at all. Each call
   is independent: a 3-frame cross-fade hides the seam, there is no head
   motion, and the mouth-patch smoothing was judged unacceptable in review.
4. **Frames cross host memory twice.** `.npy` from the renderer, then
   `bgr24` NumPy into PyAV and NVENC. Fine at 512 px; at 720p or 1080p and
   24 fps it is per-frame Python work on the asyncio loop and a jitter
   source rather than a throughput wall. The PyNvVideoCodec probe (286 fps
   decode to CUDA tensors) shows the GPU-resident path exists.
5. **Rendering is single-GPU by design.** Assistant layout: speech on GPU 0,
   LLM on GPU 1, MuseTalk and WebRTC codecs on GPU 2, five cards idle. The
   LatentSync multi-GPU pool is reachable only through the batch job API
   (`/lipsync`, whole windows, files in and out).
6. **Endpointing** is a 600 ms silence endpoint with an energy VAD by
   default (Silero is wired but optional). Not a problem for a 10 s budget,
   but barge-in quality matters more when an interrupt throws away 20 s of
   rendered video.
7. **Worth keeping.** One monotonic clock for audio and video, a generation
   counter for interruption, warm-up gating before admission, ownership
   checks, bounded session storage, and a measured 1.1 s end of speech to
   audible reply.

The current design is right for what it targeted. For the new contract it
is the wrong shape in four places: the unit of work, the buffer policy, the
renderer's statelessness and the single-GPU render path. Raising the
timeout and the buffer cap to bolt a heavier renderer onto the per-chunk
loop would give stalls at every chunk boundary, not sustained 24 fps.

## Renderer candidates against the contract

| Renderer | Measured on this box | GPUs per session | First chunk | Motion | Detail | `r` at 24 fps |
| --- | --- | ---: | ---: | --- | --- | ---: |
| MuseTalk 1.5 (shipped) | >100 fps at 512 px | 1 | ~0.4 s | none (still portrait) | mouth patch rejected in review | >4 |
| FlashHead Lite | 114–117 fps, 24-frame chunks | 1 | 0.39 s | generated head and expression | face visibly softened; identity unreviewed | ~4.8 |
| FlashHead Pro | 18 fps eager, 21.9 fps compiled (150 s compile) | 1 | ~1.1 s | generated | made the 60 s sample; 512 px; expression follows sound | 0.91 |
| LatentSync 1.6 on persona footage, 10 steps | 27–30 fps aggregate on 8 GPUs | 8 | 4–5 s | real footage, uncorrelated with speech | real pixels everywhere but the mouth | 1.1–1.2 |
| LatentSync, 20 steps | ~14 fps aggregate (projected from 40-step timing) | 8 | ~9 s | as above | as above | 0.58 |
| LatentSync, 40 steps | ~7 fps aggregate | 8 | ~17 s | as above | as above | 0.29 |
| InfiniteTalk / Wan 2.1 14B | no complete video yet; far below real time | 4 | minutes | generated | unknown here | ≪0.1 |
| LiveAvatar 14B | not tried; authors claim streaming on 5×80 GB with timestep pipelining over NVLink | 5+ | unknown | generated | unknown | unknown |

Maximum reply length that never stalls, `L <= H * r / (1 - r)`:

| Renderer | `H = 10 s` | `H = 20 s` |
| --- | ---: | ---: |
| FlashHead Pro, 1 GPU (r 0.91) | ~100 s | ~200 s |
| LatentSync 10 steps, 8 GPUs (r ≥ 1.1) | unbounded | unbounded |
| LatentSync 20 steps, 8 GPUs (r 0.58) | 14 s | 28 s |
| LatentSync 40 steps, 8 GPUs (r 0.29) | 4 s | 8 s |

Two readings of that table:

- The head start makes **FlashHead Pro on a single GPU** sufficient for every
  assistant reply we allow today, with seven GPUs left over. Without the
  head start it missed the bar (21.9 < 24 fps).
- **LatentSync at 10 steps** is the only quality tier of that model that
  sustains, and it needs the whole box for one session. 20 steps works only
  for short replies; 40 steps is out for live use.

LatentSync's aggregate rate assumes its per-chunk conditioning is
precomputed. For a fixed persona clip the masked and reference VAE
latents, masks, faces and affine matrices do not depend on the audio, so
they can be built once per persona and kept resident on every worker. The
13.5 s "wait on workers" per 400 frames in the profile still included that
VAE work, so the resident-persona rate should be somewhat higher than 30
fps; that is a measurement to take, not a promise.

## Pathways

### Pathway 1: a streaming session core (renderer-independent, needed by every option)

This is the architectural change. Everything else plugs into it.

1. **Session-scoped renderer contract.** One long-lived render session per
   avatar, persona state loaded once. Audio PCM is pushed in with timeline
   positions; frames come back as GPU tensors or encoded access units with
   presentation timestamps; cancellation is a generation number. Transport
   is a stream (gRPC or WebSocket for control, shared memory or CUDA IPC for
   pixels; the LatentSync pool already moves tensors through `/dev/shm`).
   No files and no per-chunk HTTP.
2. **Plan, then speak.** The backend turn becomes: transcript, full reply
   text, then sentence-by-sentence TTS into one continuous 24 kHz reply
   timeline (native-rate master, 16 kHz conditioning copy, gap capping as
   in the generation path). The renderer consumes the timeline, not files.
3. **One timeline and a deadline scheduler.** The persona footage is the
   clock: frame index maps to wall time at 24 fps. The acknowledgement and
   idle footage occupy `[t0, t0 + H)`. `H` is chosen per reply as
   `max(H_min, L * (1 - r) + margin)` from the measured `r`. The renderer
   is told the start frame, so idle to reply is a cut on the same footage.
   Chunks carry playout deadlines; the step count (or model tier) is chosen
   from the slack. Late chunks are a logged failure, not a silent still.
4. **Playout.** Buffer target is `H`, not 2 s. Stall count, jitter and
   audio/video skew become first-class metrics. Output fps is a session
   parameter. Idle shows moving footage, never a frozen frame.
5. **GPU-resident frame path.** Renderer output goes to NVENC without a
   host round trip (PyNvVideoCodec and DLPack are already probed), or the
   renderer encodes and ships access units.
6. **Persona as an object.** Idle footage (two to three minutes of the
   person listening, cut at natural points rather than ping-pong looped),
   face track, precomputed conditioning, voice profile, and a small set of
   pre-rendered acknowledgements per language made with the same renderer
   on the same footage. This is the same persona the generation product
   needs.
7. **Interruption.** Discard everything rendered ahead; keep the 250 ms
   target. Silero VAD becomes the default so false endpoints do not waste
   20 s renders.

Size: large overall, each item medium. Ingest keeps the WebRTC edge and the
clock; the turn orchestration moves from a serial chunk loop into the
scheduler; the backend's `/avatar/respond` splits into plan and speak.

### Pathway 2: FlashHead Pro behind the session core

Why first: it is the only candidate with a measured `r >= 0.9` on one GPU,
it generates head motion and expression, it already produced the minute
sample, and it leaves seven GPUs for more sessions or a heavier face.

Work: a clean, reproducible environment (the lab container inherited the
LatentSync image and is not portable); an adapter that honours the session
contract and cancellation; compile at mode load (150 s); chunk continuity
across its 24-frame units; SageAttention or FlashAttention to clear 24 fps
with margin; the two-GPU xfuser path as a second experiment; persona
identity conditioning; an optional GPU face-restoration and upscale stage
to 720p whose cost must be measured against the budget; an A/B review
against MuseTalk and against LatentSync on the same audio.

Risks: softness and identity drift, expression that follows sound rather
than meaning, 512 px native. The four-step model's ceiling is what it is;
a persona LoRA is the lever if identity falls short.

Capacity: one GPU per session, so up to about six concurrent assistants in
this mode.

### Pathway 3: LatentSync on persona footage, streaming mode

Why: the review preferred its facial texture, and every pixel outside the
mouth is real footage. Its motion is real but not driven by the words.

Work beyond Pathway 1: precompute all source conditioning per persona and
keep it resident on each worker (`DenoisePool.begin_job` becomes a
session; a chunk is audio embeddings plus a frame index); move the paste
back into the worker that decoded the chunk, with the full-resolution
footage resident per worker (about 12 GB at 720p or 27 GB at 1080p for
three minutes, inside the 96 GB); encode from the worker GPU; causal audio
features with the model's 80 ms lookahead; steps chosen by deadline, 10 to
20.

Numbers: 1.1 to 1.2 times real time at 10 steps on all eight GPUs, one
session per box; six GPUs would be about 0.9. Multipliers to measure, each
of which could free cards for a second session: `torch.compile` with CUDA
graphs (the `reduce-overhead` path exists in `shard_workers.py` and is
unvalidated on GPU), guidance off (halves UNet work; quality to check), fp8
or TensorRT on Blackwell, and a distilled scheduler if one appears
upstream.

Risks: independent 16-frame chunks can seam at the mouth, which is today's
behaviour too; footage that loops reads as a loop, so the persona needs
long idle footage with natural cut points; idle frames must show a closed
mouth, either by choosing listening footage or by rendering silence.

### Pathway 4: later and research

- A persona LoRA for FlashHead identity, trained from the same footage the
  generation product captures.
- A multi-GPU streaming generator (LiveAvatar-class) as a time-boxed trial.
  Its published setup assumes NVLink; this box is PCIe. Record the result
  either way.
- Hybrids (generated motion with a diffusion mouth pass) are not worth
  building until one of Pathways 2 or 3 is live and reviewed.

### What not to do

- Raise the 5 s render timeout and the 2 s buffer cap and point the
  per-chunk loop at a heavier renderer. It stalls at each boundary and
  cannot use more than one GPU.
- Keep MuseTalk as the primary face. Speed is no longer the constraint, and
  quality was the reason it was rejected.

## Recommended order and decision points

1. **Pathway 1 with MuseTalk still rendering.** Behaviour stays as today
   while the session contract, timeline, deadline scheduler, deep buffer and
   GPU frame path are proven in a real browser: measure start time, stall
   count, jitter and skew at 24 fps with a 10 s buffer over TURN.
2. **Pathway 2 adapter.** A/B review of 30 s replies for identity, sync and
   softness. Decision: is FlashHead Pro the face, with or without
   restoration?
3. **Pathway 3** if the review wants real pixels, or in parallel if people
   are available; its persona precompute is shared with the generation
   product either way.
4. **Multipliers** (compile, guidance, attention kernels) once a renderer
   is chosen, to buy margin or sessions.

## Measurements to take first

Cheap, and each one changes a decision above.

- XTTS streaming real-time factor on a 30 s reply (whether TTS or the
  renderer paces the pipeline).
- vLLM time to a full 160-token reply.
- LatentSync single-chunk latency at 10 and 20 steps with precomputed
  conditioning; again with `reduce-overhead` compile; again with guidance
  off.
- FlashHead Pro with SageAttention or FlashAttention; the two-GPU path; the
  cost of a 720p restoration stage.
- Browser playout at 24 fps with a 10 s buffer: stalls, jitter, skew.
