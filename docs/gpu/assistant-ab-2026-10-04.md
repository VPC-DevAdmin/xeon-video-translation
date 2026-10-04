# Video assistant renderer A/B: FlashHead vs LatentSync persona mode

Date: 4 October 2026, XE7740, `gpu/track` at `e1ff7a1`. Raw reports, GPU
telemetry and scores: `artifacts/bench/assistant-ab-2026-10-04/`
(`summary.md` is the generated table). Output videos on the box:
`/jobs/ab/out/` (LatentSync container) and locally under
`artifacts/review/assistant-ab/`.

## What was built and deployed

**Design A, generated motion: FlashHead from a portrait.** One GPU. The
1.3B streaming model generates the whole frame (head, eyes, background)
from a still portrait and the audio, 28 new frames per 1.12 s audio chunk
at 512×512. `experiments/flashhead/stream_render.py` drives it chunk by
chunk the way a session would and timestamps every chunk. Variants: Pro
compiled, Pro eager, Lite. Runs in the lab container on GPU 2.

**Design B, real footage: LatentSync persona mode.** All eight GPUs. The
mouth is regenerated on the person's own footage; everything else is the
recording. New in the service (`e7d79f2`): each worker keeps the
audio-independent conditioning of a named persona clip resident (mask,
masked-face and reference VAE latents per 16-frame chunk), so a repeated
reply skips the per-chunk VAE encodes; every chunk is timestamped at
submit, result and paste-back; `/lipsync` takes `persona_key`. Variants:
10 and 20 steps; cold persona (first contact with the footage) and
resident.

Both designs rendered the same two audio samples: the 13.5 s conversational
reply (`quality-v4/master.wav`) and the 59.5 s narration. Same person in
the portrait and the footage.

## Speed

Playout at 25 fps (the renderers' native rate). "Head start" is the smallest
`H` for which playout starting `H` seconds after the first chunk was
submitted never stalls; `r` is rendered fps over 25. The LatentSync numbers
exclude the per-reply `prepare` call (decode and warp of the footage,
13 s for the 20 s clip, 42 s for the 52 s clip): in a persona design that
work is resident, in this trial it ran before the timed call. The cold rows
show what happens without it.

| Run | GPUs | First chunk | Head start, 14 s reply | Head start, 60 s reply | Aggregate fps | r | Stalls at H=10 / 20 (60 s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FlashHead Lite | 1 | 0.21 s | 0.2 s | 0.2 s | 115 | 4.6 | 0 / 0 |
| FlashHead Pro, compiled | 1 | 1.28 s | 3.3 s | 10.5 s | 21.7 | 0.87 | 3 / 0 |
| FlashHead Pro, eager | 1 | 1.54 s | 6.7 s | (not run) | 18.1 | 0.72 | |
| LatentSync 10 steps, resident persona | 8 | 4.1 s | 5.4 s | 11.9 s | 23.1 / 25.7 | 0.93 / 1.03 | 18 / 0 |
| LatentSync 20 steps, resident persona | 8 | 6.9 s | 9.5 s | 23.6 s | 16.6 / 18.5 | 0.66 / 0.74 | 77 / 25 |
| LatentSync 10 steps, cold persona | 8 | 18.1 s (20 s clip), 46.9 s (52 s clip) | 19.5 s from call | 56.1 s from call | 21–24 | | |

Per chunk: FlashHead Pro 1.28 s per 1.12 s of audio, p95 equal to the mean.
LatentSync worker 3.07 s per 16-frame chunk at 10 steps (5.8 s at 20),
all 94 chunks of the 60 s run served from the persona cache. The LatentSync
aggregate is capped by paste-back, not denoise: results land 3.3 s after
submit for the first eight chunks, then the single restore thread emits a
chunk every 0.59 s (0.64 s of playout), so r sits at 1.0 while the eight
workers could denoise 41 fps (r 1.65). Moving the paste into the workers,
the change the profile already named, would put LatentSync at r about 1.6
with a 4.5 s head start.

GPU telemetry during the renders (mean SM, power of busy cards):

| Run | Busy GPUs | SM | Power | Peak memory |
| --- | --- | ---: | ---: | ---: |
| FlashHead Lite | GPU 2 | 81% | 400 W | 23 GB (5.4 GB torch) |
| FlashHead Pro, compiled | GPU 2 | 98% | 530 W | 28 GB (9.7 GB torch) |
| LatentSync 10 steps, resident | all 8 | 31–60% | 2.4–2.7 kW | 17 GB per worker, unchanged by the cache |
| LatentSync 20 steps, resident | all 8 | 42–80% | 3.1–3.5 kW | same |

Warm-up: FlashHead Pro compiled 23 s with the compile cache (150 s the first
time), Lite 0.6 s. LatentSync pool and ONNX sessions 45 s at service start.

## Quality

Scorer: `scripts/ab/score_quality.py`, run on every output and on the real
footage with its own audio as a reference row. Identity is ArcFace cosine
(insightface w600k_r50); sharpness is Laplacian variance of the 256 px
face crop and its mouth region; temporal is mean absolute difference of
consecutive face crops; sync proxy is the peak correlation of MediaPipe
mouth openness with the audio envelope; SyncNet is LatentSync's
StableSyncNet cosine between 16-frame lower-face windows and their mel
windows (offset 0 was the best offset for every video).

| Run | Faces | Identity vs portrait | Identity vs footage | Sharpness face / mouth | Temporal | Flicker | Sync proxy r | SyncNet cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real footage (reference) | 750/750 | 0.929 | 0.970 | 497 / 533 | 3.97 | 0.57 | 0.09 | 0.19 |
| FlashHead Lite, 14 s | 339/339 | 0.937 | 0.929 | 202 / 127 | 3.20 | 0.55 | 0.25 | 0.67 |
| FlashHead Pro compiled, 14 s | 339/339 | 0.923 | 0.917 | 305 / 268 | 3.96 | 0.78 | 0.46 | 0.75 |
| FlashHead Pro compiled, 60 s | 750/750 | 0.924 | 0.919 | 315 / 282 | 3.98 | 0.78 | 0.41 | 0.81 |
| LatentSync 10 steps, 14 s | 340/340 | 0.928 | 0.959 | 291 / 149 | 3.30 | 0.75 | 0.57 | 0.87 |
| LatentSync 10 steps, 60 s | 750/750 | 0.928 | 0.959 | 293 / 153 | 3.33 | 0.88 | 0.55 | 0.90 |
| LatentSync 20 steps, 14 s | 340/340 | 0.928 | 0.960 | 294 / 157 | 3.36 | 0.76 | 0.59 | 0.87 |
| LatentSync 20 steps, 60 s | 750/750 | 0.928 | 0.960 | 296 / 161 | 3.39 | 0.91 | 0.55 | 0.89 |

The cold and resident LatentSync runs produced identical outputs at the
same seed: the cache changes timing, not pixels.

How to read the sync columns: the real recording scores low on both
(noisy car-cabin audio, small natural mouth motion), and SyncNet-supervised
models articulate in a way SyncNet rewards, so these are relative measures
between renders, not absolute truth. Within that, LatentSync syncs best,
FlashHead Pro next, Lite clearly weakest, and the contact sheets agree.

What the frames show (sheets in `artifacts/bench/assistant-ab-2026-10-04/scores/*.sheet.png`):

- **Nothing is mangled.** Every frame of every output has a detected face
  and identity cosine above 0.89; no frame drops, durations match the audio.
- **LatentSync** is the real recording with a new mouth: head motion,
  eyes, lighting and background are the footage. The mouth is visibly
  softer than the original (Laplacian 149 vs 533) and slightly smoothed,
  which is the model's signature; 20 steps does not sharpen it or change
  sync, so 10 steps is the setting. Head motion is uncorrelated with the
  words because it comes from the footage.
- **FlashHead Pro** generates plausible head and eye motion, blinks, and a
  more articulated mouth (sharper than LatentSync's, 268 vs 149), at 512 px
  with a waxier skin texture and a background that wobbles with the head.
  Some sampled frames catch the eyes closed; whether those are blinks or
  held closures needs a watch. Identity against the footage is lower
  (0.92 vs 0.96) since every pixel is generated.
- **FlashHead Lite** is soft (face 202, mouth 127) with weak articulation;
  its higher identity-vs-portrait score is the smoothness, not fidelity.
  Fast enough for anything, not good enough for this product.

## Decision

For a single assistant session where quality is the top priority,
**LatentSync at 10 steps with a resident persona** is the renderer: the
best sync, the person's real pixels everywhere but the mouth, and a
measured head start of 5 s for a 14 s reply and 12 s for a 60 s reply,
inside the 10 to 20 s contract. Its costs are the whole box (2.4 to 2.7
kW for one session) and a softer mouth than the recording.

**FlashHead Pro compiled is the one-GPU alternative**: 3 s and 10.5 s head
starts, generated head and eye motion, seven GPUs left for other sessions
at a fifth of the power. It loses on identity fidelity and texture.

Not candidates after this trial: FlashHead Lite (quality) and LatentSync
at 20 steps (no visible gain for 1.5× the time).

## What the trial does not settle

- Both renderers were driven offline from complete audio. The streaming
  session core from the architecture review (audio timeline in, frames
  out, deep buffer, deadline scheduler) is still to build; these numbers
  bound what it can deliver.
- Per-reply `prepare` must become part of persona capture for LatentSync:
  resident frames, faces and affines on the coordinator, not a 13 to 42 s
  call before each reply. The cold rows are what a new persona costs once.
- Paste-back on the workers is the next LatentSync lever (r 1.0 to about
  1.6); FlashHead Pro's levers are SageAttention or FlashAttention kernels
  and the two-GPU path, both unmeasured.
- Chunk-boundary seams (LatentSync) and eye closures (FlashHead) need a
  real-time watch of the videos, not sampled frames.
- SyncNet absolute values are not calibrated against human judgement here.

## Reproduce

```bash
# FlashHead, inside the lab container (output under /experiment/ab-out)
docker exec -d polyglot-flashhead-lab bash /experiment/flashhead_batch.sh

# LatentSync persona mode, on the box host
python3 scripts/ab/run_latentsync.py --label ls10-14s-resident \
  --video /jobs/ab/inputs/persona20s-25fps.mp4 --audio /jobs/ab/inputs/reply14-16k.wav \
  --master /jobs/ab/inputs/reply14-master24k.wav --persona persona20 --steps 10 \
  --face-track --prepare --repeat 2

# Scores, inside the LatentSync container; then the summary locally
python /app/repo_scripts/ab/score_quality.py --video /jobs/ab/out/X.mp4 --audio ... --reference ... --label X
python scripts/ab/analyze.py artifacts/bench/assistant-ab-2026-10-04
```
