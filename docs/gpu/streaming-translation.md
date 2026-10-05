# Streaming translation (mode `stream`)

Date: 5 October 2026. Code: `gpu/track` at `f5369f7`. Box: XE7740, one
LatentSync pool on eight GPUs (`scripts/mode.sh translate`).

## What it is

Translate an uploaded video and start showing the result while the rest renders,
touching as few pixels as possible. The whole-clip pipeline ran transcribe,
translate, TTS and lipsync as stages over the entire video, so nothing was
watchable until the last window had rendered. Mode `stream` keeps the shared
audio and transcribe stages and one whole-clip translate, then works per
**speech span**:

1. Transcript segments separated by less than 0.6 s are merged into spans and
   padded by 0.25 s (`stream_span_gap_seconds`, `stream_span_pad_seconds`).
2. The span's translated speech is synthesized with the existing per-segment TTS
   (slot fitting, whisper-verified takes, loudness) on the span's own clock, so
   each sentence lands at its original onset. The next span's speech is
   synthesized on the speech GPU while the current span renders on the pool.
3. Only the span's frames are rendered, in 8 s windows with 0.4 s of context on
   each side (`stream_window_seconds`, `window_overlap_seconds`), using the shared
   face track; the next window is cut and prepared while the current one
   denoises, as in the whole-clip renderer.
4. Each rendered window and each passthrough gap (source frames with the
   original sound) is muxed into an MPEG-TS segment and appended to an HLS
   EVENT playlist (`stream.m3u8`) that the browser follows while it grows.
5. `lipsynced.mp4` and `translated_audio.wav` are assembled at the end and the
   usual mux stage writes the watermarked `final.mp4`.

LatentSync changes only the face region of a rendered frame, so the output is
the original video with mouth patches on speaking frames.

## When playback can start

The same rule as the video assistant's head start. With media produced at
`rate` seconds per wall second and `remaining` seconds still to render, playing
from the start cannot stall once

    ready >= (1 - rate) * remaining + 2 s

Every `stream_segment` event carries `ready_seconds`, `total_seconds`, `rate` and
`can_play`; the `StreamPlayer` component (hls.js, native HLS on Safari) starts
when `can_play` is true, with a "play now anyway" override. The player shows
what is ready, the production rate and how much buffer it is waiting for.

## Measured (52 s speech fixture `clip_speech.mov`, English to Spanish, pool warm)

| | run 1 (`13379fd`) | run 2 (`f5369f7`, next span's TTS overlapped) |
| --- | ---: | ---: |
| transcribe | 10.0 s (whisper cold after the mode switch) | 20 s (backend just restarted) |
| translate (LLM, 2 segments) | 1.4 s | 1.4 s |
| first translated segment published | 46.4 s after upload | 54.7 s (32 s after transcribe, as in run 1) |
| first 8 s window of a span | 22.1 s (decode and warp unprepared) | 21.2 to 22.0 s |
| steady 8 s window render | 14.0 to 14.3 s | 14.0 to 14.2 s |
| pause between spans (TTS of the next span) | 7 s of idle GPUs | 0 s |
| streaming stage (TTS plus render) | 139.9 s | 131.3 s |
| playback can start without stalling | 78.7 s, 24.2 s ready at rate 0.36 | 86.9 s (same offset from transcribe) |
| frames with any change | 675 of 1302 (52%) | |
| pixels changed in a changed frame | 1.4% on average, in the mouth band | |

Transcribe time is model loading after a restart in both runs; warm it is 2 s.

The fixture is nearly continuous speech (two spans covering 49.8 of 52 s), so
passthrough saved little time here; on conversational footage with pauses the
saving is proportional to the silence. The whole-clip fast job on the same clip
took 139 s, so streaming costs about 15 s of total time for a result that is
watchable 110 s earlier.

Verification run: `~/e2e-out/run-stream.sh` on the box (switches mode, submits
the job over the API, records every SSE event with timestamps, prints the
segment timeline); recordings and reports under `~/e2e-out/stream*/`.

## Try it

Translate mode on the box, then on your machine:

    ssh -L 3030:localhost:3030 user@xe7740

Open http://localhost:3030, choose "Streaming · starts playing while it
renders", upload a clip. The live player appears under the pipeline view and
starts when the head-start rule allows; the final file replaces it when the
job completes.

## Limits and next steps

- First-frame latency is transcribe + translate + the first span's TTS + the
  first window. Capping span length (splitting long whisper segments at word
  pauses, which needs per-piece translation) would cut the first span's TTS
  from 8 s to about 3 s; preparing the first window during TTS would save
  another 8 s.
- The render rate is 0.36 s of video per second on this clip (8 GPUs, 10
  steps). The renderer-side levers from the profile (batched restore, restore
  on the worker GPUs) apply directly; so does any silence in the source.
- Gaps carry the original sound in the stream but silence in the final
  dub (`translated_audio.wav`), as the whole-clip pipeline does; a background
  stem would unify them.
- Webcam capture into the span pipeline (translation while recording) is not
  wired yet.

## Slot fitting on short clips (5 Oct 2026)

A 9 s user clip transcribed as one segment whose Spanish ran 10.7 s at natural
pace, 1.18x the slot against the preferred 1.15x cap. The LLM rewrite did not
shorten the text, the rewritten take rendered slower (17 s) and replaced the
usable one, and the job failed with "cannot fit its 9.12s slot". Now the
shortest verified take is always kept, a failed rewrite is a warning rather
than an error, and a take is stretched up to `TTS_MAX_SPEED_HARD` (1.3x,
formant-preserving) when nothing shorter exists. Beyond that the job still
fails without discarding speech. This applies to every mode, not only stream.
