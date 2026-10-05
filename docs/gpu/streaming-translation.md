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

## Occluders in front of the face (5 Oct 2026)

On the user's clip two chocolate boxes are held up in front of the speaker and
the old output painted a mouth on the packaging. LatentSync pastes the whole
generated crop back wherever the landmark track places the face, and the
track carries the last good landmarks through frames with no detection.
Fix in the LatentSync service (`latentsync_driver/face_parse.py`):

* **Pixel mask.** BiSeNet face parsing (the MuseTalk weights already in the
  model volume) labels the *source* crop; only face pixels receive the
  generated face, dilated 9 px, feathered 15 px and averaged over 3 frames.
  Objects and hands drawn over the mouth stay on top.
* **Frame gate.** The shared face track now stores per-frame visibility
  (`TRACK_VERSION` v2, old caches rebuild); frames without a detected face,
  one frame either side, are not pasted at all and the paste ramps over 3
  frames. The pipeline logs `latentsync_occlusion` with the counts.

Verified on the clip: frames 85-93 and 184-190 (both boxes) changed
1,400-2,400 pixels in the old output and 0 in the new one; elsewhere the two
outputs differ only where the occluder overlaps the mouth. Parsing costs
0.24 s per 8 s window. `LATENTSYNC_OCCLUSION_MASK=0` disables the pixel mask
(the frame gate always applies). Residual: a bare hand fully over the mouth is
skin to the parser, so it is caught only when detection drops too.

### Partial covers (5 Oct 2026, later)

The user then found two partial covers: a finger and box edge beside the
mouth, and a blurred pale box crossing it. BiSeNet calls both "skin". Added in
`face_parse.py`, all measured on the same clip:

* **Temporal occluder mask.** In the aligned crop each frame is compared with
  the median of up to 30 visible frames before it and after it (128 px,
  brightness-matched on agreeing pixels); only what differs from both counts,
  so a pose change (persists into the future) and a moving mouth (compact blob,
  never reaches the crop border) are not flagged. Blobs of at least 2 percent
  of the crop touching the border are kept. Catches hands and dark objects.
* **Covered-mouth gate.** The pale translucent box is only 40 percent caught at
  pixel level, and a half-painted mouth looks worse than either extreme, so a
  frame whose lower-face band is more than 30 percent occluded is treated like
  a lost face (clean frames peak near 10 percent). Gate margin is now 2 frames.
* **Confidence fade.** The track stores the detector score (v3); paste fades
  between 0.55 and 0.72, where half-covered faces sit on this footage.

Result: frames 77-96 and 182-199 (both boxes and the hand pass) now equal the
source; 51 of 212 frames carry a reduced paste. Cost: while a hand covers half
the mouth the original mouth shows for those frames instead of a translated
one under the fingers. Knobs: `LATENTSYNC_OCCLUDER_MASK=0`, `MOUTH_COVERED`.

### Fingers on the lips and the English mouth at the end (5 Oct 2026, night)

* **Hand mask.** A finger resting on the lips for half a second is in the
  temporal median and is skin to the parser. MediaPipe Hands (already in the
  LatentSync image) runs on the source frames at 540 px wide (3 s per 200
  frames on CPU); each hand is drawn into the crop through the paste affine as
  the palm polygon plus finger bones at about a third of the palm width,
  capped at 36 px because a hand near the camera is larger than the face. The
  first version used a hull and uncapped width and swallowed half the face.
  `LATENTSYNC_HAND_MASK=0` disables it. The occlusion log line now lists the
  gated frame ranges.
* **Trailing silence.** XTTS leaves up to a second of silence after the last
  word. Whisper-verified takes skipped the silence trim, so the fit stretched
  the silence along with the speech: the Spanish ended at 8.5 s of a 9.1 s clip
  and LatentSync, which reproduces the source mouth under silence, showed the
  English "it" for the last 0.6 s. Verified takes are now trimmed too
  (headroom 0.1 s); the stretch dropped from 1.20x to 1.15x and speech runs to
  8.95 s. The last three frames of silence still show the source mouth.
