# Three products, one box: plan for the redefined use cases

Date: 4 October 2026. State of the code: `gpu/track` at `3c172ba`.

## What changed in the brief

The three modes were named for their speed. They are actually three different products that happen to share models:

| Working name | What it really is | Judged by |
| --- | --- | --- |
| Video assistant (was "avatar") | The voice assistant we built, with a face that talks back | Time from the user finishing a sentence to a moving, speaking reply; interruption; naturalness over a long session |
| Streaming translation (was "fast") | Translate an uploaded or captured video and start showing it while the rest renders, touching as few pixels as possible | Time to first translated frame; whether playback ever stalls; how much of the original is untouched |
| Script-to-video generation (was "quality") | Learn a person's face and voice from footage, then produce a new video of them delivering any script | Identity, voice likeness, naturalness of motion, zero missing or wrong words; runtime is secondary |

Only one product is live on the box at a time, chosen from a UI. That removes the co-location question entirely: each product gets a layout designed for it, and a switch is a container start plus a 40 s warm-up.

## What carries over from this week

- Backend job system (SQLite state, recovery, checkpoints, studio edits), the LLM client against vLLM, batched whisper, XTTS/F5 voices with per-segment fitting and whisper-verified cleanup.
- LatentSync as a sharded renderer: one coordinator, one worker process per GPU, worker-side conditioning, shared face track, startup warm-up. 8 GPUs give a 41 s fast window and a 74 s quality window per 16 s of video.
- MuseTalk avatar renderer, the WebRTC ingest and playback with NVENC/NVDEC, Silero VAD, and a measured 1.1 s end-of-speech to audible reply.
- The profiling tooling (`scripts/profiling/`) and the measured truth that the per-window fixed stages, decode, warp, restore and write, are now the floor for the translation renderer.
- `scripts/mode.sh` as the first version of the mode switch.

## Product 1: video assistant

**Goal.** The earlier voice-assistant behaviour (standard voices, tools, conversational memory) plus a person on screen who speaks the reply, interrupts cleanly, and idles believably.

**Design.**

- Dialogue layer: port the voice assistant's turn logic, tools and memory onto `backend/app/llm.py` so it talks to vLLM on GPU 1 like everything else. The current `/avatar` turn endpoint is the attachment point.
- Speech: streaming XTTS with a bundled speaker (standard voice) stays; add a fixed voice catalogue in the UI.
- Face: MuseTalk portrait rendering is the shipped path. FlashHead Lite measured 114 fps warm with a 0.39 s first chunk and is the upgrade candidate for head motion and expression; it needs the bounded streaming and cancellation contract the MuseTalk path already honours. Blink, gaze and breathing during pauses are the visible gap in every sample so far.
- Idle behaviour: a looping idle clip or generated idle motion, cross-faded at utterance boundaries, instead of a frozen frame.
- Layout when live: GPU 0 speech (whisper + XTTS), GPU 1 LLM, GPU 2 renderer plus WebRTC codecs, GPUs 3 to 7 free. Those five cards can run a second and third assistant session, or FlashHead Pro for a higher-fidelity face, without touching the first.

**Targets.** First audible reply p95 under 1.5 s and first moving frame under 1.8 s, interruption under 250 ms, a 30-minute session without drift. Current single-session numbers meet the first three; the long session and multi-session cases are unmeasured.

**Work items.**

1. Port the assistant dialogue layer (tools, memory, persona prompt) behind the LLM client. Medium.
2. Idle motion and blink layer for the portrait renderer. Medium.
3. FlashHead Lite adapter behind the existing streaming contract, A/B against MuseTalk. Large.
4. Multi-session scheduling across GPUs 3 to 7. Small once the renderer is per-process.
5. Measure the latency targets through a real browser and TURN, 20 sessions. Small.

## Product 2: streaming translation

**Goal.** A translated video that starts playing within tens of seconds and never stalls, with the fewest changed frames. Quality must stay at the level we accept for the generation product, just produced incrementally.

**Design: change the unit of work from clip to segment.**

- Today the pipeline runs transcribe, translate, TTS and lipsync as whole-clip stages, then windows the lipsync. Instead, run the chain per speech segment as soon as whisper emits it: translate the sentence, synthesize it, hand the frames that cover it to the renderer, and append the result to a growing output. The first segment is displayable after roughly transcribe plus one TTS plus one render window.
- Render only speech spans. Frames with no speech, and frames outside the mouth region, pass through untouched. On the fixture that is about 8% of frames; on conversational footage with pauses it is far more. This is the "not too many new frames" requirement made concrete: the output is the original video with mouth-region patches on speaking frames.
- Pipeline the renderer. Decode and warp window N+1 and restore window N while N+1 denoises. That overlaps 27 of the 41 s fast window with GPU work and is the single largest remaining lever; it applies to the generation product too.
- Deliver as a stream. Write fragmented MP4 or HLS segments per window and let the player follow the live edge; the studio keeps the final file.
- Renderer choice stays LatentSync at 10 steps. MuseTalk is 3 to 4 times cheaper per frame but the mouth-patch smoothing was judged unacceptable in review. Keep it as a configurable fallback for sources that cannot sustain the stream.

**Targets.** First translated frame within 30 s of upload for a 60 s clip. Sustained render rate at or above real time on 8 GPUs, which the per-window numbers say is reachable once the fixed stages are pipelined: 13.5 s of denoise per 16 s window today, plus whatever decode and restore still cost after overlap. End-to-end for a 60 s clip near 90 s.

**Work items.**

1. Segment-level orchestration: emit, translate, synthesize and render per sentence; keep the per-segment timing contract already enforced by TTS. Large.
2. Speech-span and mouth-region passthrough in the window planner. Medium.
3. Renderer window pipelining in `lipsync_pipeline.py`: a decode thread ahead, a restore thread behind. Medium.
4. Batched restore kernels across frames with a shared erosion radius, or restore on the worker. Medium.
5. Fragmented MP4/HLS writer and a live-edge player in the frontend. Medium.
6. Webcam capture path: the WebRTC recorder already produces the input; wire it to the segment pipeline so translation begins while the person is still recording. Medium.

## Product 3: script-to-video generation

**Goal.** From an existing video of a person and a script, the best video we can make of that person saying the script. Translate-the-script is a button in front of this, not a different pipeline.

**Design: a persona, then a render.**

- **Persona capture** (once per person, reused across jobs). From the reference video: the shared face track and canonical crops; a voice profile built from the best 20 to 30 s of clean speech, selected by the existing reference-span logic rather than the first 1.8 s; and, when the footage allows, a fine-tuned voice. XTTS and F5 both fine-tune on 5 to 10 minutes of a speaker and the likeness gain over zero-shot is large. Store the persona as a job-independent artifact with consent recorded.
- **Script to speech.** Conversational wording pass through the LLM when the user asks for it, then synthesis per sentence with the whisper-verified cleanup, gap capping at 0.65 s, native-rate master and 16 kHz conditioning copy. This is the preparation that fixed the 14 s preview; it becomes the standard path.
- **Motion source, two tiers.**
  - Tier A, reference-driven: the person's own footage drives head and body motion; LatentSync at 40 steps regenerates the mouth with whole-clip smoothing. This is what we have, it is the most realistic because every non-mouth pixel is real, and its limit is footage length: a 60 s script against a 30 s reference means looping, which LatentSync does with reversal and which reads as a loop. Mitigation: cut the reference into natural segments at pauses and shuffle rather than ping-pong.
  - Tier B, generated motion: audio-driven video generation from a still or short clip for scripts longer than the footage, or when gestures should follow the words. Candidates already on the box: FlashHead Pro (18 to 22 fps, 512 px, generated this week's minute-long sample) and InfiniteTalk on Wan 2.1 14B (weights downloaded, four-GPU NCCL start-up still fails; single-GPU attention path exists). Wan 2.2 S2V is the newer open option to evaluate. An identity LoRA trained on the reference footage is the step that turns "a face like this person" into "this person" for Tier B.
- **Finishing.** Face restoration at the generated resolution, upscaling to the source size, the loudness and tail checks, and an automatic QA pass: ASR word match against the script, a lip-sync offset estimate, and a flicker metric, written into the job for review.

**Targets.** Blind preference over the Tier A output of this week; zero missing words; voice likeness rated by a native speaker; runtime of minutes per minute is acceptable, hours is not.

**Work items.**

1. Persona artifact: face track, voice reference selection, voice fine-tune job, consent record, reuse across jobs. Large.
2. Script preparation as the standard path: optional LLM conversational pass, translation button, gap capping, native master. Small, mostly moving the trial scripts into the pipeline.
3. Reference segmentation and shuffle for Tier A when the script outruns the footage. Medium.
4. Tier B integration: FlashHead Pro behind the generate job with the persona voice; InfiniteTalk single-GPU, then multi-GPU once NCCL is fixed; Wan 2.2 S2V trial. Large, staged.
5. Identity LoRA training job for Tier B. Large, after Tier B renders end to end.
6. Automatic QA pass and review scorecard in the studio. Medium.

## Cross-cutting design decisions

- **Exclusive modes and the UI.** `scripts/mode.sh` is the mechanism. For the UI to switch modes, add a small control service that owns the Docker socket and exposes `POST /mode` behind the internal key, rather than giving the backend socket access. The frontend shows the live mode, the switch progress and the warm-up state each mode already reports.
- **Per-mode GPU layouts.** Assistant: 0 speech, 1 LLM, 2 render, 3 to 7 for extra sessions or a higher-fidelity face. Streaming translation: pool on all eight, LLM on 1 and speech on 0 co-located, which the measurements show is harmless. Generation: Tier A uses the same pool; Tier B takes the GPUs its model needs, four for InfiniteTalk, one for FlashHead, with voice fine-tuning on whatever is left.
- **One renderer core.** Window pipelining, batched restore, worker-side conditioning and the face track serve both translation and Tier A generation. Fix them once in `lipsync_pipeline.py`.
- **Persona as a first-class object.** Translation jobs gain from it too: a known speaker's voice profile beats the per-job reference selection.
- **Quality gates stay automatic.** Word-exact ASR verification, gap limits, loudness, duration checks and the review scorecard apply in every product. The reviewer must not be able to import context, as the `1.5x` guard enforces.

## Other use cases worth planning for

| Use case | What it needs beyond the three products | Fit |
| --- | --- | --- |
| One script, many languages | Persona reuse plus a per-language render queue; translation button in bulk | Falls out of Product 3 plus the LLM |
| Multi-speaker dubbing | Diarization and per-speaker voice profiles; the optional audio-quality service already covers alignment, diarization and separation | Product 2 extension, medium |
| Live meeting or call translation | True real time: streaming ASR, incremental translation, streaming TTS, a live-capable face renderer, sub-second budgets | Different architecture; the assistant's streaming stack is the seed |
| Slides or document to presenter video | Script generation from source material, then Product 3; layout compositing of slides and presenter | Product 3 plus an LLM authoring step |
| Audio-only dubbing and subtitles | Already present as the dub mode; add caption export and burned-in subtitles | Small |
| Review and correction loop | Studio editing of transcript and translation exists; extend to script edits and re-render of only the affected segments | Builds on segment-level orchestration |
| Consent, provenance and disclosure | Persona consent records, C2PA provenance, the watermark that is already on | Required before any external demo of Product 3 |

## Order of work

1. Renderer window pipelining and batched restore. Both translation and generation speed up, and the result decides whether streaming translation can hold real time on 8 GPUs.
2. Segment-level orchestration with passthrough and a fragmented output, which turns the fast path into streaming translation.
3. Persona capture and the standard script-preparation path, which lifts generation quality immediately with the models we have.
4. Assistant dialogue port, idle motion, then FlashHead Lite.
5. Tier B generation: FlashHead Pro behind the generate job, InfiniteTalk single GPU, then the multi-GPU and LoRA work.
6. The mode control service and UI switch, once the three stacks are stable enough to be worth switching between from a browser.
