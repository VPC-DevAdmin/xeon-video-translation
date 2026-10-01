# Mode 3 — real-time voice avatar

**Goal:** a voice assistant with a face. The user talks (mic over WebRTC);
the assistant answers in speech, and a talking-head video generated from
a single still image plays back in sync. Conversational latency.

This is design only. Nothing on the branch implements it beyond the ingest
service being the natural place for the live consumer.

## Shape

```
mic ──WebRTC──▶ ingest ──▶ streaming ASR ──▶ LLM turn ──▶ streaming TTS ──▶ talking head ──WebRTC──▶ browser
                                   (VAD-gated)   (text, streamed)  (audio chunks)   (video frames)
```

Latency target: first audio within ~1 s of the user finishing, video
locked to the audio. The budget is dominated by the LLM's first token and
the TTS first chunk; the talking-head model has to keep up at 25–30 fps
from a running audio stream.

## Candidate components

| Block | Candidates | Notes |
|---|---|---|
| VAD + streaming ASR | Silero VAD + faster-whisper on utterance end; or a streaming model (NVIDIA Parakeet/Canary, whisper-streaming) | utterance-end detection is the first ~300–500 ms of latency |
| LLM | Claude API (fastest to good); or local 70B-class on one 96 GB GPU | streamed tokens feed TTS sentence by sentence |
| TTS | streaming-capable: XTTS streaming inference, CosyVoice 2, Chatterbox, F5-TTS chunked | voice is fixed per avatar, so reference prep is one-time |
| Talking head from one image | Ditto (real-time, audio-driven), LivePortrait-driven variants, Hallo2 (offline quality), SadTalker (older) | must consume audio incrementally; most "from one image" models are offline, Ditto is the real-time exception worth testing first |
| Output | aiortc video track back to the browser | reuse the ingest peer connection |

## What has to be built

1. **Live consumer on the ingest track.** Today `MediaRecorder` is the only
   consumer; mode 3 reads audio frames as they arrive and feeds VAD.
2. **Turn manager.** Barge-in, end-of-utterance, and cancelling TTS/video
   when the user interrupts.
3. **Avatar preprocessing.** One-time: face crop, keypoints/latents from
   the still image, voice reference for TTS.
4. **Streaming TTS → frame generator → WebRTC video track**, with an
   audio/video clock so lips don't drift.
5. **Session state.** Per-conversation history for the LLM.

## Sequencing

Mode 3 shares G1 (backend on GPU) and G6 (TTS speed/length control) with
the other modes. It does not depend on G2–G5. A standalone prototype
(mic → ASR → LLM → TTS → Ditto → browser) on one GPU is the right first
step, before wiring it into the job model, which it mostly doesn't need.
