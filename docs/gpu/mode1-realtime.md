# Mode 1 — real-time translation

**Goal:** a webcam clip is translated, voice-cloned and lip-synced within a
few minutes of pressing Stop. Quality can be traded for speed.

## Pipeline

Same six stages as today, with GPU models and the fast lipsync backend.
Ingest is `services/ingest-webrtc`; mode `fast` maps to:

| Stage | Choice | Why |
|---|---|---|
| ASR | faster-whisper large-v3, float16 | ~30x+ real-time on this class of GPU; accuracy jump over `base` is large |
| Translate | NLLB 3.3B fp16 (G8: LLM) | sub-second per segment |
| TTS | `auto` routing (XTTS / F5 / IndicF5) | unchanged; GPU makes it several times faster than real-time |
| Lipsync | MuseTalk, no CodeFormer | MuseTalk 1.5 runs ≥ 30 fps on a single modern GPU; CodeFormer is the single biggest cost in the quality ladder |
| Mux | ffmpeg | unchanged |

## Latency budget (60 s clip, after G1 + G2)

| Stage | Expected |
|---|---|
| audio extract | < 1 s |
| transcribe | ~2 s |
| translate | ~5 s |
| TTS | ~10–20 s |
| MuseTalk | ~30–60 s |
| mux | ~2 s |

Call it one to two minutes end to end, dominated by TTS and lipsync. These
are projections from published GPU numbers, not measurements; G1 and G2
replace them with real ones.

## Speed knobs, in order of cost/benefit

1. Skip CodeFormer (already off in `fast`).
2. MuseTalk at `jaw` blend without the BiSeNet parse (faster, more visible seam).
3. whisper `large-v3-turbo` instead of `large-v3`.
4. Translate with NLLB 1.3B instead of 3.3B.
5. Run the clip in two halves on two MuseTalk workers (needs G4).

## Quality floor

MuseTalk's 256 px VAE is the ceiling: mouths are soft. Acceptable for this
mode by definition; if it isn't, the answer is mode 2, not more knobs here.

## Toward true streaming

If "within minutes" later needs to become "a few seconds behind live", the
work is in G6 and G7: utterance-level VAD, streaming TTS, and MuseTalk's
per-chunk real-time path. Nothing in mode 1's record-then-submit design
blocks that; the ingest service just gains a second consumer.
