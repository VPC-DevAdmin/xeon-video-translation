# Local implementation checklist

Status: local implementation and validation complete. See [GPU testing handoff](gpu-testing-handoff.md) for evidence and limitations. Checked items cover local code and contracts; model quality and hardware behavior remain separate acceptance gates.

## 1. Reliability and reproducibility
- [x] Transactional job state, ownership, attempts, leases and crash recovery.
- [x] Stage checkpoints, retry from failed stage, cancellation and artifact cleanup.
- [x] Preserve short speech spans and test failure paths; quiet-microphone quality remains a hardware acceptance check.
- [x] Model/configuration provenance and reproducible test environments.
- [x] Browser and transport tests for recording, retries and session cleanup.

## 2. Measurement
- [x] Structured stage/session metrics, queue time and resource samples.
- [x] Evaluation fixtures, quality scorecards and comparison reports.
- [x] Readiness diagnostics and an executable GPU acceptance matrix.

## 3. Execution and memory
- [x] Bounded, overlapping video windows with checkpointed render/encode.
- [x] Resource scheduling, quotas, retention and per-window cancellation.
- [x] Explicit optimization settings and reproducible A/B experiments.

## 4. Translation and audio
- [x] Context/glossaries, protected terminology and validation.
- [x] Duration fitting/retries and editable failures without speech loss.
- [x] Word alignment integration, speaker assignments and voice selection.
- [x] Background audio separation/remix integration and level controls.

## 5. Avatar
- [x] Trained VAD integration and endpoint tuning.
- [x] Overlapped bounded generation/render/delivery with audio fallback.
- [x] Played-speech history, interruption fencing and session telemetry.
- [x] Audio-context overlap, portrait/voice inputs and renderer capability declarations.
- [x] Provisional microphone transcripts; final utterance transcripts are committed separately.

## 6. Product and operations
- [x] Job history, queue, cancel, retry and delete.
- [x] Transcript/translation editor and selective regeneration.
- [x] Source/result comparison, voice preview, subtitles and download bundles.
- [x] Recording timer, connection diagnostics and recoverable UI states.
- [x] User ownership, quotas, expiring TURN credentials and readiness.
- [x] Upgrade/rollback and GPU handoff documentation.

## Hardware acceptance (blocked only on GPU availability)
- [ ] Build and run every CUDA image; verify model and framework compatibility.
- [ ] Numerical/visual parity for precision, compilation and sharding variants.
- [ ] Real speech/portrait quality, window seams, multilingual/speaker quality.
- [ ] Latency/throughput/VRAM matrix with simultaneous interactive and batch load.
- [ ] Real cameras, speakers, mobile browsers and external TURN/NAT tests.

## Review remediation

- [x] Automatically window clips that exceed renderer frame budgets; share budget defaults across backend and renderers.
- [x] Drain native inference on graceful shutdown and preserve recoverable jobs.
- [x] Clean isolated TTS clicks only with complete expected-text alignment.
- [x] Fix cached F5/IndicF5 regeneration and cover both backends.
- [x] Restore conservative XTTS repeated-tail cleanup and bounded same-text retries.
- [x] End streams and release local registries after lost claims; preserve winning workers and explicit cancellation.
- [x] Restore process sharding and make the conservative ETA factor configurable.
- [x] Store artifact byte totals in SQLite; move reconciliation off the event loop.
- [x] Normalize legacy revision stages and reject missing prerequisites with HTTP 409.
- [x] Reconcile frontend documentation with Next 15/React 19/Tailwind 4 and build/browser evidence.
- [ ] Validate automatic window seams, speech alignment/retry quality, process sharding speed and shutdown draining on the GPU host.
