# Review remediation — 3 October 2026

All ten findings have local fixes and regression coverage. Changes remain uncommitted. GPU inference, speech/face quality, and latency still require target-host testing.

| Finding | Resolution | Local evidence |
| --- | --- | --- |
| Frame budgets reject supported clips | The dispatcher probes dimensions, frame rate and duration. Oversized MuseTalk/LatentSync jobs automatically use bounded 25 fps windows, including overlap and 20% frame-budget headroom. Backend and worker budget defaults agree. Worker guards remain in place for direct calls. | Both 52-second 1080×1920 budget cases, tight 4K budgets, short-clip admission, and real ffmpeg window rendering/resume tests. |
| Shutdown cancels durable jobs | Graceful shutdown drains native calls and requeues interrupted work while preserving explicit user cancellation. Queued jobs resume on startup. Compose allows ten minutes to drain. | Actual shutdown/startup hooks with one native worker and one queued job; shutdown waits for the native call. |
| Clicks cause false TTS overruns | Complete, confident expected-text alignment permits isolated click removal. Recognized short and quiet words remain protected. The same text is resynthesized before optional shortening. | Real synthetic WAV click, short-word and quiet-word tests; bounded retry tests. |
| Cached F5 retry uses an undefined callable | Generator selection happens before cache lookup. Retries retain the selected speaker reference/voice. Raw-cache format is versioned. | First-segment cache-hit → overrun → rewrite → regenerate tests for F5 and IndicF5. |
| XTTS hallucinated tails survive | Restored conservative word-timestamp cleanup requiring complete text coverage. Repeated intended words are preserved; extra recognized tail words can be removed. Partial/uncertain alignment cannot authorize cutting. Removed obsolete duration truncation fallback. | Repeated-tail, incomplete text, low confidence, overlapping timestamps, bounded failure and resynthesis tests. |
| Failed claim leaks registry/SSE | Lost claims detach local state and end the stream without changing another worker's lease/status. Duplicate dispatch leaves the winning worker intact. Predispatch cancellation is terminal immediately. | Early cancel, duplicate worker, foreign claim, stream completion and durable status tests. |
| Thread sharding replaces process default | Restored process default in GPU Compose and environment example. ETA factor is configurable, retaining 5.0 as a conservative historical estimate. | Compose validation; existing process-worker lifecycle tests. Speedup needs GPU measurement. |
| Submission walks all job files | Added migrated SQLite byte totals. Admission/metrics queries use SQL off the event loop. Uploads, revisions, stages and completion refresh one job; startup reconciles history and periodic maintenance reconciles active jobs. | Legacy DB migration, owner isolation, deletion, quota enforcement without directory walks, active-only refresh and responsive health during a slow capacity check. |
| Revision indexes incomplete stages | Normalize stages by name. Validate upstream checkpoints before creating an edited revision; unavailable prerequisites return HTTP 409. | Empty/partial metadata returns 409; reordered legacy stages retain edits in a canonical stage list. |
| Frontend documentation is stale | Corrected Next 15.5.27 / React 19.3.0 / Tailwind 4.3.3 / PostCSS 8.5.28 documentation and current implementation status. | Production build, six Chromium workflows through actual proxy routes, Linux image build and Compose validation. |

## Validation

- **82 backend/worker tests passed**, including **27 new review regressions**. Model calls use fixtures; media tests use actual ffmpeg/ffprobe.
- **15 ingest/avatar tests passed**, including real loopback WebRTC media transport and real CPU Silero ONNX inference.
- **6 Chromium workflow tests passed** against the built frontend with deterministic API fixtures.
- Production frontend build passed. Linux frontend image build passed using cached unchanged frontend layers. GPU/avatar/optional-audio Compose configuration validates.
- Static undefined-name/syntax checks passed. FastAPI/Starlette deprecation warnings remain.

## Remaining GPU acceptance gates

1. Run the original 52-second portrait clip plus 120-second clips through MuseTalk and LatentSync. Check every automatic window boundary, A/V timing, peak host RAM and VRAM. The frame budget does not bound all model memory.
2. Evaluate real Spanish short utterances, clicks, repeated tails, quiet words, non-Latin languages and uncertain ASR. Recognizer fixtures verify control flow, not real recognition accuracy. Measure false rejection, artifacts and added validation latency. Ambiguous alignment retains audio; unresolved overruns fail rather than cut required speech.
3. Compare process/thread sharding and calibrate `LATENTSYNC_REALTIME_FACTOR` with the actual GPU count, render windows and quality settings. Previous throughput figures do not prove current performance.
4. Restart during real local and remote inference; verify checkpoint reuse. Graceful drain/requeue is covered locally. SIGKILL, host loss or expiry of the ten-minute grace period leaves interrupted jobs failed for an explicit checkpoint retry.
5. Repeat concurrency and external-camera/TURN tests from `gpu-testing-handoff.md`.

Storage quotas remain admission checks; accepted jobs can grow between accounting updates. Direct renderer API calls still enforce frame limits; automatic windowing is provided by the translation dispatcher. No GPU services were deployed or GPU results claimed in this pass.
