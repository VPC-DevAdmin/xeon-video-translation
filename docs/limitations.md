# Known limitations

Honest expectations for a Xeon-CPU build.

## Latency

End-to-end on a 16-core Xeon for a 15 s clip, defaults (`whisper=base`, NLLB-600M):

| Stage         | Typical | Notes                                |
| ------------- | ------- | ------------------------------------ |
| Stage 1       | < 1 s   | ffmpeg, IO-bound                     |
| Stage 2       | 2–5 s   | faster-whisper int8                  |
| Stage 3       | 5–20 s  | scales with segment count            |
| Stages 4–6    | TBD     | not implemented yet                  |

This is acceptable for a demo but it is *much* slower than the GPU build the
original spec assumed. If you need sub-30 s end-to-end on CPU:

- Use `whisper=tiny` and accept worse transcripts
- Trim clips to ≤ 10 s before upload
- Run translation on shorter segments (the current code splits by Whisper's
  VAD-driven segmentation)

## Quality

- **Single-speaker only.** Whisper will transcribe multiple speakers but they
  collapse into one text track — translations and (eventually) dubbing get
  ugly.
- **Best with front-facing, well-lit clips.** Lip sync (M4) won't recover
  from heavy occlusion or extreme angles.
- **English source is best supported.** Whisper's accuracy degrades on
  smaller languages.
- **NLLB-600M is the *distilled* model** — it's chosen for CPU footprint,
  not maximum quality. The 1.3B and 3.3B variants are noticeably better and
  drop in as `NLLB_MODEL=facebook/nllb-200-1.3B` if you have the RAM.

## Production caveats

- **Single-process job registry.** `MAX_CONCURRENT_JOBS=1` is a real limit;
  jobs queue but don't run in parallel. Multiple workers would need Redis
  (or similar) for shared state.
- **No auth.** Bind to `localhost` only for demos; do not put this on the
  open internet without adding auth and rate limiting.
- **Disk usage grows unbounded.** Job artifacts are not garbage-collected
  yet. Mount `/jobs` somewhere with capacity, or add a cron job.

## Ethics & disclosure

See [ethics.md](ethics.md). The watermark is on by default and should remain
on for any output that leaves the demo machine.
