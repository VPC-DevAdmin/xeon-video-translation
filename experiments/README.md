# GPU experiment lab

Latest quality trials and measured allocation: [quality update](../docs/gpu/quality-v2-2026-10-03.md), [InfiniteTalk trial](infinitetalk/README.md).

> Hardware update: the project has now been deployed and tested on XE7740. See the [measured results and remaining acceptance gates](../docs/gpu/xe7740-validation-2026-10-03.md). Earlier local-only status below describes the pre-deployment snapshot.

The catalog is a queue of falsifiable trials, not a list of installed integrations. See `docs/gpu/research-and-experiments.md` for the evidence, ranking and evaluation gates. Plan mode works without GPU libraries and changes no serving configuration.

```bash
python scripts/experiment_lab.py
python scripts/experiment_lab.py --ids gpu-resident-frames flashhead-lite voxcpm2
```

## First hardware trial

Use an isolated Linux AMD64 environment on an explicitly allocated SM120 card. Install and record a compatible CUDA PyTorch build and PyNvVideoCodec release there; do not alter the serving environment. The probe uses NVIDIA's documented `SimpleDecoder`, GPU RGBP output and DLPack import. Start with a short SDR H.264 fixture.

Create a local recipe file with real paths and a pinned environment. Example shape (replace all `/absolute/...` paths):

```json
{
  "gpu-resident-frames": {
    "argv": [
      "/absolute/experiment-venv/bin/python",
      "/absolute/repo/scripts/probe_gpu_frames.py",
      "/absolute/fixtures/webcam.mp4",
      "--frames", "120",
      "--batch-size", "4"
    ],
    "cwd": "/absolute/repo",
    "model_revision": "not-applicable-codec-probe"
  }
}
```

```bash
python scripts/experiment_lab.py --execute \
  --ids gpu-resident-frames --recipes /absolute/recipes.json \
  --gpus GPU-REPLACE-WITH-ALLOCATED-UUID --timeout 120
```

The smoke result proves only whether the initial decode/import/CUDA-resize chain runs. Its FPS includes initialization and conservative synchronization; do not compare it with full pipeline throughput or claim that it qualifies colors, timestamps, encode, or visual quality. CUDA absence is a blocked result. No CPU inference fallback exists in this probe.

## Recipes and isolation

- A recipe maps an experiment ID to an explicit `argv` array, existing `cwd` and `model_revision`. Use an absolute executable from a pinned candidate environment. No shell interpolation is performed.
- Each run gets a unique folder containing `run.json`, `stdout.log` and `stderr.log`. `EXPERIMENT_OUTPUT_DIR` points to that folder; adapters should put outputs, manifests and metrics there. Stdout/stderr may contain fixture content or prompts; keep local artifacts private.
- `CUDA_VISIBLE_DEVICES` contains normalized UUIDs for explicitly selected cards. This configures direct CUDA processes; it does not configure Docker daemon GPU assignments or remote inference endpoints. Container commands must explicitly select the same UUIDs.
- Inventory includes observed driver/device/memory information. This is a snapshot, not a reservation or scheduler. Use the project's allocation/lease policy and verify occupancy before launch. The runner does not evict other workloads.
- The harness kills the local process group on timeout or interruption, including child processes. It cannot cancel a remote job or guarantee Docker cleanup. Such recipes must implement their own bounded cancellation and cleanup before they are suitable for an unattended sweep. Failed external cleanup invalidates subsequent performance comparisons.
- Nonzero exits, malformed recipes and timeouts are retained and the next independent recipe runs. Ctrl-C records the interrupted run and stops the sweep. Missing adapters, missing GPUs and insufficient card counts are blocked. The overall exit code is nonzero for blocked/failed/timed-out runs.
- Status `completed` means only that the command exited zero. `quality_status` remains `unreviewed`. Inspect the adapter's detailed result: a failed command can contain a more specific blocked-dependency result.

## Adapter contract before a model trial

Implement one adapter per isolated candidate environment. Preserve these common outputs:

1. Exact input hashes; source and weight revisions; installed dependency/container manifest; GPU UUIDs; dtype; seed; all generation settings.
2. For TTS: source text, target language, reference voice hash, PCM audio, sample rate, first-chunk time, chunk boundaries and total synthesis time. Test cancellation, final-tail delivery and explicit resampling.
3. For ASR/translation: recognized/translated segments and language, word timing where supported, model template/prompt, names/numbers/negation checks and native-speaker review.
4. For avatars: portrait hash, input audio, generated frames/video with timestamps, first moving frame, sustained FPS, dropped/late frames, A/V skew and identity/temporal review.
5. For video optimization: exact fixture, pixel/color/timestamp metadata, transfer and memory measurements, final video and blind paired review.

Start with upstream offline inference before writing a production service adapter. Then satisfy the current service's streaming, cancellation and artifact contracts. Do not report an upstream demo as integrated application functionality.

## Rebaseline and compare

The existing benchmark runs against an already-configured backend. These flags label the run; they do not change the model, reset caches, reserve GPUs or verify the stated allocation. Warm the intended models first for `warm-models`; use new artifacts as appropriate. Cold runs require a separately controlled reset. Save service/model manifests alongside the results.

```bash
python scripts/benchmark_modes.py /absolute/fixtures/webcam.mp4 \
  --modes fast --target es --repeats 5 --tag current-fast \
  --phase warm-models --hardware-profile shared-renderer-2 \
  --config renderer=musetalk-1.5 --output artifacts/bench/current-fast
python scripts/compare_benchmarks.py artifacts/bench/current-fast/results.jsonl
```

Comparator groups preserve fixture, language, cache phase, mode, hardware label, observed GPU details, source revisions, job options and explicit configuration. Candidates with different source/configuration can share a workload/hardware cohort hash while remaining separate rows. Legacy records are readable but marked as incomplete provenance. Untracked source hashing excludes model/artifact directories; use `scripts/model_manifest.py` separately for model weights.

P95 is withheld below 20 completed runs. Use a larger representative set for qualification. Unsuccessful runs remain in the denominator and are reported separately; success-only latency must not hide a high failure rate. Quality review fields must be filled and inspected—there is no automatic quality pass.

## Local checks

```bash
python -m pytest tests/research -q
```

These tests run real child processes for success, failure and timeout handling, verify GPU selection and benchmark separation, and check the probe's fail-closed behavior. They do not install or execute GPU models.
