# LatentSync jitter debug plan

Systematic session for isolating which pipeline stage introduces the
frame-to-frame face jitter that users have been observing in output
videos. Follows the reviewer's proposal: measure quantitatively,
bisect the pipeline, apply fix to the stage that's actually broken.

Budget: one focused ~4-hour session.

## Ground truth

Current best-guess state after PR #49/#50/#53 work:
- Weight loading: 100% clean (strict-load confirmed `1246/1246`)
- Audio conditioning: healthy (`mean≈0.59, std≈2.24`, no NaN)
- Config: `stage2_512.yaml` (matches release)
- Affine smoothing: Savitzky-Golay, decomposed similarity parameters,
  window 9 (PR #53)

User still observes jitter in output even when source is near-static.
The question is: **where in the pipeline is the jitter being
introduced or amplified?**

## Step 0 — capture environment state

Before any debugging, lock down what's actually running:

```bash
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/environment_audit.py > /tmp/env_before.json
cat /tmp/env_before.json | jq '.torch.deterministic_algorithms, .torch.version, .env.LATENTSYNC_IPEX_DTYPE, .env.LATENTSYNC_ENABLE_DEEPCACHE, .env.LATENTSYNC_AFFINE_SMOOTH_WINDOW'
```

Note any surprises. Save `env_before.json` — we'll diff against it
later if a config change accidentally lands.

## Step 1 — Baseline stability metric

Measure landmark jitter on the **source** and **current output**:

```bash
# Source — this is our noise floor. How much did the subject / camera actually move?
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/stability_metric.py /jobs/<latest>/input.mov

# Current output
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/stability_metric.py /jobs/<latest>/final.mp4
```

Interpret:

| Mean disp source | Mean disp output | Diagnosis |
|---|---|---|
| < 1 px | < 1 px | Not jittering — user perception disagrees with metric, need visual bisection |
| < 1 px | 2-4 px | Pipeline is adding ~2-4 px of jitter |
| < 1 px | > 4 px | Pipeline is amplifying significantly |
| 2-4 px | 2-4 px | Source already at this level; pipeline is passing through |
| 2-4 px | > 4 px | Source jittery *and* pipeline amplifying |

Target: **output mean ≤ source mean + 0.5 px**. That's "adds effectively no jitter."

## Step 2 — Bisect the pipeline

Run each bypass and re-measure with the stability metric. The
stage whose bypass closes the source-vs-output gap is the one
introducing jitter.

### 2a. Bypass the smoothing (rule out: SG is actively hurting)

```bash
echo "LATENTSYNC_AFFINE_SMOOTH_WINDOW=1" >> .env
docker compose up -d lipsync-latentsync
make run-latentsync FIXTURE=...
# measure output mean disp, compare to with-smoothing run
```

- Smoothing off **more** jittery → SG is helping (window too small is the issue)
- Smoothing off **same** jittery → SG is doing nothing (jitter not from affine)
- Smoothing off **less** jittery → SG is actively hurting (rare; investigate)

Revert after test:
```bash
sed -i '/LATENTSYNC_AFFINE_SMOOTH_WINDOW=1/d' .env
docker compose up -d lipsync-latentsync
```

### 2b. Bypass the UNet entirely (rule out: UNet is the jitter source)

Add env var `LATENTSYNC_BYPASS_UNET=1` → pipeline passes the
canonical face crop through un-regenerated, then runs restore /
compositing as normal. Pure geometric pipeline.

See `LATENTSYNC_BYPASS_UNET` in `latentsync_driver/inference.py`
(added in this PR). When enabled, `synced_video_frames` gets
populated with the input face crops (scaled from `faces` tensor)
rather than VAE-decoded UNet output.

If output is **still jittery** after UNet bypass → jitter is 100%
geometric (affine / warp / restore). If output is **stable** →
UNet is contributing.

### 2c. Bypass the mask (rule out: mask is the jitter source)

Set `LATENTSYNC_BYPASS_MASK=1` → composite full warped UNet output
with no soft mask. If jitter disappears, the soft-mask computation
is unstable frame-to-frame.

### 2d. Synthetic-input test (rule out: source micro-motion)

Generate a video where every frame is literally the same bytes:

```bash
# Duplicate one source frame 75 times — zero motion possible
ffmpeg -i artifacts/inputs/IMG_7228.MOV -vf "select=eq(n\,30)" -vframes 1 /tmp/still.png
ffmpeg -loop 1 -i /tmp/still.png -t 3 -r 25 -c:v libx264 -pix_fmt yuv420p /tmp/static.mp4

# Run the pipeline on it
make run-latentsync FIXTURE=/tmp/static.mp4
# Measure output. Any jitter here must be pipeline-internal.
```

## Step 3 — Deterministic reproduction

If two "identical" runs produce different jitter, we're chasing
ghosts. Enable deterministic mode:

```bash
cat >> .env <<'EOF'
# Deterministic reproduction
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=0
LATENTSYNC_DETERMINISTIC=1
LATENTSYNC_SEED=42
EOF
docker compose up -d lipsync-latentsync
```

`LATENTSYNC_DETERMINISTIC=1` triggers (in the driver):
```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(seed)
np.random.seed(seed)
```

Two runs with identical inputs + this flag should produce
bit-identical outputs. If they don't, there's non-determinism
upstream (IPEX kernels, VAE pre-cache, etc.).

## Step 4 — Apply fix to identified stage

Based on which bypass closed the gap in Step 2:

- **Smoothing window too small**: bump `LATENTSYNC_AFFINE_SMOOTH_WINDOW` to 15 or 21, re-test
- **UNet is the jitter source**: investigate VAE dtype mismatch, CFG math, seed-per-frame behavior
- **Mask is the jitter source**: look at `inv_soft_mask` computation in `affine_transform.py`
- **Even synthetic-input jittery**: something in the geometric pipeline is numerically unstable (kornia ops, precision accumulation). Add `torch.set_default_dtype(torch.float32)` and/or use `align_corners=True` explicitly in the warps
- **Source micro-motion**: enable `ENABLE_VIDEO_STABILIZATION=true` and/or (when merged) `ENABLE_OUTPUT_STABILIZATION=true` — see PRs #51 and #54

## Step 5 — Verify with metric

Don't call it done until the stability metric confirms:

```bash
docker compose exec -T lipsync-latentsync \
    python /app/repo_scripts/latentsync_debug/stability_metric.py /jobs/<new>/final.mp4
```

Target: output mean disp ≤ source mean disp + 0.5 px.

## Step 6 — Document the finding

Whatever the fix is, write it up in `docs/lipsync.md` under the
LatentSync section so the next person (or us, six months from now)
doesn't re-do this session.
