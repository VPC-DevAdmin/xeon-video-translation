# ByteDance LatentSync — vendored code audit

Systematic comparison of our vendored tree against upstream
`bytedance/LatentSync` at commit `a229c39` (the same commit we
vendored from).

Date of audit: see git log on this file.

## Headline

**Zero unintentional divergence.** Every non-identical file is either
a documented CPU patch, observational debug instrumentation, or our
own `_smooth_affine_sequence` infrastructure. Nothing upstream is
being accidentally mutated.

## File-level summary

| Category | Count | Files |
|---|---|---|
| Identical to upstream | 36 | all models (attention, motion_module, resnet, stable_syncnet, unet_blocks, wav2lip_syncnet, utils), audio2feature, `whisper/whisper/*` except `__init__.py`, all utils except the 3 patched ones |
| Modified (intentional) | 6 | `models/unet.py`, `pipelines/lipsync_pipeline.py`, `utils/affine_transform.py`, `utils/face_detector.py`, `utils/image_processor.py`, `whisper/whisper/__init__.py` |
| Extra (our additions) | 6 | `__init__.py` stubs (×5), `utils/_debug.py` |
| Missing from ours | 11 | `data/*.py` and `trepa/*` — training-only code, correctly excluded for inference |

## Detailed diff audit

### `models/unet.py` — trivial CUDA guard (+3 lines)

```diff
             del ckpt
-            torch.cuda.empty_cache()
+            if torch.cuda.is_available():
+                torch.cuda.empty_cache()
```

**Risk**: none. `empty_cache()` on CPU is a no-op anyway; the guard
just silences a log warning.

### `pipelines/lipsync_pipeline.py` — our affine smoothing + callback + cache hooks

Three kinds of additions:

1. **`_decompose_similarity` / `_compose_similarity` / `_smooth_affine_sequence`** (new) — our similarity-parameter smoothing that upstream does not have. This is where we added custom logic.
2. **Progress callback plumbing** — `_emit_progress("phase", pct)` calls that write to `/jobs/<id>/latentsync_progress.json` for the backend's progress ticker.
3. **Denoise cache hooks** — checkpoint the post-denoise state so retries after a restore-time failure skip the expensive half.
4. **Single-line CPU patch**: `ImageProcessor(device=str(device))` was `ImageProcessor(device="cuda")`.

**Risk**: the affine smoothing is the only inference-path addition. Upstream has no temporal smoothing of the output affines (just a weak per-frame IIR bias in `transformation_from_points`). If our smoothing is doing nothing for the observed jitter, it means either (a) the jitter isn't from affines, or (b) our smoothing window isn't right for the jitter frequency. The bypass test in `DEBUG_PLAN.md` tells us which.

### `utils/affine_transform.py` — dtype default + debug dumps

```diff
-    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
+    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float32):
```

**Risk**: the dtype change is necessary — kornia/PIL ops on CPU don't reliably support `float16`. Debug dumps are observational (try/except, can't affect inference).

### `utils/face_detector.py` — CPU provider routing

```diff
-        providers=["CUDAExecutionProvider"],
+        providers=["CUDAExecutionProvider"] if is_cuda else ["CPUExecutionProvider"]
-        self.app.prepare(ctx_id=cuda_to_int(device), det_size=...)
+        ctx_id = cuda_to_int(device) if is_cuda else -1
+        self.app.prepare(ctx_id=ctx_id, det_size=...)
```

**Risk**: required. Without this, InsightFace can't initialize on CPU at all.

### `utils/image_processor.py` — removed CPU guard

```diff
-        if device == "cpu":
-            self.face_detector = None
-        else:
-            self.face_detector = FaceDetector(device=device)
+        self.face_detector = FaceDetector(device=device)

     def affine_transform(self, image):
-        if self.face_detector is None:
-            raise NotImplementedError("Using the CPU for face detection is not supported")
+        # (Removed — FaceDetector now works on CPU, see face_detector.py patch)
```

**Risk**: required pair with `face_detector.py` patch. Without this, every CPU inference path would hit the `NotImplementedError`.

### `whisper/whisper/__init__.py` — trivial CUDA guard

```diff
     del checkpoint
-    torch.cuda.empty_cache()
+    if torch.cuda.is_available():
+        torch.cuda.empty_cache()
```

**Risk**: none. Same as the UNet case.

## What upstream does that we **don't**

- `data/` — training dataset code; we don't train
- `trepa/` — TREPA (temporal consistency loss); training-only
- `DeepCache` wrapping in `scripts/inference.py` — we do this in our driver with our own env flag

## What we do that upstream **doesn't**

- Temporal smoothing of affine matrices + boxes (`_smooth_affine_sequence`)
- Content-addressable post-denoise cache (for cheap retries after failure)
- Progress-file writing (for the backend's real progress bar)
- Observational debug dumps in the restore step

## Conclusions

1. **The vendored code IS what upstream has**, aside from the
   documented patches. No accidental mutations.

2. **Our additions are additive** — a smoothing pass, a cache, a
   progress file, debug dumps. None of them replace or subvert
   upstream logic.

3. **Upstream has no temporal smoothing** of output affines. Their
   demo clips happen to be stable enough that they don't need it;
   when they release a clip with subject motion, they presumably
   accept whatever natural motion the pipeline produces.

4. **Therefore, if our output has jitter that upstream's same
   inputs wouldn't produce, the cause must be one of:**
   - Our input source (more noisy landmarks than upstream's demo clips)
   - Our CPU numerical path (precision drift relative to CUDA)
   - Our `_smooth_affine_sequence` hurting rather than helping
   - A subtle bug in one of the 6 CPU patches above

The debug plan in `DEBUG_PLAN.md` bisects exactly this hypothesis space.
