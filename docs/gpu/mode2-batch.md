# Mode 2 — batch translation

**Goal:** the highest quality the open-source stack can produce. Turnaround
is not a constraint, but with 6 GPUs it should still be minutes, not hours.

## Pipeline

Mode `quality` maps to:

| Stage | Choice | Notes |
|---|---|---|
| ASR | whisper large-v3 float16, beam 5, word timestamps | as mode 1 |
| Translate | G8 LLM with cross-segment context and length control; NLLB 3.3B until then | length control is what keeps dubbed segments inside their source slots |
| TTS | `auto` routing, per-segment synthesis, smart reference | XTTS per-segment path already exists; extend per-segment to F5/IndicF5 |
| Lipsync | LatentSync 1.6, 512 px, fp16, 20+ steps, guidance 1.5 | upstream GPU defaults; CPU-only knobs (IPEX dtype, DeepCache, bf16) retire |
| Restore | CodeFormer per frame | cheap on GPU |
| Post | output stabilisation on | catches residual jitter |

## Turnaround

CPU today: ~90 min per source second at 512 fp32.
GPU, one card: roughly real-time to a few times slower than real-time at 20
steps on a 96 GB card (projection from upstream reports; G3 measures it).
GPU, six cards with G5 sharding: a 60 s clip in roughly a minute.

## Where the quality work is

1. **Jitter.** The CPU investigation got mean landmark displacement from
   6.2 px to 2.1 px. Re-run `scripts/latentsync_debug/stability_metric.py`
   on GPU fp16 first; the bf16 "quantised bouncing" was CPU-autocast
   specific and may simply vanish.
2. **Timing fit.** Today a long translated segment pushes later segments
   late and only a whole-file rubberband stretch corrects it. G6 fixes
   this at the source.
3. **Frame rate.** LatentSync forces 25 fps. For 30/60 fps webcam input
   either interpolate back or run the UNet at source fps (memory permits on
   96 GB).
4. **Translation quality.** NLLB 3.3B is good; an LLM with context is
   better for idiom and register, and is the only practical way to get
   length control. G8.
5. **TTS.** Evaluate per-language against XTTS: F5-TTS v1 for EN/ZH,
   IndicF5 for Indic (finish or drop the in-flight integration), and a
   permissively licensed candidate for everything else.

## Memory

LatentSync holds every frame of the clip in RAM and GPU memory is bounded by
`num_frames=16` chunks, so host RAM, not VRAM, is the limit for long clips.
The CPU compose cap of 128 GB came from a greedy allocator on CPU; start
the GPU service at 64 GB and measure.
