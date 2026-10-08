# FlashHead experiments on XE7740

These are completed isolated model trials, not a production avatar service. Recorded on 3 October 2026, RTX PRO 6000 Blackwell Server Edition (SM120), one GPU. Results include failures to meet the target.

| Candidate | Warm chunk throughput | Torch peak allocation | Decision |
|---|---:|---:|---|
| Lite, eager SDPA | 114–117 FPS | 5.35 GB | Advance to identity, lip-sync and streaming integration evaluation |
| Pro, eager SDPA | about 18 FPS | See result JSON | Below 25 FPS target |
| Pro, compiled SDPA | 21.8–21.9 FPS | 9.68 GB | Still below 25 FPS; first chunk took 150 s to compile |

Four-second audio, one portrait, 512×512 output, four inference steps, seed 42. Lite first chunk took 392 ms and model preparation 3.90 s. Rates measure model chunk generation, excluding final CPU transfer and encoding. They do not measure full voice-assistant latency or sustained WebRTC delivery. Identity, temporal quality and lip sync remain unreviewed. Lite visibly softens the test face; speed alone is insufficient for promotion.

## Exact versions and evidence

- [Official source](https://github.com/Soul-AILab/SoulX-FlashHead), commit `9bc03de06bb0de82cd6bc477804512ae06144bf2`.
- [Model](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B), revision `59119b6c681230c3eeee157e224ae1941746711e`.
- `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- PyTorch 2.7.1+cu128, transformers 4.57.3, diffusers 0.35.2, xfuser 0.4.3, xformers 0.0.31. Native PyTorch SDPA; no FlashAttention/Sage installed.
- `results/*.json`: actual model timings, hardware, revisions and memory.
- `observed-environment.txt`: installed-package inventory, **not an installable lock file** (contains inherited Conda paths). The lab inherited LatentSync dependencies; build a clean environment and requalify it before any service integration.
- Remote source, installation log/report, exact original benchmark scripts, model manifests and disclosed MP4 outputs: `/home/user/xeon-video-translation-releases/codex-20261003T211852Z/flashhead`.

## Replay

Use an isolated container on an explicitly allocated GPU. Do not install these dependencies into the serving environments. The original lab used the LatentSync GPU image, then installed the package versions listed above plus mediapipe 0.10.9, accelerate 1.12.0, pyloudnorm, easydict, ftfy, loguru and imageio-ffmpeg. Resolve and record the full environment again; these top-level versions alone are not a reproducibility guarantee. Unused inherited LatentSync metadata and decord were removed before the final dependency check. The idle lab container is retained; an attempted image snapshot failed because Docker could not find a content digest, so there is no claimed portable lab image.

Mount a writable `/experiment-models` directory and a separate output directory. Check out the exact source revision, then run the supplied scripts from that source directory with its root on `PYTHONPATH`:

```sh
python /path/to/prepare_models.py
PYTHONPATH=. python /path/to/benchmark.py --model lite \
  --image /fixtures/portrait.png --audio /fixtures/speech-16k-mono.wav \
  --output /results/lite
# Repeat with --model pro, and then --model pro --compile.
```

The supplied benchmark is the executed script with configurable paths and input hashes added. That argument refactor has syntax validation; the recorded runs used the original fixed paths saved on XE7740. The audio must be PCM16 mono at the model's required sample rate. Tail padding is removed from the final video. `peak_torch_allocated_bytes` excludes driver, non-PyTorch and codec allocations.

Next: multiple identities, actual webcam motion, speech/voice review and audiovisual sync; then a bounded service adapter with cancellation, timestamp continuity and WebRTC tests. Two-GPU Pro and optimized attention are untested experiments, not assumed improvements.


## Realism follow-up

See [the measured review](../../docs/gpu/realism-review-2026-10-03.md) and `realism-trial/` for executed voice comparisons and diagnostics. The benchmark now accepts `--playback-audio` for a native-rate PCM16 mono master; `--audio` remains its timeline-matched 16 kHz conditioning copy. Pass `--model-manifest` when the manifest is outside the model root. This avoids delivering the reduced-bandwidth conditioning audio. The GPU preview and four timeline regressions passed; human naturalness is still unqualified.
