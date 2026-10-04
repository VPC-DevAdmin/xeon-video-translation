# InfiniteTalk quality trial on XE7740

Status: experimental, no completed video. Do not route production jobs here yet.

The [official implementation](https://github.com/MeiGen-AI/InfiniteTalk) supports full diffusion sampling and distributed inference. This trial uses source commit `50aa0a94184315407a991ae804d9b58d6d311ba8`, full BF16 weights, 40 sampling steps, no quantization, no acceleration LoRA, no TeaCache, and no CPU model offload. The initial target is an 81-frame preview using the 720 size preset, followed by a one-minute streaming run only if identity and motion improve over FlashHead Pro.

## Recorded results

- Downloaded approximately 88 GB of pinned Wan, InfiniteTalk and wav2vec weights. See `model-revisions.json`.
- Fixed an obsolete `inspect.ArgSpec` import and wav2vec attention-output incompatibility. Moved wav2vec inference onto CUDA.
- Native PyTorch SDPA handles the direct attention entry point and per-frame audio attention without requiring an unsupported external attention wheel. Variable-length attention preserves query padding and key lengths.
- A BF16 GPU comparison with a masked reference produced maximum absolute difference 0.0 on the small tested input. This does not validate the whole model.
- At sequence length 65,536, a kernel probe measured 0.278 s without the padding mask versus 0.807 s with it. Removing the redundant dense mask allows native Flash SDPA; actual variable lengths are handled by slicing. These are kernel measurements, not an end-to-end speedup.
- The original masked single-GPU run took 167 s for its first diffusion step and was stopped. The revised full render has not completed.
- Four-GPU FSDP/Ulysses stalled during NCCL initialization, before loading the model. Advertised CUDA peer access was available on all four devices; actual collective communication was not established. Disabling cuMem host allocation alone did not resolve it. A transport fallback probe was launched, but its final result could not be retrieved after SSH access disappeared.

## Files and environment

Apply `native-cuda.patch` to the exact source commit in an isolated checkout. The patch includes native attention, CUDA wav2vec, and a single NVENC final encode. It still copies frames through host memory for encoding; it is not an entirely GPU-resident media pipeline.

`observed-environment.txt` is an installed package inventory, not a clean dependency lock. The lab inherited unused packages with dependency conflicts. Build and qualify a clean image before production integration.

The lab filesystem was preserved on the server as `polyglot-demo/quality-lab:20261004`, image ID `sha256:c67b2f6d084a796d375681a6caba38395b875c1fc7cbdea77466b2b0b4e9a210`. A normal Docker commit failed due to a missing parent content digest; export/import preserved the installed filesystem instead. Source, model and job bind mounts are outside that snapshot.

Remote work: `/home/user/quality-v2/InfiniteTalk`; container `polyglot-quality-multigpu`; model volume `xeon-video-translation_quality-models`. Its visible devices map to physical GPUs 4–7. Never infer availability from the container name: check actual occupancy and active jobs first.

## Replay and diagnostics

1. Check out the pinned commit, run `git apply --check native-cuda.patch`, then apply it.
2. Use `python prepare_models.py --root /quality-models` to retrieve the pinned revisions. The original download discovered revisions at run time; this saved replay script pins them.
3. Run `PYTHONPATH=. python test_attention_gpu.py` in the patched source checkout. Record GPU, PyTorch and CUDA versions alongside the result.
4. Test collectives before launching a model:

   ```sh
   timeout -k 5 60 torchrun --nproc_per_node=4 --standalone /path/to/probe_nccl.py
   ```

   The external timeout is required because initialization has ignored the process-group timeout. The revised probe prints loaded NCCL library paths and verifies reduction results; this revised diagnostic still needs a GPU run. Inspect the unexpected `NCCL 2.26.2+cuda12.2` banner in the PyTorch cu128 environment before considering host changes. Compare default transport with `NCCL_CUMEM_HOST_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` in the isolated lab. Do not alter host IOMMU, ACS or drivers as a speculative fix. See [NVIDIA's troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html).
5. After collectives pass, replay `/home/user/quality-v2/multigpu-command.json`. If distributed inference remains unusable, try the corrected attention path on a single allocated GPU without FSDP/Ulysses.
6. For a complete minute use streaming mode with a sufficient frame limit (`--max_frame_num 1500` at 25 FPS), and verify actual audio/video duration and frame count after encoding. The upstream default of 1,000 frames is insufficient.

Promotion requires a completed video, identity/motion/lip-sync review, no missing narration, exact duration verification, memory measurements, clean dependencies and cancellation/recovery tests. Larger weights and more steps alone do not establish better quality.
