"""Process-per-GPU denoise workers for LatentSync (GPU track, G5).

Why processes: the thread-based sharding in ``lipsync_pipeline.py`` capped
at ~2x on four GPUs. Each 20-step UNet pass launches hundreds of small
kernels, and four Python threads doing that contend for the interpreter
lock, so the replicas sat near 58% utilisation. One process per GPU gives
each replica its own interpreter.

Each worker owns a UNet (fp16 by default), a VAE for decoding its own
chunks, and a DDIM scheduler. The main pipeline process keeps doing what
it did: conditioning on cuda:0 (VAE encodes of masks / reference), then
paste + restore. Per chunk it ships ~2 MB of conditioning tensors to a
worker as CPU tensors through ``torch.multiprocessing`` shared memory and
gets back the decoded 16x3x512x512 fp16 pixels (~25 MB).

The pool is persistent: it is created with the cached pipeline and lives
for the service's lifetime, so the ~5-10 s model load per worker is paid
once. ``LATENTSYNC_SHARD_MODE=thread`` selects the old in-process path.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

log = logging.getLogger("latentsync.shard")


def _worker_main(dev_index: int, in_q, out_q, build: dict) -> None:  # pragma: no cover - runs in child
    """Child entry point. Loads models, then serves chunks until told to stop."""
    try:
        app_root = build["app_root"]
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.cuda.set_device(dev_index)
        device = torch.device(f"cuda:{dev_index}")
        dtype = getattr(torch, build["dtype"])

        from diffusers import AutoencoderKL, DDIMScheduler
        from einops import rearrange

        from latentsync.models.unet import UNet3DConditionModel

        t0 = time.perf_counter()
        unet, _ = UNet3DConditionModel.from_pretrained(build["unet_config"], build["unet_ckpt"], device=str(device))
        unet = unet.to(dtype=dtype).eval()
        vae = AutoencoderKL.from_pretrained(build["vae_repo"], torch_dtype=dtype).to(device).eval()
        vae.config.scaling_factor = build["vae_scaling_factor"]
        vae.config.shift_factor = build["vae_shift_factor"]
        scheduler = DDIMScheduler.from_pretrained(build["scheduler_dir"])
        if build.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        out_q.put(("ready", dev_index, time.perf_counter() - t0))

        timesteps = None
        guidance = 1.5
        cfg = True
        eta = 0.0
        accepts_eta = "eta" in scheduler.step.__code__.co_varnames

        while True:
            msg = in_q.get()
            if msg is None:
                return
            kind = msg[0]
            if kind == "job":
                _, steps, guidance, cfg, eta = msg
                scheduler.set_timesteps(int(steps), device=device)
                timesteps = scheduler.timesteps
                continue
            if kind != "chunk":
                continue
            _, i, cond = msg
            try:
                with torch.no_grad():
                    lat, ml, mil, rl, ae = (None if x is None else x.to(device, non_blocking=True) for x in cond)
                    step_kwargs = {"eta": eta} if accepts_eta else {}
                    for t in timesteps:
                        unet_input = torch.cat([lat] * 2) if cfg else lat
                        unet_input = scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, ml, mil, rl], dim=1)
                        noise_pred = unet(unet_input, t, encoder_hidden_states=ae).sample
                        if cfg:
                            uncond, audio = noise_pred.chunk(2)
                            noise_pred = uncond + guidance * (audio - uncond)
                        lat = scheduler.step(noise_pred, t, lat, **step_kwargs).prev_sample
                    z = lat / vae.config.scaling_factor + vae.config.shift_factor
                    z = rearrange(z, "b c f h w -> (b f) c h w")
                    pixels = vae.decode(z).sample
                    torch.cuda.synchronize(device)
                out_q.put(("done", i, pixels.to("cpu")))
            except Exception as e:  # report, keep serving
                out_q.put(("error", i, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))
    except Exception as e:
        out_q.put(("fatal", dev_index, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))


class DenoisePool:
    """One persistent worker process per CUDA device."""

    def __init__(self, workers: list, in_queues: list, out_q, devices: list[int]):
        self.workers = workers
        self.in_queues = in_queues
        self.out_q = out_q
        self.devices = devices
        self._results: dict[int, Any] = {}
        self._next = 0
        self.size = len(workers)

    @classmethod
    def start(cls, devices: list[int], build: dict, timeout_s: float = 600.0) -> "DenoisePool":
        torch.multiprocessing.set_sharing_strategy("file_system")
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        in_queues = [ctx.Queue() for _ in devices]
        workers = []
        for d, q in zip(devices, in_queues):
            p = ctx.Process(target=_worker_main, args=(d, q, out_q, build), daemon=True, name=f"latentsync-denoise-{d}")
            p.start()
            workers.append(p)
        pool = cls(workers, in_queues, out_q, devices)
        ready = 0
        deadline = time.perf_counter() + timeout_s
        while ready < len(devices):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                pool.close()
                raise RuntimeError("denoise workers did not become ready in time")
            msg = out_q.get(timeout=remaining)
            if msg[0] == "ready":
                ready += 1
                log.info("denoise worker on cuda:%d ready (models loaded in %.1fs)", msg[1], msg[2])
            elif msg[0] == "fatal":
                pool.close()
                raise RuntimeError(f"denoise worker on cuda:{msg[1]} failed to start: {msg[2]}")
        return pool

    def alive(self) -> bool:
        return all(p.is_alive() for p in self.workers)

    def begin_job(self, steps: int, guidance: float, cfg: bool, eta: float) -> None:
        self._results.clear()
        self._next = 0
        for q in self.in_queues:
            q.put(("job", int(steps), float(guidance), bool(cfg), float(eta)))

    def submit(self, i: int, cond: tuple) -> None:
        """`cond` = (latents, mask_latents, masked_image_latents, ref_latents, audio_embeds) as CPU tensors."""
        self.in_queues[self._next % self.size].put(("chunk", i, cond))
        self._next += 1

    def result(self, i: int, timeout_s: float = 1800.0):
        """Decoded pixels for chunk `i` (CPU fp16), blocking until it lands.

        Results arrive out of order; anything not yet asked for is buffered.
        """
        import queue as _queue

        deadline = time.perf_counter() + timeout_s
        while i not in self._results:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError(f"timed out waiting for denoise chunk {i}")
            if not self.alive():
                raise RuntimeError("a denoise worker process died")
            try:
                msg = self.out_q.get(timeout=min(remaining, 5.0))
            except _queue.Empty:
                continue
            kind = msg[0]
            if kind == "done":
                self._results[msg[1]] = msg[2]
            elif kind == "error":
                raise RuntimeError(f"denoise chunk {msg[1]} failed in worker: {msg[2]}")
            elif kind == "fatal":
                raise RuntimeError(f"denoise worker cuda:{msg[1]} died: {msg[2]}")
        return self._results.pop(i)

    def close(self) -> None:
        for q in self.in_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def build_args_for(unet_config: dict, unet_ckpt: Path, scheduler_dir: Path, dtype: torch.dtype,
                   vae_repo: str = "stabilityai/sd-vae-ft-mse") -> dict:
    app_root = str(Path(__file__).resolve().parent.parent)
    return {
        "app_root": app_root,
        "unet_config": unet_config,
        "unet_ckpt": str(unet_ckpt),
        "scheduler_dir": str(scheduler_dir),
        "dtype": str(dtype).split(".")[-1],
        "vae_repo": vae_repo,
        "vae_scaling_factor": 0.18215,
        "vae_shift_factor": 0,
        "hf_home": os.environ.get("HF_HOME", ""),
    }
