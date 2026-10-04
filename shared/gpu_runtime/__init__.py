"""Shared GPU policy and probes. Heavy dependencies are loaded on demand."""

import functools
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def required():
    return os.getenv("GPU_REQUIRED", "").lower() in {"1", "true"} or os.getenv(
        "DEVICE", "cpu"
    ).startswith("cuda")


def require_encoder(encoder):
    if required() and encoder != "h264_nvenc":
        raise RuntimeError(
            "GPU execution requires h264_nvenc; select the CPU deployment for software encoding"
        )


def require_ort_cuda(app):
    """Check loaded sessions, not merely the providers compiled into ORT.

    CPU shape/control operators remain permitted; whole-model CPU fallback
    is rejected. ORT profiling is needed to audit individual node placement.
    """
    for name, model in app.models.items():
        providers = model.session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"{name}: loaded ONNX session is not using CUDA: {providers}"
            )
        model.session.disable_fallback()


_probe_lock = threading.Lock()
_probe_cache = {}


def capabilities(device=None, codec=True, ttl=60):
    """Bounded, cached execution smoke checks; model quality is a separate gate."""
    device = device or os.getenv("DEVICE", "cpu")
    key = (device, codec, required())
    with _probe_lock:
        cached = _probe_cache.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        report = {
            "ready": True,
            "requested_device": device,
            "checks": {},
            "gpus": [],
            "scope": "runtime smoke checks; model quality and per-node placement unvalidated",
        }
        if device.startswith("cuda") or required():
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA runtime unavailable")
                report["torch_version"] = torch.__version__
                report["cuda_version"] = torch.version.cuda
                # Probe only the device this process computes on. A multi-GPU
                # coordinator (LatentSync) must not open a CUDA context on
                # every worker card just to answer /ready. GPU_PROBE_ALL_DEVICES=1
                # restores the full sweep for a one-off inventory.
                if os.getenv("GPU_PROBE_ALL_DEVICES", "0") == "1":
                    indices = range(torch.cuda.device_count())
                else:
                    indices = [torch.cuda.current_device()]
                for index in indices:
                    with torch.cuda.device(index), torch.inference_mode():
                        x = torch.ones(
                            (32, 32), device=f"cuda:{index}", dtype=torch.float16
                        )
                        result = x @ x
                        torch.cuda.synchronize(index)
                        if not bool((result == 32).all().item()):
                            raise RuntimeError(
                                f"CUDA matrix check failed on device {index}"
                            )
                        p = torch.cuda.get_device_properties(index)
                        report["gpus"].append(
                            {
                                "ordinal": index,
                                "name": p.name,
                                "uuid": str(getattr(p, "uuid", "unavailable")),
                                "vram_bytes": p.total_memory,
                                "capability": [p.major, p.minor],
                            }
                        )
                report["checks"]["cuda_fp16_kernel"] = True
            except Exception as exc:
                report["checks"]["cuda_fp16_kernel"] = False
                report["cuda_error"] = str(exc)[-500:]
            if codec:
                try:
                    with tempfile.TemporaryDirectory(prefix="gpu-codec-") as root:
                        path = str(Path(root) / "probe.mp4")
                        commands = [
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-nostdin",
                                "-y",
                                "-f",
                                "lavfi",
                                "-i",
                                "color=s=256x256:r=25",
                                "-frames:v",
                                "4",
                                "-c:v",
                                "h264_nvenc",
                                "-pix_fmt",
                                "yuv420p",
                                path,
                            ],
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-nostdin",
                                "-hwaccel",
                                "cuda",
                                "-hwaccel_output_format",
                                "cuda",
                                "-i",
                                path,
                                "-f",
                                "null",
                                "-",
                            ],
                        ]
                        for name, command in zip(("nvenc", "nvdec"), commands):
                            result = subprocess.run(
                                command, capture_output=True, timeout=15
                            )
                            report["checks"][name] = result.returncode == 0
                            if result.returncode:
                                raise RuntimeError(
                                    result.stderr.decode(errors="replace")[-500:]
                                )
                except Exception as exc:
                    report["checks"].setdefault("nvenc", False)
                    report["checks"].setdefault("nvdec", False)
                    report["codec_error"] = str(exc)[-500:]
        report["ready"] = all(report["checks"].values())
        _probe_cache[key] = (time.monotonic(), report)
        return report


@contextmanager
def span(stage, *, device=None, **fields):
    """Structured wall timings; opt-in CUDA events for hardware profiling.

    Callers supply identifiers/counters only, never transcripts, paths or keys.
    CUDA timings cover the current stream only. Profiling synchronizes at the
    end of the span and must not be used for normal latency measurements.
    """
    start = time.perf_counter()
    events = None
    if os.getenv("GPU_PROFILE", "0") == "1" and str(device).startswith("cuda"):
        import torch

        with torch.cuda.device(device):
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        entry = {"event": "execution", "stage": stage, "status": status, **fields}
        if events:
            import torch

            with torch.cuda.device(device):
                events[1].record()
                events[1].synchronize()
                entry["cuda_stream_ms"] = events[0].elapsed_time(events[1])
        entry["wall_ms"] = round((time.perf_counter() - start) * 1000, 3)
        logging.getLogger("gpu.execution").info(json.dumps(entry, sort_keys=True))


def traced(stage):
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with span(stage):
                return function(*args, **kwargs)

        return call

    return decorate


def ort_cuda_provider():
    """Avoid exhaustive first-frame cuDNN searches; allow explicit A/B testing."""
    search = os.environ.get("ORT_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC").upper()
    if search not in {"HEURISTIC", "EXHAUSTIVE", "DEFAULT"}:
        raise ValueError("ORT_CUDNN_CONV_ALGO_SEARCH must be HEURISTIC, EXHAUSTIVE or DEFAULT")
    return ("CUDAExecutionProvider", {"cudnn_conv_algo_search": search})
