import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import pytest
import gpu_runtime as runtime


def test_cuda_forbids_software_encoder(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="requires h264_nvenc"):
        runtime.require_encoder("libx264")
    runtime.require_encoder("h264_nvenc")


def test_cpu_explicitly_supported(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("GPU_REQUIRED", raising=False)
    runtime.require_encoder("libx264")
    assert runtime.capabilities("cpu")["ready"]


def test_ort_checks_loaded_provider():
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="not using CUDA"):
        runtime.require_ort_cuda(
            SimpleNamespace(models={"detector": SimpleNamespace(session=session)})
        )


def test_ort_disables_whole_model_fallback():
    calls = []
    session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        disable_fallback=lambda: calls.append(True),
    )
    runtime.require_ort_cuda(
        SimpleNamespace(models={"detector": SimpleNamespace(session=session)})
    )
    assert calls == [True]


def test_failed_span_records_failure_without_swallowing(caplog):
    with caplog.at_level(logging.INFO, logger="gpu.execution"):
        with pytest.raises(ValueError):
            with runtime.span("test"):
                raise ValueError("private input must not appear")
    entry = json.loads(caplog.records[-1].message)
    assert entry["status"] == "failed" and entry["wall_ms"] >= 0
    assert "private input" not in caplog.text


spec = importlib.util.spec_from_file_location(
    "gpu_layout", Path(__file__).parents[2] / "scripts/gpu_layout.py"
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


@pytest.mark.parametrize("profile", ["dedicated", "shared"])
@pytest.mark.parametrize("quality", [True, False])
def test_layout_no_overlap(profile, quality):
    roles = layout.layout(profile, quality)
    assert sorted(i for ids in roles.values() for i in ids) == list(range(8))
    if quality:
        assert 7 not in roles["LATENTSYNC_GPUS"]
    if profile == "shared":
        assert roles["LLM_GPUS"] == [1, 2]


def test_layout_uuid_mapping_and_conflict():
    with pytest.raises(ValueError, match="both"):
        layout.validate({"a": [1], "b": [1]})
    inventory = [{"index": i, "uuid": f"GPU-test-{i}"} for i in range(8)]
    assert "LATENTSYNC_GPUS=GPU-test-5,GPU-test-6,GPU-test-7" in layout.env_text(
        layout.layout("dedicated"), inventory
    )


def test_readiness_rejects_unavailable_cuda(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    report = runtime.capabilities("cuda", codec=False, ttl=0)
    assert report["ready"] is False
    assert report["checks"]["cuda_fp16_kernel"] is False
    assert "unavailable" in report["cuda_error"]


def test_readiness_codec_failure_is_not_ready(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stderr=b"NVENC session unavailable")

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    report = runtime.capabilities("cuda", codec=True, ttl=0)
    assert report["ready"] is False and report["checks"]["nvenc"] is False
    assert len(calls) == 1 and "h264_nvenc" in calls[0]


def test_ort_search_is_configurable_and_validated(monkeypatch):
    monkeypatch.delenv("ORT_CUDNN_CONV_ALGO_SEARCH", raising=False)
    assert runtime.ort_cuda_provider()[1]["cudnn_conv_algo_search"] == "HEURISTIC"
    monkeypatch.setenv("ORT_CUDNN_CONV_ALGO_SEARCH", "exhaustive")
    assert runtime.ort_cuda_provider()[1]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    monkeypatch.setenv("ORT_CUDNN_CONV_ALGO_SEARCH", "typo")
    with pytest.raises(ValueError, match="ORT_CUDNN"):
        runtime.ort_cuda_provider()
