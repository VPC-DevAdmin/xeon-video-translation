"""Shared face track: slicing, gap handling, cache round trip. No CUDA/model libraries."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
from latentsync_driver import face_track as ft  # noqa: E402


def _track(n):
    return np.stack([np.full((3, 2), i, dtype=np.float32) for i in range(n)])


def test_slice_inside_track_and_padded_tail():
    track = _track(10)
    assert ft.slice_for_window(track, 2, 3)[:, 0, 0].tolist() == [2, 3, 4]
    padded = ft.slice_for_window(track, 8, 5)
    assert padded.shape == (5, 3, 2) and padded[:, 0, 0].tolist() == [8, 9, 9, 9, 9]
    beyond = ft.slice_for_window(track, 20, 2)
    assert beyond[:, 0, 0].tolist() == [9, 9]
    with pytest.raises(ValueError):
        ft.slice_for_window(track, -1, 2)


def test_fill_missing_forward_and_back_fills():
    a, b = np.ones((3, 2)), np.full((3, 2), 2.0)
    filled, missing = ft.fill_missing([None, a, None, b, None])
    assert missing == [0, 2, 4]
    assert [f[0, 0] for f in filled] == [1, 1, 1, 2, 2]
    assert ft.fill_missing([None, None]) == ([], [0, 1])


def test_smooth_is_identity_for_window_one_or_short_tracks():
    track = _track(4)
    assert np.array_equal(ft.smooth(track, 1), track)
    assert np.array_equal(ft.smooth(_track(2), 5), _track(2))


def test_smooth_savgol_preserves_linear_motion():
    pytest.importorskip("scipy")
    track = _track(25)
    out = ft.smooth(track, 5)
    assert out.shape == track.shape
    assert np.allclose(out, track, atol=1e-3)


def test_load_or_build_detects_once_then_hits_cache(tmp_path, monkeypatch):
    import gpu_runtime.media as media

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really video but hashed")
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(6)]
    monkeypatch.setattr(media, "iter_frames", lambda path, **kw: iter(frames))
    calls = []

    def extract(frame):
        calls.append(1)
        return None if len(calls) == 3 else np.full((3, 2), float(len(calls)), np.float32)

    kwargs = dict(model_cache_dir=tmp_path / "models", fps=25, extract=extract,
                  smooth_window=1, max_miss_ratio=0.5, frame_budget_bytes=10**9)
    track = ft.load_or_build(source, **kwargs)
    first = track["landmarks"]
    assert first.shape == (6, 3, 2) and len(calls) == 6
    assert first[2, 0, 0] == 2.0  # gap carried forward from frame 2
    assert track["visible"].tolist() == [True, True, False, True, True, True]
    second = ft.load_or_build(source, **kwargs)
    assert len(calls) == 6 and np.array_equal(first, second["landmarks"])
    assert np.array_equal(track["visible"], second["visible"])
    assert list((tmp_path / "models" / "cache" / "latentsync_tracks").glob("*.npz"))


def test_load_or_build_rejects_faceless_clips(tmp_path, monkeypatch):
    import gpu_runtime.media as media

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(media, "iter_frames", lambda path, **kw: iter([np.zeros((2, 2, 3), np.uint8)] * 4))
    with pytest.raises(RuntimeError, match="no face detected"):
        ft.load_or_build(source, model_cache_dir=tmp_path, fps=25, extract=lambda f: None,
                         smooth_window=1, max_miss_ratio=0.5, frame_budget_bytes=10**9)


def _load_trimmed_pack_root():
    """face_detector imports torch/insightface; lift just the helper (same trick as test_worker_protocol)."""
    import ast, types
    path = Path(__file__).parents[1] / "app/latentsync/utils/face_detector.py"
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_trimmed_pack_root")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return types.SimpleNamespace(_trimmed_pack_root=ns["_trimmed_pack_root"])


def test_trimmed_insightface_pack_keeps_only_used_models(tmp_path):
    fd = _load_trimmed_pack_root()

    pack = tmp_path / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    for name, size in (("det_10g.onnx", 10), ("2d106det.onnx", 5), ("w600k_r50.onnx", 100), ("1k3d68.onnx", 80)):
        (pack / name).write_bytes(b"x" * size)
    root = fd._trimmed_pack_root(str(tmp_path), "buffalo_l", ("det_10g.onnx", "2d106det.onnx"))
    kept = sorted(p.name for p in (Path(root) / "models" / "buffalo_l").iterdir())
    assert kept == ["2d106det.onnx", "det_10g.onnx"]
    # idempotent and stable
    assert fd._trimmed_pack_root(str(tmp_path), "buffalo_l", ("det_10g.onnx", "2d106det.onnx")) == root
    # missing source file -> untouched root
    assert fd._trimmed_pack_root(str(tmp_path), "other", ("a.onnx",)) == str(tmp_path)
