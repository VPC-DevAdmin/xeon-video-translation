"""Streaming-margin report from per-chunk timestamps. No torch needed."""

import ast
from pathlib import Path

path = Path(__file__).parents[1] / "app/latentsync/pipelines/lipsync_pipeline.py"
module = ast.parse(path.read_text())
node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_timeline_report")
namespace: dict = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
report = namespace["chunk_timeline_report"]


def test_min_head_start_is_the_worst_chunk_lateness():
    # 16 frames per chunk at 25 fps = 0.64 s of playout per chunk.
    times = {0: {"submit": 0.1, "result": 3.0, "restored": 4.0},
             1: {"submit": 0.2, "result": 3.5, "restored": 4.5},   # due at H + 0.64
             2: {"submit": 0.3, "result": 9.0, "restored": 9.9}}   # due at H + 1.28 -> needs H >= 8.62
    out = report(times, 16, 25, 1000.0, 48, "demo")
    assert out["first_chunk_restored_seconds"] == 4.0
    assert out["min_head_start_seconds"] == 8.62
    assert out["restored_fps_aggregate"] == round(48 / (9.9 - 0.1), 2)
    assert out["persona_key"] == "demo" and len(out["chunks"]) == 3


def test_report_without_restored_chunks_is_still_well_formed():
    out = report({0: {"submit": 0.0}}, 16, 25, 0.0, 16)
    assert "min_head_start_seconds" not in out and out["chunks"][0]["chunk"] == 0
