"""Exercise the real pool protocol without loading CUDA/model libraries."""

import ast
from pathlib import Path
import queue
import time
import uuid
from typing import Any

path = Path(__file__).parents[1] / "app/latentsync_driver/shard_workers.py"
module = ast.parse(path.read_text())
node = next(
    n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "DenoisePool"
)
namespace = {"time": time, "uuid": uuid, "Any": Any}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
DenoisePool = namespace["DenoisePool"]


class Worker:
    def is_alive(self):
        return True


def test_previous_attempt_result_is_ignored():
    output = queue.Queue()
    pool = DenoisePool([Worker()], [queue.Queue()], output, [0])
    pool.begin_job(20, 1.5, True, 0)
    old = pool.job_id
    pool.begin_job(20, 1.5, True, 0)
    output.put(("done", old, 0, "stale pixels"))
    output.put(("done", pool.job_id, 0, "current pixels"))
    assert pool.result(0, timeout_s=0.1) == "current pixels"


def test_out_of_order_results_keep_chunk_identity():
    output = queue.Queue()
    pool = DenoisePool([Worker()], [queue.Queue()], output, [0])
    pool.begin_job(20, 1.5, True, 0)
    output.put(("done", pool.job_id, 1, "second"))
    output.put(("done", pool.job_id, 0, "first"))
    assert pool.result(0, timeout_s=0.1) == "first"
    assert pool.result(1, timeout_s=0.1) == "second"


def test_done_message_timing_is_kept_per_chunk():
    output = queue.Queue()
    pool = DenoisePool([Worker()], [queue.Queue()], output, [0])
    pool.begin_job(10, 1.5, True, 0)
    output.put(("done", pool.job_id, 0, "pixels", {"received": 1.0, "done": 2.5, "device": 3, "persona_cached": True}))
    assert pool.result(0, timeout_s=0.1) == "pixels"
    assert pool.timings[0]["done"] == 2.5 and pool.timings[0]["persona_cached"] is True
    pool.begin_job(10, 1.5, True, 0)
    assert pool.timings == {}


def _load(name):
    found = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)
    space = {}
    exec(compile(ast.Module(body=[found], type_ignores=[]), str(path), "exec"), space)
    return space[name]


def test_persona_cache_is_lru_and_counts_hits():
    cache = _load("_PersonaCache")(max_entries=2)
    assert cache.get(("p", 0)) is None
    cache.put(("p", 0), "a")
    cache.put(("p", 1), "b")
    assert cache.get(("p", 0)) == "a"      # touch 0 so 1 is the oldest
    cache.put(("p", 2), "c")               # evicts 1
    assert cache.get(("p", 1)) is None
    assert cache.get(("p", 2)) == "c"
    assert (cache.hits, cache.misses, len(cache)) == (2, 2, 2)
