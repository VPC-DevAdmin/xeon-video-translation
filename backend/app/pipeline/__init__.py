"""Pipeline stages for polyglot-demo.

Each stage exposes a `run(...)` function that reads from disk, writes to disk,
and returns a `StageResult`. The orchestrator wires them together and emits
progress events to a per-job queue.
"""

from .orchestrator import StageResult, StageStatus, run_pipeline

__all__ = ["StageResult", "StageStatus", "run_pipeline"]
