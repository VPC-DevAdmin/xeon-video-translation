"""Stage 5 — lip sync (dispatcher).

Four backends are defined:

- `none`       — skip lipsync entirely. Stage emits a passthrough result and
                 the mux stage dubs the new audio over the original video.
                 The fastest and most reliable path.

- `wav2lip`    — Wav2Lip (2020). Vendored in ``_lipsync.wav2lip``. Downloads a
                 ~400 MB CC-BY-NC 4.0 checkpoint on first use. ~30–60 s per 3 s
                 clip on a 16-core Xeon; quality is mediocre (visible softness
                 around the mouth) but works.

- `musetalk`   — stubbed in this PR. See docs/lipsync.md for why (integration
                 effort is substantial; deferred to a follow-up).

- `latentsync` — stubbed in this PR. SD 1.5 latent-diffusion based, effectively
                 unusable on CPU (~30–60 min per 3 s clip). Same deal.

Each backend's ``run`` function signature is:

    run(video_in: Path, audio_in: Path, output_path: Path,
        progress: ProgressCallback | None = None) -> LipsyncResult

where ProgressCallback receives a float in [0.0, 1.0].
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import settings

ProgressCallback = Callable[[float], None]


class LipsyncError(RuntimeError):
    pass


@dataclass
class LipsyncResult:
    backend: str
    output_path: str
    passthrough: bool  # True if backend didn't modify the video

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "output_path": self.output_path,
            "passthrough": self.passthrough,
        }


def backend_in_use(override: str | None = None) -> str:
    """Return the active backend, honoring a per-job override from the API."""
    return (override or settings.lipsync_backend).lower()


def realtime_factor(backend: str) -> float:
    """Rough CPU-wall-clock cost per second of source video, by backend.

    Used by the orchestrator to publish ETA estimates over SSE. These are
    deliberately conservative; the UI will revise them once a job starts.
    """
    return {
        "none": 0.0,
        "wav2lip": 15.0,
        "musetalk": 200.0,
        "latentsync": 600.0,
    }.get(backend, 0.0)


def run(
    backend: str,
    video_in: Path,
    audio_in: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
) -> LipsyncResult:
    backend = backend.lower()

    if backend == "none":
        return _run_passthrough(video_in, output_path)

    if backend == "wav2lip":
        from ._lipsync.wav2lip_runner import run as wav2lip_run
        return wav2lip_run(video_in, audio_in, output_path, progress=progress)

    if backend == "musetalk":
        raise LipsyncError(
            "MuseTalk is not yet wired up in this build. "
            "See docs/lipsync.md for integration notes."
        )

    if backend == "latentsync":
        raise LipsyncError(
            "LatentSync is not yet wired up in this build. "
            "Even when it is, CPU inference is effectively unusable — see "
            "docs/lipsync.md."
        )

    raise LipsyncError(f"unknown lipsync backend: {backend!r}")


def _run_passthrough(video_in: Path, output_path: Path) -> LipsyncResult:
    """No-op: copy the original video for the next stage. Audio gets replaced
    downstream by the mux stage. The resulting final.mp4 is a straight dub.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video_in, output_path)
    return LipsyncResult(
        backend="none",
        output_path=output_path.name,
        passthrough=True,
    )
