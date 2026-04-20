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

- `latentsync` — microservice scaffold in this PR. The service is reachable
                 and returns a structured 501 explaining staging; real
                 inference lands in PR-LS-1c. CPU budget is ~10 min per
                 second of source video (SD 1.5 latent diffusion per
                 frame) — batch workflow, not live.

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

    Used by the orchestrator to publish ETA estimates over SSE. The
    orchestrator's time-based progress ticker also reads this value to
    estimate progress mid-run (see _tick_lipsync_progress); a wildly
    wrong factor here is the reason the bar plateaus too early and
    sits at 98% for most of the run.

    LatentSync specifically: measured numbers from a 16-core Sapphire
    Rapids Xeon at steps=20 —
      fp32, no DeepCache:      ~90 min / source-second (5400 s/s)
      bf16 IPEX, no DeepCache: ~25 min / source-second (1500 s/s)
      bf16 IPEX + DeepCache:   ~18 min / source-second (1080 s/s)
    Default container config ships with both bf16 and DeepCache on, so
    we use the 1080 number. Operators who flip either knob off via
    .env will see the bar pessimize (hit 98% earlier than the real
    finish) — acceptable, and still monotonic. If an AMX-less older
    CPU falls through to fp32, the bar will visibly plateau at 98% for
    a long tail; that's the cue to add LATENTSYNC_IPEX_DTYPE=fp32 and
    let the ticker stay honest.
    """
    return {
        "none": 0.0,
        "wav2lip": 15.0,
        "musetalk": 200.0,
        "latentsync": 1080.0,
    }.get(backend, 0.0)


def run(
    backend: str,
    video_in: Path,
    audio_in: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
    quality_overrides: dict | None = None,
) -> LipsyncResult:
    """Run the selected lipsync backend.

    `quality_overrides` is forwarded to backends that understand per-request
    tuning:
      - musetalk: blend_mode / blend_feather / face_restore* (see PR 33)
      - latentsync: num_inference_steps / guidance_scale / seed (wired in
        the payload now; actual tuning lands in PR-LS-1c)
    `none` and `wav2lip` ignore it.
    """
    backend = backend.lower()

    if backend == "none":
        return _run_passthrough(video_in, output_path)

    if backend == "wav2lip":
        from ._lipsync.wav2lip_runner import run as wav2lip_run
        return wav2lip_run(video_in, audio_in, output_path, progress=progress)

    if backend == "musetalk":
        from ._lipsync.musetalk_client import run as musetalk_run
        return musetalk_run(
            video_in, audio_in, output_path,
            progress=progress,
            quality_overrides=quality_overrides,
        )

    if backend == "latentsync":
        from ._lipsync.latentsync_client import run as latentsync_run
        return latentsync_run(
            video_in, audio_in, output_path,
            progress=progress,
            quality_overrides=quality_overrides,
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
