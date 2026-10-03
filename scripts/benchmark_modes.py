#!/usr/bin/env python3
"""Benchmark explicit modes and preserve provenance plus artifacts for quality review.

Requires httpx and ffprobe. Run on the GPU host; this does not infer quality
from latency. No environment secrets are written to the result log.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def probe(path):
    value = command(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    )
    return json.loads(value) if value else None


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = command("git", "rev-parse", "HEAD")
    diff = command("git", "diff", "HEAD") or ""
    provenance = {
        "commit": revision,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked_files": command("git", "ls-files", "--others", "--exclude-standard"),
        "fixture_sha256": hashlib.file_digest(
            args.fixture.open("rb"), "sha256"
        ).hexdigest(),
        "fixture": str(args.fixture),
        "input_media": probe(args.fixture),
        "tag": args.tag,
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }
    failed = False
    with httpx.Client(
        base_url=args.api.rstrip("/"),
        timeout=120,
        headers={"Authorization": "Bearer " + os.environ["API_TOKEN"]}
        if os.getenv("API_TOKEN")
        else {},
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        provenance["backend"] = health.json()
        # Explicit knobs supplied by the operator, never the contents of .env.
        provenance["configuration"] = dict(item.split("=", 1) for item in args.config)
        fingerprint = json.dumps(provenance["configuration"], sort_keys=True).encode()
        provenance["configuration_sha256"] = hashlib.sha256(fingerprint).hexdigest()
        for mode in args.modes:
            for repetition in range(args.repeats):
                started = time.monotonic()
                with args.fixture.open("rb") as video:
                    response = client.post(
                        "/jobs",
                        data={
                            "mode": mode,
                            "target_language": args.target,
                            "options_json": args.options,
                        },
                        files={"video": (args.fixture.name, video, "video/mp4")},
                    )
                response.raise_for_status()
                job_id = response.json()["job_id"]
                samples = []
                while True:
                    status = client.get(f"/jobs/{job_id}")
                    status.raise_for_status()
                    job = status.json()
                    if job["status"] in ("completed", "failed", "cancelled"):
                        break
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError(
                            f"job {job_id} exceeded benchmark timeout; inspect or cancel it"
                        )
                    metric = client.get("/metrics")
                    if metric.is_success:
                        samples.append(
                            {
                                "elapsed": time.monotonic() - started,
                                **metric.json().get("resources", {}),
                            }
                        )
                    time.sleep(1)
                record = {
                    **provenance,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "resources": samples,
                    "job_options": json.loads(args.options),
                    "mode": mode,
                    "repetition": repetition,
                    "wall_seconds": time.monotonic() - started,
                    "job": job,
                    "quality_review": {
                        "translation_accuracy": None,
                        "missing_speech": None,
                        "voice_identity": None,
                        "lip_sync": None,
                        "temporal_flicker": None,
                    },
                }
                if job.get("started_at"):
                    record["queue_seconds"] = (
                        datetime.fromisoformat(job["started_at"])
                        - datetime.fromisoformat(job["created_at"])
                    ).total_seconds()
                if job["status"] == "completed":
                    folder = args.output / job_id
                    folder.mkdir(exist_ok=True)
                    for name in (
                        "final.mp4",
                        "transcript.json",
                        "translation.json",
                        "translated_audio.timing.json",
                    ):
                        with client.stream(
                            "GET", f"/jobs/{job_id}/artifacts/{name}"
                        ) as artifact:
                            if artifact.status_code == 404:
                                continue
                            artifact.raise_for_status()
                            with (folder / name).open("wb") as out:
                                for chunk in artifact.iter_bytes():
                                    out.write(chunk)
                    record["output_media"] = probe(folder / "final.mp4")
                    record["artifacts"] = str(folder)
                else:
                    failed = True
                with (args.output / "results.jsonl").open("a") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"{mode} {job_id}: {job['status']} in {record['wall_seconds']:.1f}s",
                    flush=True,
                )
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--api", default="http://localhost:8088")
    parser.add_argument("--target", default="es")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fast", "quality", "dub"],
        default=["fast", "quality", "dub"],
    )
    parser.add_argument(
        "--options", default="{}", help="JobOptions JSON, e.g. windowed_lipsync"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Nonsecret KEY=VALUE experiment setting; repeated",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench/modes"))
    raise SystemExit(run(parser.parse_args()))
