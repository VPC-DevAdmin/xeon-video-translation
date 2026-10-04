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
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
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


def untracked_source_manifest():
    """Hash untracked source without logging contents or reading model artifacts."""
    root = command("git", "rev-parse", "--show-toplevel")
    names = command("git", "ls-files", "--others", "--exclude-standard", "-z")
    result = {}
    if root and names:
        for name in names.split("\0"):
            path = Path(root) / name
            if (name and not path.is_symlink() and path.is_file()
                    and (path.suffix in {".py", ".sh", ".toml", ".yaml", ".yml", ".ts", ".tsx", ".js", ".json"}
                         or path.name.startswith("Dockerfile"))
                    and not any(part in {"artifacts", "models", "node_modules", ".venv"} for part in path.parts)):
                with path.open("rb") as source:
                    result[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    supplied = json.loads(args.source_provenance.read_text()) if getattr(args, "source_provenance", None) else None
    revision = command("git", "rev-parse", "HEAD")
    diff = command("git", "diff", "--binary", "HEAD") or ""
    untracked = untracked_source_manifest()
    if supplied is not None:
        required = ("commit", "tracked_diff_sha256", "untracked_source_files", "untracked_source_sha256")
        if not all(supplied.get(key) is not None for key in required):
            raise ValueError("source provenance is incomplete")
    provenance = {
        "commit": revision,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked_source_files": untracked,
        "untracked_source_sha256": hashlib.sha256(json.dumps(untracked, sort_keys=True).encode()).hexdigest(),
        "fixture_sha256": hashlib.file_digest(
            args.fixture.open("rb"), "sha256"
        ).hexdigest(),
        "fixture": str(args.fixture),
        "input_media": probe(args.fixture),
        "tag": args.tag,
        "target_language": args.target,
        "phase": args.phase,
        "hardware_profile": args.hardware_profile,
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }
    failed = False
    if supplied is not None:
        provenance.update({key: supplied[key] for key in required})
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
                try:
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
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    # A rejected or ambiguous submission is still a benchmark
                    # outcome. Do not drop it from the denominator or assume a
                    # timed-out POST means no server job was created.
                    failed = True
                    status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                    record = {
                        **provenance,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "mode": mode, "repetition": repetition,
                        "job_options": json.loads(args.options),
                        "wall_seconds": time.monotonic() - started,
                        "resources": [], "quality_review": {},
                        "job": {"job_id": None, "status": "submission_failed",
                                "error": type(exc).__name__, "stages": []},
                        "submission_http_status": status_code,
                        "server_job_state": "unknown; inspect server before retrying" if status_code is None else "submission rejected",
                    }
                    with (args.output / "results.jsonl").open("a") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"{mode}: submission failed ({type(exc).__name__}, HTTP {status_code})", flush=True)
                    continue
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
    parser.add_argument("--source-provenance", type=Path, help="Host-generated Git metadata when the benchmark container has no git executable")
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
    parser.add_argument("--phase", choices=["cold", "warm-models", "warm-artifacts", "unspecified"], default="unspecified", help="Label measured cache state; this does not reset caches")
    parser.add_argument("--hardware-profile", default="unknown", help="Allocation/topology identifier, including GPU count and co-tenants")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Nonsecret KEY=VALUE experiment setting; repeated",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench/modes"))
    raise SystemExit(run(parser.parse_args()))
