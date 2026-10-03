#!/usr/bin/env python3
"""Preflight GPU deployment and emit explicit pass/fail checks. Requires httpx."""

import argparse, json, os, shutil, subprocess
from pathlib import Path
import httpx


def check(args):
    results = []

    def record(name, ok, detail):
        results.append(dict(check=name, passed=bool(ok), detail=detail))

    executable = shutil.which("nvidia-smi")
    if executable:
        run = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        record(
            "nvidia_smi", run.returncode == 0, run.stdout.strip() or run.stderr.strip()
        )
    else:
        record("nvidia_smi", False, "nvidia-smi missing")
    headers = (
        {"Authorization": "Bearer " + os.environ["API_TOKEN"]}
        if os.getenv("API_TOKEN")
        else {}
    )
    with httpx.Client(timeout=20, headers=headers) as client:
        for name, url in [
            ("backend", args.api.rstrip("/") + "/ready"),
            ("musetalk", args.musetalk.rstrip("/") + "/health"),
            ("musetalk_dependencies", args.musetalk.rstrip("/") + "/ready"),
            ("avatar_capabilities", args.musetalk.rstrip("/") + "/avatar/capabilities"),
            ("latentsync_dependencies", args.latentsync.rstrip("/") + "/ready"),
            ("latentsync", args.latentsync.rstrip("/") + "/health"),
        ]:
            try:
                response = client.get(url)
                response.raise_for_status()
                body = response.json()
                ok = body.get("ready", True)
                if name in ("musetalk", "latentsync"):
                    ok = body.get("weights_ready") and body.get("inference_implemented")
                if name.endswith("_dependencies"):
                    ok = body.get("status") == "ok"
                if name == "avatar_capabilities":
                    ok = body.get("lip_motion") and body.get("fps") == 25
                record(name, ok, body)
            except Exception as exc:
                record(name, False, str(exc))
    report = {
        "checks": results,
        "preflight_passed": all(r["passed"] for r in results),
        "model_quality_validated": False,
        "next": "Run benchmark_modes.py on the evaluation fixtures and complete the human quality scorecard.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["preflight_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8088")
    parser.add_argument("--musetalk", default="http://localhost:8089")
    parser.add_argument("--latentsync", default="http://localhost:8090")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/bench/preflight.json")
    )
    raise SystemExit(check(parser.parse_args()))
