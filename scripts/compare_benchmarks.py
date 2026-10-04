#!/usr/bin/env python3
"""Stratify measurements before comparing speed. Missing quality is not a pass."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

QUALITY_FIELDS = ("translation_accuracy", "missing_speech", "voice_identity", "lip_sync", "temporal_flicker")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stratum(record):
    """Configuration differs between candidates; workload and hardware must match."""
    return {
        "mode": record["mode"],
        "fixture_sha256": record.get("fixture_sha256", "unknown"),
        "target_language": record.get("target_language", "unknown"),
        "phase": record.get("phase", "unspecified"),
        "hardware_profile": record.get("hardware_profile", "unknown"),
        "observed_gpu": record.get("gpu", "unknown"),
    }


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        key = canonical({"tag": record.get("tag", "untagged"), **stratum(record),
                         "commit": record.get("commit", "unknown"),
                         "tracked_diff_sha256": record.get("tracked_diff_sha256", "unknown"),
                         "untracked_source_sha256": record.get("untracked_source_sha256", "unknown"),
                         "configuration": record.get("configuration", {}),
                         "job_options": record.get("job_options", {})})
        groups[key].append(record)
    report = []
    for key, runs in sorted(groups.items()):
        identity = json.loads(key)
        complete = [r for r in runs if r["job"]["status"] == "completed"]
        times = sorted(float(r["wall_seconds"]) for r in complete)
        reviewed = sum(all(r.get("quality_review", {}).get(k) is not None for k in QUALITY_FIELDS) for r in complete)
        report.append({**identity, "runs": len(runs), "completed": len(complete),
            "failed_or_cancelled": len(runs)-len(complete),
            "median_seconds": statistics.median(times) if times else None,
            "p95_seconds": times[math.ceil(.95*len(times))-1] if len(times)>=20 else None,
            "p95_status": "empirical; report sample count" if len(times)>=20 else "insufficient samples (minimum 20)",
            "quality_reviewed": reviewed,
            "quality_status": "review recorded; inspect scores" if complete and reviewed==len(complete) else "unreviewed or incomplete",
            "provenance_complete": all(identity[k] not in (None, "", "unknown", "unspecified") for k in ("fixture_sha256","target_language","phase","hardware_profile","observed_gpu","commit","tracked_diff_sha256","untracked_source_sha256")),
            "cohort_sha256": hashlib.sha256(canonical(stratum(runs[0])).encode()).hexdigest()})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize([json.loads(line) for file in args.results for line in file.read_text().splitlines() if line.strip()])
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result+"\n")
    print(result)
