#!/usr/bin/env python3
"""Compare measured mode latency; missing human scores stay explicitly unreviewed."""

import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("tag", "untagged"), record["mode"])].append(record)
    report = []
    for (tag, mode), runs in sorted(groups.items()):
        complete = [r for r in runs if r["job"]["status"] == "completed"]
        times = sorted(r["wall_seconds"] for r in complete)
        review = [r.get("quality_review", {}) for r in complete]
        report.append(
            {
                "tag": tag,
                "mode": mode,
                "runs": len(runs),
                "completed": len(complete),
                "median_seconds": statistics.median(times) if times else None,
                "p95_seconds": times[math.ceil(0.95 * len(times)) - 1]
                if times
                else None,
                "quality_reviewed": sum(
                    bool(r) and all(v is not None for v in r.values()) for r in review
                ),
                "fixtures": sorted({r.get("fixture_sha256", "unknown") for r in runs}),
            }
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(
        [
            json.loads(line)
            for file in args.results
            for line in file.read_text().splitlines()
            if line.strip()
        ]
    )
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)
