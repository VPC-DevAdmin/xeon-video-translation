#!/usr/bin/env python3
"""Turn a profile_run.py capture into stage-aligned utilization numbers + py-spy hot functions."""
import json, re, sys, collections, statistics
from pathlib import Path

RUN = Path(sys.argv[1])
meta = json.loads((RUN / "meta.json").read_text())
T0 = meta["t0"]

gpu = [json.loads(l) for l in (RUN / "gpu.jsonl").read_text().splitlines() if l.strip()]
host = [json.loads(l) for l in (RUN / "host.jsonl").read_text().splitlines() if l.strip()]
cont = [json.loads(l) for l in (RUN / "containers.jsonl").read_text().splitlines() if l.strip()]
jobs = {m: json.loads((RUN / f"job-{m}.json").read_text()) for m in ("fast", "quality") if (RUN / f"job-{m}.json").exists()}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stage_intervals(job):
    """[(mode, stage, t_start, t_end)] from the watcher's transition timestamps."""
    ev = sorted(job["events"], key=lambda e: e["t"])
    out, start = [], {}
    for e in ev:
        if e["status"] == "running":
            start[e["stage"]] = e["t"]
        elif e["status"] in ("done", "failed") and e["stage"] in start:
            out.append((e["stage"], start.pop(e["stage"]), e["t"]))
    return out


def window(samples, a, b):
    return [s for s in samples if a <= s["t"] <= b]


def gpu_stats(a, b):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in window(gpu, a, b):
        for g in s["gpus"]:
            i = g["index"]
            for k in ("utilization.gpu", "utilization.memory", "utilization.encoder", "utilization.decoder", "memory.used", "power.draw", "clocks.sm"):
                v = f(g[k])
                if v is not None:
                    per[i][k].append(v)
    rows = {}
    for i, d in sorted(per.items(), key=lambda kv: int(kv[0])):
        rows[i] = {k: (round(statistics.mean(v), 1), round(max(v), 1)) for k, v in d.items() if v}
    return rows


def host_stats(a, b):
    w = window(host, a, b)
    if not w:
        return {}
    return {"cpu_busy_pct": round(statistics.mean(x["cpu_busy_pct"] for x in w if x["cpu_busy_pct"] is not None), 1),
            "cpu_busy_max": round(max(x["cpu_busy_pct"] for x in w if x["cpu_busy_pct"] is not None), 1),
            "load1": round(statistics.mean(x["load1"] for x in w), 1),
            "mem_used_gb": round(statistics.mean(x["mem_used_gb"] for x in w), 1)}


def cont_stats(a, b):
    per = collections.defaultdict(list)
    for s in window(cont, a, b):
        for c in s["containers"]:
            per[c["name"]].append(f(c["cpu"].rstrip("%")) or 0.0)
    return {n: (round(statistics.mean(v), 0), round(max(v), 0)) for n, v in per.items() if v}


report = {"meta": meta, "modes": {}}
for mode, job in jobs.items():
    j = job["job"]
    stages = []
    for name, a, b in stage_intervals(job):
        stages.append({"stage": name, "start": round(a - T0, 1), "end": round(b - T0, 1), "seconds": round(b - a, 1),
                       "gpu": gpu_stats(a, b), "host": host_stats(a, b), "containers": cont_stats(a, b)})
    report["modes"][mode] = {"job_id": j["job_id"], "status": j["status"], "error": j.get("error"),
                             "stage_durations_ms": {s["name"]: s.get("duration_ms") for s in j["stages"] if s.get("duration_ms")},
                             "stages": stages}

# idle baseline: first 4 s
report["idle"] = {"gpu": gpu_stats(T0, T0 + 4), "host": host_stats(T0, T0 + 4), "containers": cont_stats(T0, T0 + 4)}

# py-spy raw (collapsed stacks): "frame;frame;frame count"
def hot(path, top=18):
    if not path.exists():
        return None
    self_time = collections.Counter(); incl = collections.Counter(); total = 0
    for line in path.read_text().splitlines():
        m = re.match(r"^(.*) (\d+)$", line)
        if not m:
            continue
        frames, n = m.group(1).split(";"), int(m.group(2)); total += n
        leaf = frames[-1].strip()
        self_time[leaf] += n
        for fr in set(frames):
            incl[fr.strip()] += n
    def short(fr):
        fr = re.sub(r" \((.*?)\)$", lambda m: " (" + m.group(1).split("/")[-1] + ")", fr)
        return fr[:110]
    return {"samples": total,
            "self": [(short(k), round(100 * v / total, 1)) for k, v in self_time.most_common(top)],
            "inclusive": [(short(k), round(100 * v / total, 1)) for k, v in incl.most_common(60)
                          if not k.startswith("<module>") and "runpy" not in k and "threading" not in k][:top]}

report["pyspy"] = {p.stem.replace("pyspy-", ""): hot(p) for p in sorted(RUN.glob("pyspy-*.txt"))}
# timeline for charting: per-second gpu util per index + host cpu
report["timeline"] = {
    "t": [round(s["t"] - T0, 1) for s in gpu],
    "gpu_util": {g["index"]: [f(x["gpus"][int(g["index"])]["utilization.gpu"]) if int(g["index"]) < len(x["gpus"]) else None for x in gpu] for g in gpu[0]["gpus"]},
    "gpu_mem_gb": {g["index"]: [round((f(x["gpus"][int(g["index"])]["memory.used"]) or 0) / 1024, 1) for x in gpu] for g in gpu[0]["gpus"]},
    "enc": {g["index"]: [f(x["gpus"][int(g["index"])]["utilization.encoder"]) for x in gpu] for g in gpu[0]["gpus"]},
    "dec": {g["index"]: [f(x["gpus"][int(g["index"])]["utilization.decoder"]) for x in gpu] for g in gpu[0]["gpus"]},
    "host_t": [round(s["t"] - T0, 1) for s in host],
    "host_cpu": [s["cpu_busy_pct"] for s in host],
}
(RUN / "report.json").write_text(json.dumps(report, indent=1))

# human summary
for mode, m in report["modes"].items():
    print(f"\n=== {mode}: {m['status']} {m['error'] or ''}")
    for s in m["stages"]:
        busy = {i: v["utilization.gpu"][0] for i, v in s["gpu"].items() if v.get("utilization.gpu") and v["utilization.gpu"][0] >= 5}
        encdec = {i: (v["utilization.encoder"][0], v["utilization.decoder"][0]) for i, v in s["gpu"].items() if v.get("utilization.encoder") and (v["utilization.encoder"][0] >= 2 or v["utilization.decoder"][0] >= 2)}
        print(f"  {s['stage']:<12} {s['seconds']:7.1f}s  host cpu {s['host'].get('cpu_busy_pct')}% (max {s['host'].get('cpu_busy_max')}%)  gpu-busy(mean%) {busy}  enc/dec {encdec}")
        top = sorted(s["containers"].items(), key=lambda kv: -kv[1][0])[:3]
        print(f"               containers cpu% mean/max: {top}")
print("\n=== py-spy")
for name, h in report["pyspy"].items():
    if not h:
        continue
    print(f"\n--- {name} ({h['samples']} samples) self-time top:")
    for k, v in h["self"][:12]:
        print(f"   {v:5.1f}%  {k}")
    print("   inclusive top:")
    for k, v in h["inclusive"][:12]:
        print(f"   {v:5.1f}%  {k}")
