#!/usr/bin/env python3
"""Aggregate the A/B trial: render timelines, GPU telemetry and quality scores into one table.

Inputs (a results directory):
  <label>.json            LatentSync runner reports (events.latentsync_chunks etc.) or FlashHead
                          stream_render reports (event == flashhead_chunks)
  <label>.gpu.jsonl       per-run GPU samples (LatentSync) or flashhead-batch.gpu.jsonl (shared)
  scores/<label>.json     score_quality.py output

Writes summary.json and summary.md next to them.

Usage: analyze.py RESULTS_DIR [--playout-fps 25] [--head-starts 10,20]
"""
import argparse, json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("results")
parser.add_argument("--playout-fps", type=float, default=25.0)
parser.add_argument("--head-starts", default="10,20")
args = parser.parse_args()
root = Path(args.results)
head_starts = [float(h) for h in args.head_starts.split(",")]


def load_gpu(path, t_start, t_end):
    if not path or not Path(path).exists():
        return None
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if t_start <= r["t"] <= t_end]
    if not rows:
        return None
    per_gpu = {}
    for r in rows:
        for g in r["gpus"]:
            d = per_gpu.setdefault(g["index"], {"sm": [], "power": [], "mem": [], "enc": [], "dec": []})
            for key, field in (("sm", "utilization.gpu"), ("power", "power.draw"), ("mem", "memory.used"),
                               ("enc", "utilization.encoder"), ("dec", "utilization.decoder")):
                try:
                    d[key].append(float(g[field]))
                except (ValueError, KeyError):
                    pass
    out = {}
    for idx, d in sorted(per_gpu.items(), key=lambda kv: int(kv[0])):
        out[idx] = {"sm_mean": round(float(np.mean(d["sm"])), 1), "sm_max": round(float(np.max(d["sm"])), 1),
                    "power_mean_w": round(float(np.mean(d["power"])), 0), "mem_max_mb": round(float(np.max(d["mem"])), 0),
                    "enc_mean": round(float(np.mean(d["enc"])), 1) if d["enc"] else None}
    busy = [idx for idx, v in out.items() if v["sm_mean"] > 5]
    total_power = sum(out[i]["power_mean_w"] for i in busy) if busy else 0
    return {"samples": len(rows), "busy_gpus": busy, "busy_power_mean_w": total_power, "per_gpu": out}


def margins(chunks, num_frames, origin_key="submit"):
    """Head start needed for stall-free playout, measured from the call start and from the
    first chunk submission (the latter is what a resident persona would see)."""
    per_chunk = num_frames / args.playout_fps
    restored = [(c["chunk"], c["restored"]) for c in chunks if "restored" in c]
    if not restored:
        return {}
    first_submit = min(c.get("submit", 0.0) for c in chunks)
    h_call = max(t - i * per_chunk for i, t in restored)
    h_submit = max((t - first_submit) - i * per_chunk for i, t in restored)
    span = restored[-1][1] - first_submit
    frames = sum(c.get("frames", num_frames) for c in chunks)
    rate = frames / span if span > 0 else None
    out = {"head_start_from_call_s": round(h_call, 2), "head_start_from_first_submit_s": round(h_submit, 2),
           "first_chunk_s": round(restored[0][1], 2), "aggregate_fps": round(rate, 2) if rate else None,
           "r": round(rate / args.playout_fps, 3) if rate else None}
    worker = [c["worker_done"] - c["worker_received"] for c in chunks if "worker_done" in c]
    if worker:
        out["worker_seconds_per_chunk_mean"] = round(float(np.mean(worker)), 2)
    gen = [c["seconds"] for c in chunks if "seconds" in c]
    if gen:
        out["chunk_seconds_mean"] = round(float(np.mean(gen)), 3)
        out["chunk_seconds_p95"] = round(float(np.percentile(gen, 95)), 3)
    for h in head_starts:
        # stalls if playout at fps starting at h ever overtakes the restored chunks
        late = [i for i, t in restored if t - first_submit > h + i * per_chunk]
        out[f"stalls_with_H{int(h)}"] = len(late)
    return out


rows = []
for path in sorted(root.glob("*.json")):
    if path.name in ("summary.json",):
        continue
    data = json.loads(path.read_text())
    label = data.get("label", path.stem)
    row = {"label": label}
    if data.get("event") == "flashhead_chunks":
        row["design"] = data["design"]
        row["frames"] = data["frames"]
        row.update(margins(data["chunks"], data["num_frames"]))
        row["load_s"] = data.get("load_and_prepare_seconds")
        row["warmup_s"] = data.get("warmup_seconds")
        row["peak_torch_gb"] = round(data["peak_torch_allocated_bytes"] / 1e9, 2)
        t0 = data["t0_wall"]; t1 = t0 + data["last_chunk_restored_seconds"]
        row["gpu"] = load_gpu(root / "flashhead-batch.gpu.jsonl", t0, t1)
    elif "events" in data:
        row["design"] = data.get("design", "latentsync")
        chunks = (data["events"].get("latentsync_chunks") or [{}])[-1]
        stages = (data["events"].get("latentsync_stages") or [{}])[-1]
        finish = (data["events"].get("latentsync_finish") or [{}])[-1]
        row["frames"] = chunks.get("frames")
        row["steps"] = data["request"].get("num_inference_steps")
        row["wall_s"] = data.get("wall_seconds")
        row["prepare_s"] = data.get("prepare_seconds")
        if chunks.get("chunks"):
            row.update(margins(chunks["chunks"], chunks["num_frames"]))
            row["persona_cache_hits"] = chunks.get("persona_cache_hits")
        row["stage_decode_audio_face_s"] = stages.get("decode_audio_face_seconds")
        row["stage_wait_workers_s"] = stages.get("wait_for_worker_seconds")
        row["stage_write_s"] = finish.get("write_seconds")
        t0 = data.get("t0_wall"); t1 = t0 + data.get("wall_seconds", 0)
        local = root / f"{label}.gpu.jsonl"
        row["gpu"] = load_gpu(local if local.exists() else data.get("gpu_samples"), t0, t1) if t0 else None
    else:
        continue
    score_path = root / "scores" / f"{label}.json"
    if score_path.exists():
        sc = json.loads(score_path.read_text())
        row["quality"] = {
            "faces_detected": f'{sc.get("faces_detected")}/{sc.get("frames")}',
            "identity_vs_portrait_mean": (sc.get("identity_vs_portrait") or {}).get("mean"),
            "identity_vs_portrait_p05": (sc.get("identity_vs_portrait") or {}).get("p05"),
            "identity_vs_source_mean": (sc.get("identity_vs_source_video") or {}).get("mean"),
            "identity_drift": sc.get("identity_drift_first_to_last_quarter"),
            "sharp_face": (sc.get("sharpness_face_laplacian_var") or {}).get("mean"),
            "sharp_mouth": (sc.get("sharpness_mouth_laplacian_var") or {}).get("mean"),
            "temporal_diff": (sc.get("temporal_face_mean_abs_diff") or {}).get("mean"),
            "temporal_diff_p95": (sc.get("temporal_face_mean_abs_diff") or {}).get("p95"),
            "flicker": sc.get("mouth_flicker_std"),
            "sync_corr": (sc.get("sync_proxy") or {}).get("peak_correlation"),
            "sync_lag_ms": (sc.get("sync_proxy") or {}).get("lag_ms"),
            "syncnet_cos0": (sc.get("syncnet") or {}).get("cosine_at_zero"),
            "syncnet_margin": (sc.get("syncnet") or {}).get("margin_over_shifted"),
            "syncnet_offset": (sc.get("syncnet") or {}).get("best_offset_frames"),
            "size": f'{sc.get("width")}x{sc.get("height")}',
        }
    rows.append(row)

(root / "summary.json").write_text(json.dumps(rows, indent=1))

lines = ["| run | design | frames | first chunk s | H from submit s | agg fps | r | stalls H10/H20 | busy GPUs | W |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
for r in rows:
    g = r.get("gpu") or {}
    lines.append(f'| {r["label"]} | {r.get("design")} | {r.get("frames")} | {r.get("first_chunk_s")} | '
                 f'{r.get("head_start_from_first_submit_s")} | {r.get("aggregate_fps")} | {r.get("r")} | '
                 f'{r.get("stalls_with_H10")}/{r.get("stalls_with_H20")} | {",".join(g.get("busy_gpus", []))} | {g.get("busy_power_mean_w")} |')
lines += ["", "| run | faces | id portrait | id source | drift | sharp face | sharp mouth | temporal | flicker | sync r (lag ms) | syncnet cos0 (margin, offset) |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
for r in rows:
    q = r.get("quality")
    if not q:
        continue
    lines.append(f'| {r["label"]} | {q["faces_detected"]} | {q["identity_vs_portrait_mean"]} | {q["identity_vs_source_mean"]} | '
                 f'{q["identity_drift"]} | {q["sharp_face"]} | {q["sharp_mouth"]} | {q["temporal_diff"]} | {q["flicker"]} | '
                 f'{q["sync_corr"]} ({q["sync_lag_ms"]}) | {q["syncnet_cos0"]} ({q["syncnet_margin"]}, {q["syncnet_offset"]}) |')
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
