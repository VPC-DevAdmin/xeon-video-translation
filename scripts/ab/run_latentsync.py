#!/usr/bin/env python3
"""A/B runner for the LatentSync persona design (host side, talks to the service on :8090).

For one audio sample and one persona clip it: starts a GPU sampler, posts /lipsync
with a persona_key, pulls the service log lines emitted during the run
(latentsync_stages, latentsync_chunks, latentsync_restore_overlapped, latentsync_finish),
remuxes the native-rate master audio onto the output, and writes OUT/<label>.json.

Usage: run_latentsync.py --label NAME --video /jobs/... --audio /jobs/... [--master /jobs/...]
                         --out /jobs/ab/out --persona KEY --steps 10 [--repeat 2]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--audio", required=True, help="16 kHz conditioning copy, visible as /jobs/... in the container")
parser.add_argument("--master", help="native-rate master to remux onto the output (container path)")
parser.add_argument("--out", default="/jobs/ab/out")
parser.add_argument("--persona", default=None)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--guidance", type=float, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeat", type=int, default=1, help="run N times (second run hits the persona cache)")
parser.add_argument("--container", default="polyglot-lipsync-latentsync")
parser.add_argument("--url", default="http://localhost:8090")
parser.add_argument("--host-out", default=os.path.expanduser("~/ab-results"))
args = parser.parse_args()

host_out = Path(args.host_out); host_out.mkdir(parents=True, exist_ok=True)


def dexec(*cmd, check=True):
    return subprocess.run(["docker", "exec", args.container, *cmd], capture_output=True, text=True, check=check)


for attempt in range(args.repeat):
    label = args.label if args.repeat == 1 else f"{args.label}-run{attempt + 1}"
    output = f"{args.out}/{label}.mp4"
    sampler = subprocess.Popen([sys.executable, str(Path(__file__).with_name("gpu_sample.py")), str(host_out / f"{label}.gpu.jsonl")])
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    body = {"video_path": args.video, "audio_path": args.audio, "output_path": output,
            "num_inference_steps": args.steps, "seed": args.seed}
    if args.guidance is not None:
        body["guidance_scale"] = args.guidance
    if args.persona:
        body["persona_key"] = args.persona
    t0 = time.time()
    req = urllib.request.Request(f"{args.url}/lipsync", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            response = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        response = {"status": "error", "code": e.code, "detail": e.read().decode()[:2000]}
    wall = time.time() - t0
    time.sleep(1.5)
    sampler.send_signal(signal.SIGTERM); sampler.wait()
    logs = subprocess.run(["docker", "logs", "--since", since, args.container], capture_output=True, text=True)
    events = {}
    for line in (logs.stdout + logs.stderr).splitlines():
        line = line.strip()
        if line.startswith("{") and '"event"' in line:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            events.setdefault(ev["event"], []).append(ev)
    remuxed = None
    if args.master and response.get("status") == "ok":
        remuxed = output.replace(".mp4", "-master.mp4")
        dexec("ffmpeg", "-v", "error", "-y", "-i", output, "-i", args.master, "-map", "0:v:0", "-map", "1:a:0",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", remuxed, check=False)
    probe = {}
    if response.get("status") == "ok":
        p = dexec("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                  "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", output, check=False)
        try:
            probe = json.loads(p.stdout)
        except ValueError:
            probe = {"raw": p.stdout[-500:]}
    report = {"label": label, "design": "latentsync-persona", "request": body, "response": response,
              "wall_seconds": round(wall, 2), "events": events, "output": output, "output_master_audio": remuxed,
              "probe": probe, "t0_wall": t0, "gpu_samples": str(host_out / f"{label}.gpu.jsonl")}
    (host_out / f"{label}.json").write_text(json.dumps(report, indent=1))
    chunks = (events.get("latentsync_chunks") or [{}])[-1]
    print(json.dumps({"label": label, "status": response.get("status"), "wall_s": round(wall, 1),
                      "first_chunk_s": chunks.get("first_chunk_restored_seconds"),
                      "min_head_start_s": chunks.get("min_head_start_seconds"),
                      "restored_fps": chunks.get("restored_fps_aggregate"),
                      "persona_cache_hits": chunks.get("persona_cache_hits")}), flush=True)
