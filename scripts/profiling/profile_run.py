#!/usr/bin/env python3
"""Whole-flow profiler: GPU/CPU/container sampling + py-spy captures around one fast and one quality job.

Writes everything under OUT. Bounded: stops after MAX_SECONDS even if jobs hang.
"""
import json, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FIXTURE = sys.argv[2]
MAX_SECONDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
API = "http://localhost:8088"
CONTAINERS = ["polyglot-backend", "polyglot-lipsync-fast", "polyglot-lipsync-latentsync", "polyglot-llm-gpu",
              "polyglot-ingest-webrtc", "xeon-video-translation-backend-avatar-1", "xeon-video-translation-musetalk-avatar-1"]
stop = threading.Event()
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def gpu_sampler():
    q = "index,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate"
    with (OUT / "gpu.jsonl").open("w") as f:
        while not stop.is_set():
            t = time.time()
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            rows = [dict(zip(q.split(","), [c.strip() for c in line.split(",")])) for line in r.stdout.strip().splitlines()]
            f.write(json.dumps({"t": t, "gpus": rows}) + "\n"); f.flush()
            time.sleep(max(0, 1.0 - (time.time() - t)))


def read_cpu():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts)); idle = vals[3] + vals[4]
    return sum(vals), idle


def host_sampler():
    prev = read_cpu()
    with (OUT / "host.jsonl").open("w") as f:
        while not stop.is_set():
            time.sleep(1.0)
            cur = read_cpu(); dt = cur[0] - prev[0]; didle = cur[1] - prev[1]; prev = cur
            mem = {}
            with open("/proc/meminfo") as m:
                for line in m:
                    k, v = line.split(":"); mem[k] = int(v.split()[0])
            la = os.getloadavg()
            f.write(json.dumps({"t": time.time(), "cpu_busy_pct": 100 * (1 - didle / dt) if dt else None,
                                "load1": la[0], "mem_used_gb": (mem["MemTotal"] - mem["MemAvailable"]) / 1e6,
                                "swap_used_gb": (mem["SwapTotal"] - mem["SwapFree"]) / 1e6}) + "\n"); f.flush()


def docker_sampler():
    with (OUT / "containers.jsonl").open("w") as f:
        while not stop.is_set():
            t = time.time()
            r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *CONTAINERS], capture_output=True, text=True)
            rows = []
            for line in r.stdout.strip().splitlines():
                try:
                    d = json.loads(line); rows.append({"name": d["Name"], "cpu": d["CPUPerc"], "mem": d["MemUsage"], "pids": d.get("PIDs")})
                except Exception:
                    pass
            f.write(json.dumps({"t": t, "containers": rows}) + "\n"); f.flush()
            time.sleep(max(0, 3.0 - (time.time() - t)))


def api(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


def submit(mode):
    import mimetypes, uuid
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in (("mode", mode), ("target_language", "es"), ("options_json", "{}")):
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    data = Path(FIXTURE).read_bytes()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{Path(FIXTURE).name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API + "/jobs", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["job_id"]


def pyspy(container, pid, seconds, name):
    """Sample a process inside a container; py-spy is installed on first use."""
    def run():
        subprocess.run(["docker", "exec", "--privileged", container, "sh", "-c",
                        "command -v py-spy >/dev/null || pip install -q py-spy >/dev/null 2>&1; "
                        f"py-spy record -p {pid} -d {seconds} --format raw --nonblocking -o /tmp/{name}.txt >/tmp/{name}.log 2>&1; "
                        f"py-spy dump -p {pid} > /tmp/{name}.dump 2>&1 || true"], capture_output=True)
        for ext in ("txt", "log", "dump"):
            subprocess.run(["docker", "cp", f"{container}:/tmp/{name}.{ext}", str(OUT / f"pyspy-{name}.{ext}")], capture_output=True)
        log(f"py-spy {name} done")
    th = threading.Thread(target=run, daemon=True); th.start(); return th


def container_pids(container, pattern):
    r = subprocess.run(["docker", "exec", container, "sh", "-c", f"pgrep -f '{pattern}' | head -5"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def watch(job_id, mode, captures):
    """Record stage transitions with timestamps; fire py-spy captures on first entry to a stage."""
    seen = {}
    fired = set()
    events = []
    while not stop.is_set():
        j = api(f"/jobs/{job_id}")
        for s in j["stages"]:
            key = (s["name"], s["status"])
            if key not in seen:
                seen[key] = time.time(); events.append({"t": seen[key], "stage": s["name"], "status": s["status"]})
                log(f"{mode} {job_id[:8]} {s['name']} -> {s['status']}")
                if s["status"] == "running" and s["name"] in captures and s["name"] not in fired:
                    fired.add(s["name"]); captures[s["name"]]()
        if j["status"] in ("completed", "failed", "cancelled"):
            (OUT / f"job-{mode}.json").write_text(json.dumps({"job": j, "events": events}, indent=2))
            log(f"{mode} {j['status']} {j.get('error') or ''}")
            return j
        time.sleep(1)


def main():
    for fn in (gpu_sampler, host_sampler, docker_sampler):
        threading.Thread(target=fn, daemon=True).start()
    (OUT / "meta.json").write_text(json.dumps({"t0": T0, "fixture": FIXTURE, "api": API}))
    time.sleep(5)  # idle baseline
    threads = []

    def capture_fast_lipsync():
        time.sleep(25)  # let the first window reach denoise
        coord = container_pids("polyglot-lipsync-fast", "uvicorn")
        workers = container_pids("polyglot-lipsync-fast", "multiprocessing.spawn")
        if coord: threads.append(pyspy("polyglot-lipsync-fast", coord[0], 60, "fast-coordinator"))
        if workers: threads.append(pyspy("polyglot-lipsync-fast", workers[0], 60, "fast-worker"))
        threads.append(pyspy("polyglot-backend", 1, 60, "backend-during-lipsync"))

    def capture_tts():
        threads.append(pyspy("polyglot-backend", 1, 25, "backend-tts"))

    def capture_quality_lipsync():
        time.sleep(40)
        coord = container_pids("polyglot-lipsync-latentsync", "uvicorn")
        workers = container_pids("polyglot-lipsync-latentsync", "multiprocessing.spawn")
        if coord: threads.append(pyspy("polyglot-lipsync-latentsync", coord[0], 60, "batch-coordinator"))
        if workers: threads.append(pyspy("polyglot-lipsync-latentsync", workers[0], 60, "batch-worker"))

    deadline = T0 + MAX_SECONDS
    job = submit("fast"); log(f"submitted fast {job}")
    watch(job, "fast", {"tts": capture_tts, "lipsync": capture_fast_lipsync})
    if time.time() < deadline:
        job = submit("quality"); log(f"submitted quality {job}")
        watch(job, "quality", {"lipsync": capture_quality_lipsync})
    for th in threads: th.join(timeout=120)
    time.sleep(5)
    stop.set(); time.sleep(2)
    log("done")


if __name__ == "__main__":
    main()
