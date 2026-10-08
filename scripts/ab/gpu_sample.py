#!/usr/bin/env python3
"""Sample nvidia-smi once a second into a JSONL file until killed (host side).

Usage: gpu_sample.py OUT.jsonl
"""
import json, subprocess, sys, time

FIELDS = "index,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,memory.used,power.draw,clocks.sm,temperature.gpu"
with open(sys.argv[1], "w") as out:
    while True:
        t = time.time()
        r = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        rows = [dict(zip(FIELDS.split(","), [c.strip() for c in line.split(",")]))
                for line in r.stdout.strip().splitlines()]
        out.write(json.dumps({"t": t, "gpus": rows}) + "\n")
        out.flush()
        time.sleep(max(0.0, 1.0 - (time.time() - t)))
