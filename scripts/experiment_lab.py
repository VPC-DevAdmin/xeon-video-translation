#!/usr/bin/env python3
"""Plan GPU trials or run explicit local recipes, preserving unsuccessful trials.

Recipes are trusted argv arrays. This harness manages local process groups;
recipes that start containers or remote jobs must manage their own cancellation.
A completed command is never a quality pass.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def validate_catalog(catalog):
    seen = set()
    for item in catalog["experiments"]:
        name = item["id"]
        if name in seen or not name.replace("-", "").isalnum():
            raise ValueError(f"duplicate/invalid experiment ID: {name}")
        seen.add(name)
        if not item.get("hypothesis") or not item.get("reject_if"):
            raise ValueError(f"missing hypothesis or rejection criteria: {name}")
        if not isinstance(item.get("gpus_min"), int) or item["gpus_min"] < 0:
            raise ValueError(f"invalid GPU requirement: {name}")
    return catalog


def gpu_inventory():
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi unavailable", "devices": []}
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        devices = []
        for line in result.stdout.splitlines():
            index, identifier, name, driver, total, used = [x.strip() for x in line.split(",")]
            devices.append({"index": index, "uuid": identifier, "name": name, "driver": driver,
                            "total_mb": int(total), "used_mb": int(used)})
        return {"available": bool(devices), "devices": devices}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "devices": []}


def normalize_gpus(ids, inventory):
    aliases = {d[key]: d["uuid"] for d in inventory["devices"] for key in ("index", "uuid")}
    if any(value not in aliases for value in ids):
        raise ValueError("GPU IDs must identify visible devices")
    normalized = [aliases[value] for value in ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("GPU IDs must identify distinct physical devices")
    return normalized


def stop_group(process):
    """Terminate descendants too, including when the group leader has exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_one(item, recipe, root, gpu_ids, inventory, timeout):
    folder = root / f"{item['id']}-{uuid.uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    record = {"experiment_id": item["id"], "at": datetime.now(timezone.utc).isoformat(),
              "quality_status": "unreviewed", "hardware": inventory,
              "selected_gpus": gpu_ids,
              "model_revision": recipe.get("model_revision") if isinstance(recipe, dict) else None}
    try:
        if recipe is None:
            record.update(status="blocked", reason="reviewed adapter recipe not supplied")
        elif item.get("requires_gpu", True) and not inventory["available"]:
            record.update(status="blocked", reason=inventory.get("reason", "GPU unavailable"))
        elif item.get("requires_gpu", True) and len(gpu_ids) < item["gpus_min"]:
            record.update(status="blocked", reason=f"requires at least {item['gpus_min']} explicitly assigned GPUs")
        else:
            process = None
            started = time.monotonic()
            try:
                command = recipe.get("argv") if isinstance(recipe, dict) else None
                if not isinstance(command, list) or not command or not all(isinstance(v, str) and v for v in command):
                    raise ValueError("recipe argv must be a nonempty list of nonempty strings")
                if not isinstance(recipe.get("cwd"), str):
                    raise TypeError("recipe cwd must be an existing directory")
                cwd = Path(recipe["cwd"]).resolve()
                if not cwd.is_dir():
                    raise ValueError("recipe cwd must be an existing directory")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                env["EXPERIMENT_OUTPUT_DIR"] = str(folder.resolve())
                record["recipe_sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
                with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
                    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
                    code = process.wait(timeout=timeout)
                    record.update(status="completed" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                record.update(status="timed_out", reason="experiment deadline exceeded")
            except (OSError, ValueError, TypeError) as exc:
                record.update(status="failed", reason=str(exc))
            except KeyboardInterrupt:
                record.update(status="cancelled", reason="operator interrupted experiment")
                raise
            finally:
                if process is not None:
                    stop_group(process)
                record["wall_seconds"] = time.monotonic() - started
    finally:
        (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("experiments/gpu-candidates.json"))
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--gpus", default="", help="Comma-separated physical indices or GPU UUIDs")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    catalog = validate_catalog(json.loads(args.catalog.read_text()))
    recipes = json.loads(args.recipes.read_text()) if args.recipes else {}
    if not isinstance(recipes, dict):
        parser.error("recipes must map experiment IDs to recipe objects")
    inventory = gpu_inventory()
    selected = set(args.ids or [e["id"] for e in catalog["experiments"]])
    unknown = selected - {e["id"] for e in catalog["experiments"]}
    if unknown:
        parser.error(f"unknown experiments: {sorted(unknown)}")
    gpu_ids = [s.strip() for s in args.gpus.split(",") if s.strip()]
    if inventory["available"]:
        try:
            gpu_ids = normalize_gpus(gpu_ids, inventory)
        except ValueError as exc:
            parser.error(str(exc))
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    results = []
    try:
        for item in catalog["experiments"]:
            if item["id"] not in selected:
                continue
            if args.execute:
                results.append(run_one(item, recipes.get(item["id"]), args.output, gpu_ids, inventory, args.timeout))
            else:
                results.append({"experiment_id": item["id"], "status": "planned", "integration": item["integration"],
                                "recipe_supplied": item["id"] in recipes, "gpu_available": inventory["available"]})
    except KeyboardInterrupt:
        return 130
    print(json.dumps(results, indent=2))
    return int(any(r["status"] in {"failed", "timed_out", "blocked"} for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
