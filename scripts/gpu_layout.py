#!/usr/bin/env python3
"""Generate a conflict-free eight-card environment; never start/stop services.

By default IDs are indices. --inventory-json replaces them with persistent
GPU UUIDs from a [{"index": 0, "uuid": "GPU-..."}, ...] inventory.
The external LLM is owned by its operator; this script only reserves its IDs.
"""

import argparse
import json
from pathlib import Path


def layout(profile, quality_audio=False):
    if profile == "dedicated":
        roles = {
            "BACKEND_GPU": [0],
            "LLM_GPUS": [1],
            "MUSETALK_GPU": [2],
            "AVATAR_BACKEND_GPU": [3],
            "AVATAR_MUSETALK_GPU": [4],
            "LATENTSYNC_GPUS": [5, 6, 7],
        }
    elif profile == "shared":
        roles = {
            "BACKEND_GPU": [0],
            "LLM_GPUS": [1, 2],
            "MUSETALK_GPU": [3],
            "AVATAR_BACKEND_GPU": [5],
            "AVATAR_MUSETALK_GPU": [6],
            "LATENTSYNC_GPUS": [4, 7],
        }
    else:
        raise ValueError("profile must be dedicated or shared")
    if quality_audio:
        roles["LATENTSYNC_GPUS"].remove(7)
        roles["AUDIO_QUALITY_GPU"] = [7]
    validate(roles)
    return roles


def validate(roles):
    owners = {}
    for role, ids in roles.items():
        if not ids:
            raise ValueError(f"{role} has no GPU")
        for identifier in ids:
            if identifier in owners:
                raise ValueError(
                    f"GPU {identifier} is assigned to both {owners[identifier]} and {role}"
                )
            owners[identifier] = role
    return roles


def env_text(roles, inventory=None):
    validate(roles)
    if inventory is not None:
        by_index = {int(item["index"]): item["uuid"] for item in inventory}
        roles = {role: [by_index[i] for i in ids] for role, ids in roles.items()}
        validate(roles)
    return (
        "# Generated GPU assignments; review against running host workloads.\n"
        + "".join(f"{role}={','.join(map(str, ids))}\n" for role, ids in roles.items())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["dedicated", "shared"], required=True)
    parser.add_argument("--quality-audio", action="store_true")
    parser.add_argument("--inventory-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = (
        json.loads(args.inventory_json.read_text()) if args.inventory_json else None
    )
    text = env_text(layout(args.profile, args.quality_audio), inventory)
    # Refuse to overwrite operator-owned configuration.
    with args.output.open("x") as handle:
        handle.write(text)
    print(args.output)


if __name__ == "__main__":
    main()
