#!/usr/bin/env python3
"""Hash model artifacts for reproducible GPU runs; never include credential files."""

import argparse
import hashlib
import json
from pathlib import Path

SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".json",
    ".yaml",
    ".yml",
    ".model",
    ".txt",
}


def manifest(root):
    return {
        str(path.relative_to(root)): hashlib.file_digest(
            path.open("rb"), "sha256"
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in SUFFIXES
        and not any(
            part.startswith(".")
            or part.lower()
            in {
                "token",
                "token.json",
                "hf_token",
                "hf_token.json",
                "credentials",
                "credentials.json",
            }
            or "secret" in part.lower()
            for part in path.relative_to(root).parts
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    actual = manifest(args.root)
    if not actual:
        raise SystemExit("No model artifacts found")
    if args.verify:
        expected = json.loads(args.output.read_text())
        changed = sorted(
            set(actual) ^ set(expected)
            | {k for k in actual.keys() & expected.keys() if actual[k] != expected[k]}
        )
        if changed:
            raise SystemExit("Model manifest mismatch: " + ", ".join(changed))
    else:
        args.output.write_text(json.dumps(actual, sort_keys=True, indent=2) + "\n")
    print(
        "MODEL_MANIFEST_SHA256=" + hashlib.sha256(args.output.read_bytes()).hexdigest()
    )
