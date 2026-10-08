#!/usr/bin/env python3
"""Create local configuration without replacing existing values or printing secrets."""

import os
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(root=ROOT):
    path = root / ".env"
    content = path.read_text() if path.exists() else (root / ".env.example").read_text()
    if not path.exists():
        gpu_defaults = {
            "WHISPER_MODEL": "large-v3",
            "WHISPER_COMPUTE_TYPE": "float16",
            "NLLB_MODEL": "facebook/nllb-200-3.3B",
            "MAX_VIDEO_DURATION_SECONDS": "120",
            "OLLAMA_HOST": "http://host.docker.internal:11434",
        }
        for key, value in gpu_defaults.items():
            content = re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.M)
    match = re.search(r"^INTERNAL_API_KEY=(.*)$", content, flags=re.M)
    if not match or not match.group(1).strip(" \"'"):
        value = "INTERNAL_API_KEY=" + secrets.token_hex(32)
        content = (
            re.sub(r"^INTERNAL_API_KEY=.*$", value, content, flags=re.M)
            if match
            else content.rstrip() + "\n" + value + "\n"
        )
    # Old localhost client URLs break remote browsers; use same-origin routes.
    for key, value in (
        ("NEXT_PUBLIC_API_BASE_URL", "/api"),
        ("NEXT_PUBLIC_INGEST_BASE_URL", "/ingest"),
    ):
        if re.search(rf"^{key}=", content, re.M):
            content = re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.M)
        else:
            content += f"{key}={value}\n"
    with path.open("w") as file:
        os.chmod(path, 0o600)
        file.write(content)
    print(
        "Local .env configured. Existing model and GPU settings preserved; secret not displayed."
    )


if __name__ == "__main__":
    configure()
