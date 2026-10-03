#!/usr/bin/env python3
"""Download a checksum-verified Silero CPU VAD model into the model cache."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx"
SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def fetch(destination):
    destination = Path(destination)
    if (
        destination.is_file()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256
    ):
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(10 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("VAD checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination", nargs="?", default="models/silero/silero_vad.onnx"
    )
    args = parser.parse_args()
    fetch(args.destination)
    print(args.destination)
