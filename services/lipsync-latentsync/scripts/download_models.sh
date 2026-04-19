#!/usr/bin/env bash
# Pre-fetch LatentSync weights into $MODEL_CACHE_DIR/latentsync/.
#
# PR-LS-1b: this script is live. PR-LS-1a shipped a placeholder that
# just printed a next-step note.
#
# Pulls the three files LatentSync upstream's README calls out:
#
#   $MODEL_CACHE_DIR/latentsync/
#   ├── latentsync_unet.pt      (~5 GB; SD 1.5 UNet fine-tuned for lipsync)
#   ├── stable_syncnet.pt       (~1.6 GB; SyncNet for supervision)
#   └── whisper/
#       └── tiny.pt             (~75 MB; OpenAI Whisper tiny)
#
# Source repo: https://huggingface.co/ByteDance/LatentSync-1.6
# License: Apache 2.0 on the code; weights carry their own terms — read
# them before any commercial use.
#
# The download uses `huggingface_hub.hf_hub_download()` rather than
# `snapshot_download` so we don't accidentally pull training logs,
# tensorboard runs, or other non-inference artifacts that live in the
# repo alongside the release checkpoints.
set -euo pipefail

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models}"
TARGET_DIR="$MODEL_CACHE_DIR/latentsync"

# Default to the 1.6 release. Override via LATENTSYNC_HF_REPO if you
# want to try a different version (1.5 is still up) or a local mirror.
LATENTSYNC_HF_REPO="${LATENTSYNC_HF_REPO:-ByteDance/LatentSync-1.6}"

mkdir -p "$TARGET_DIR"

echo "==> Downloading LatentSync weights"
echo "    repo:   $LATENTSYNC_HF_REPO"
echo "    target: $TARGET_DIR"

# The inline python keeps this script dependency-free on the host side —
# you run it via `docker compose exec lipsync-latentsync bash ...` and
# the service image already has huggingface_hub pinned.
python - <<PY
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    sys.stderr.write(
        f"huggingface_hub not importable in this image ({e}).\n"
        "This script is meant to run inside the lipsync-latentsync container, \n"
        "where PR-LS-1b's Dockerfile installs it. Usage:\n"
        "  docker compose exec lipsync-latentsync bash /app/scripts/download_models.sh\n"
    )
    sys.exit(1)

repo = os.environ["LATENTSYNC_HF_REPO"]
target = Path(os.environ["TARGET_DIR"]).resolve()
target.mkdir(parents=True, exist_ok=True)

# (repo-relative path, on-disk relative path under target)
# Upstream publishes whisper/tiny.pt — hf_hub_download preserves the
# subdir structure when local_dir is set, so just letting it land in
# target/ gets us the right layout.
FILES = [
    "latentsync_unet.pt",
    "stable_syncnet.pt",
    "whisper/tiny.pt",
]

for rel in FILES:
    dest = target / rel
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [skip] {rel} already present ({dest.stat().st_size:,} bytes)")
        continue
    print(f"  [get]  {rel}")
    hf_hub_download(
        repo_id=repo,
        filename=rel,
        local_dir=str(target),
        # local_dir_use_symlinks=False was the HF 0.x default; 1.x removed
        # the flag (always uses blob cache + hardlinks/copies). Both
        # behaviors put the file at `target/rel`, which is what we want.
    )
print("Done.")
PY

echo
echo "==> Verifying"
for f in latentsync_unet.pt stable_syncnet.pt whisper/tiny.pt; do
  if [ -f "$TARGET_DIR/$f" ]; then
    size="$(stat -c%s "$TARGET_DIR/$f" 2>/dev/null || stat -f%z "$TARGET_DIR/$f")"
    printf "  %-30s %15s bytes\n" "$f" "$size"
  else
    printf "  %-30s MISSING\n" "$f"
  fi
done

echo
echo "Next: curl http://localhost:\${LATENTSYNC_PORT:-8090}/weights"
echo "      should now show every entry with \"ok\": true."
