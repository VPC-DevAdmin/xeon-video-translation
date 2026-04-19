#!/usr/bin/env bash
# Pre-fetch LatentSync weights into $MODEL_CACHE_DIR/latentsync/.
#
# Pulls the three files LatentSync upstream's README calls out:
#
#   $MODEL_CACHE_DIR/latentsync/
#   +-- latentsync_unet.pt      (~5 GB; SD 1.5 UNet fine-tuned for lipsync)
#   +-- stable_syncnet.pt       (~1.6 GB; SyncNet for supervision)
#   +-- whisper/
#       +-- tiny.pt             (~75 MB; OpenAI Whisper tiny)
#
# Source repo: https://huggingface.co/ByteDance/LatentSync-1.6
# License: Apache 2.0 on the code; weights carry their own terms -- read
# them before any commercial use.
#
# The download uses huggingface_hub.hf_hub_download() rather than
# snapshot_download so we don't accidentally pull training logs,
# tensorboard runs, or other non-inference artifacts that live in the
# repo alongside the release checkpoints.
#
# Env overrides:
#   LATENTSYNC_HF_REPO   default: ByteDance/LatentSync-1.6
#   MODEL_CACHE_DIR      default: /models  (set by Compose in-container)
set -euo pipefail

# The python heredoc below is intentionally *quoted* (<<'PY'). An
# unquoted heredoc lets bash interpret $vars and `backticks` inside the
# python source, which previously turned a `target/rel` in a comment
# into a bash command-substitution attempt. Quote-heredoc keeps python
# source as literal text and we pass env through os.environ instead.
python - <<'PY'
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

repo = os.environ.get("LATENTSYNC_HF_REPO", "ByteDance/LatentSync-1.6")
cache = Path(os.environ.get("MODEL_CACHE_DIR", "/models")).resolve()
target = cache / "latentsync"
target.mkdir(parents=True, exist_ok=True)

print(f"==> Downloading LatentSync weights")
print(f"    repo:   {repo}")
print(f"    target: {target}")

# Upstream publishes whisper/tiny.pt -- hf_hub_download preserves the
# subdir structure under local_dir, so the on-disk layout matches what
# LatentSync's inference code expects without any extra moves.
FILES = [
    "latentsync_unet.pt",
    "stable_syncnet.pt",
    "whisper/tiny.pt",
]

# Defensive lower bounds: the HF repo occasionally ships a placeholder
# file or partial upload; skipping on size > 1 MB avoids re-downloading
# a real 5 GB file but catches the "0-byte stub" case.
_MIN_RESUMABLE_BYTES = 1_000_000

for rel in FILES:
    dest = target / rel
    if dest.exists() and dest.stat().st_size > _MIN_RESUMABLE_BYTES:
        print(f"  [skip] {rel} already present ({dest.stat().st_size:,} bytes)")
        continue
    print(f"  [get]  {rel}")
    hf_hub_download(
        repo_id=repo,
        filename=rel,
        local_dir=str(target),
    )
print("Done.")
PY

# Post-download report. Sizes use GNU `stat` format (Debian slim image);
# the BSD fallback (-f%z) is kept for dev hosts running this on macOS.
TARGET_DIR="${MODEL_CACHE_DIR:-/models}/latentsync"
echo
echo "==> Verifying"
for f in latentsync_unet.pt stable_syncnet.pt whisper/tiny.pt; do
  path="$TARGET_DIR/$f"
  if [ -f "$path" ]; then
    size="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")"
    printf "  %-30s %15s bytes\n" "$f" "$size"
  else
    printf "  %-30s MISSING\n" "$f"
  fi
done

echo
echo "Next: curl http://localhost:\${LATENTSYNC_PORT:-8090}/weights"
echo "      should now show every entry with \"ok\": true."
