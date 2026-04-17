#!/usr/bin/env bash
# Pre-fetch the weights the MuseTalk lipsync service needs.
#
# Run inside the service container:
#   docker compose exec lipsync-musetalk bash /app/scripts/download_models.sh
#
# Total on disk ~1.4 GB, broken down:
#   MuseTalk V1.5 UNet + config    ~900 MB  (TMElyralab/MuseTalk, CC-BY-NC 4.0)
#   SD 1.5 VAE (ft-MSE)            ~330 MB  (stabilityai/sd-vae-ft-mse, CreativeML Open RAIL-M)
#   Whisper tiny encoder           ~75 MB   (openai/whisper-tiny, MIT)
#   BiSeNet face parser            ~52 MB   (via Google Drive, MIT — see docs)
#   ResNet18 ImageNet backbone     ~46 MB   (pytorch.org, BSD)
set -euo pipefail

CACHE="${MODEL_CACHE_DIR:-/models}/musetalk"
mkdir -p "$CACHE"

echo "==> Weights will be cached under: $CACHE"

# Prefer `python` (container default) but fall back to python3 for host venvs.
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

"$PY" - <<'PY'
import os
import sys
import urllib.request
from pathlib import Path

cache = Path(os.environ.get("MODEL_CACHE_DIR", "/models")).resolve() / "musetalk"
cache.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None

# --- MuseTalk V1.5 ---------------------------------------------------------
print("==> MuseTalk V1.5 UNet + config (~900 MB) ...")
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    allow_patterns=["musetalkV15/unet.pth", "musetalkV15/musetalk.json"],
    local_dir=str(cache),
    token=token,
)

# --- SD 1.5 VAE ------------------------------------------------------------
print("==> SD 1.5 VAE (~330 MB) ...")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
    local_dir=str(cache / "sd-vae"),
    token=token,
)

# --- Whisper tiny (encoder only in use) ------------------------------------
print("==> Whisper tiny (~75 MB) ...")
snapshot_download(
    repo_id="openai/whisper-tiny",
    allow_patterns=[
        "config.json",
        "pytorch_model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ],
    local_dir=str(cache / "whisper"),
    token=token,
)

# --- BiSeNet face parser ---------------------------------------------------
# Upstream weights live on Google Drive. gdown is the least-bad path; if it
# breaks, docs/lipsync.md lists a manual fallback.
bisenet_dir = cache / "face-parse-bisent"
bisenet_dir.mkdir(parents=True, exist_ok=True)
bisenet_weights = bisenet_dir / "79999_iter.pth"
if bisenet_weights.exists() and bisenet_weights.stat().st_size > 10_000_000:
    print("==> BiSeNet face parser already present.")
else:
    print("==> BiSeNet face parser (~52 MB, via gdown) ...")
    import gdown

    file_id = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
    try:
        gdown.download(id=file_id, output=str(bisenet_weights), quiet=False)
    except Exception as e:
        print(f"    gdown failed: {e}", file=sys.stderr)
        print("    Manual fallback: download 79999_iter.pth from", file=sys.stderr)
        print(f"      https://drive.google.com/file/d/{file_id}/view", file=sys.stderr)
        print(f"    and drop it at {bisenet_weights}", file=sys.stderr)
        sys.exit(1)

# ResNet18 ImageNet backbone the BiSeNet model initializes from.
resnet_path = bisenet_dir / "resnet18-5c106cde.pth"
if resnet_path.exists():
    print("==> ResNet18 backbone already present.")
else:
    print("==> ResNet18 ImageNet backbone (~46 MB) ...")
    urllib.request.urlretrieve(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        str(resnet_path),
    )

print("==> Done. Weights layout:")
for p in sorted(cache.rglob("*")):
    if p.is_file():
        print(f"    {p.relative_to(cache)}  ({p.stat().st_size // (1024*1024)} MB)")
PY
