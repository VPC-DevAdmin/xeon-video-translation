#!/usr/bin/env bash
# Pre-download models so the first user-facing run isn't a multi-minute wait.
# Targets the CPU-only build: faster-whisper int8 + NLLB-200 distilled-600M.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

WHISPER_MODEL="${WHISPER_MODEL:-base}"
NLLB_MODEL="${NLLB_MODEL:-facebook/nllb-200-distilled-600M}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-./models}"

mkdir -p "$MODEL_CACHE_DIR"

echo "==> Models will be cached under: $MODEL_CACHE_DIR"
echo "==> Whisper: $WHISPER_MODEL (CPU int8)"
echo "==> NLLB:    $NLLB_MODEL"

python - <<PY
import os
from pathlib import Path

cache = Path(os.environ.get("MODEL_CACHE_DIR", "./models")).resolve()
whisper_dir = cache / "faster-whisper"
hf_dir = cache / "huggingface"
whisper_dir.mkdir(parents=True, exist_ok=True)
hf_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading faster-whisper '{os.environ.get('WHISPER_MODEL', 'base')}'…")
from faster_whisper import WhisperModel
WhisperModel(
    os.environ.get("WHISPER_MODEL", "base"),
    device="cpu",
    compute_type="int8",
    download_root=str(whisper_dir),
)

nllb = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
print(f"Downloading NLLB '{nllb}'…")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoTokenizer.from_pretrained(nllb, cache_dir=str(hf_dir))
AutoModelForSeq2SeqLM.from_pretrained(nllb, cache_dir=str(hf_dir))

print("Done.")
PY
