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

# Prefer `python` (backend container ships it as the default interpreter) but
# fall back to `python3` for host-side venv usage where `python` is absent.
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "error: neither 'python' nor 'python3' found on PATH. " \
       "Run this inside the backend container:" >&2
  echo "  docker compose exec -e LIPSYNC_BACKEND=\$LIPSYNC_BACKEND " \
       "backend bash /app/scripts/download_models.sh" >&2
  exit 1
fi

"$PYTHON" - <<PY
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

print("Downloading Coqui XTTS-v2 (~1.8 GB)…")
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = str(cache / "coqui-tts")
from TTS.api import TTS
TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

# Wav2Lip checkpoint (~400 MB). Skip if LIPSYNC_BACKEND=none.
backend = os.environ.get("LIPSYNC_BACKEND", "none")
if backend == "wav2lip":
    import urllib.request
    w2l_dir = cache / "wav2lip"
    w2l_dir.mkdir(parents=True, exist_ok=True)
    ckpt = w2l_dir / "wav2lip_gan.pth"
    if not ckpt.exists():
        url = os.environ.get(
            "WAV2LIP_CHECKPOINT_URL",
            "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth",
        )
        print(f"Downloading Wav2Lip checkpoint from {url} …")
        tmp = ckpt.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(ckpt)
    else:
        print("Wav2Lip checkpoint already present.")
else:
    print(f"Skipping Wav2Lip download (LIPSYNC_BACKEND={backend}). "
          "Set LIPSYNC_BACKEND=wav2lip to pre-fetch.")

print("Done.")
PY
