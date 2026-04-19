#!/usr/bin/env bash
# Placeholder — the real download script lands in PR-LS-1b.
#
# When implemented it will pull:
#   - LatentSync UNet checkpoint (~1 GB, ByteDance/LatentSync HF repo)
#   - SD 1.5 VAE ft-MSE (~330 MB, stabilityai/sd-vae-ft-mse)
#   - Whisper tiny (~75 MB, openai/whisper-tiny)
# into $MODEL_CACHE_DIR/latentsync/ inside this service's container.
#
# Same shape as services/musetalk/scripts/download_models.sh for consistency.
set -euo pipefail

cat >&2 <<'EOF'
lipsync-latentsync: model download is not wired up in PR-LS-1a.

This script is a placeholder. PR-LS-1b adds the actual HF downloads for:

  - ByteDance/LatentSync  -> latentsync_unet.pt      (~1 GB)
  - stabilityai/sd-vae-ft-mse -> sd-vae/             (~330 MB)
  - openai/whisper-tiny       -> whisper/            (~75 MB)

Until then `/lipsync` returns a 501 explaining the staging. Use
LIPSYNC=musetalk if you need working lipsync on this machine today.

See docs/lipsync.md for the full rollout plan.
EOF
exit 1
