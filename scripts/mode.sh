#!/usr/bin/env bash
# Switch the box between its exclusive modes. One mode is live at a time:
#   translate  backend + LLM + one LatentSync pool spanning LATENTSYNC_GPUS
#              (fast = 10 steps, quality = 40 steps, chosen per job)
#   avatar     dedicated avatar speech backend + MuseTalk renderer + ingest;
#              the LatentSync pool is stopped so its GPUs are free
#   assistant  video assistant: FlashHead render service (GPU 2) + backend +
#              LLM + ingest + frontend; LatentSync and MuseTalk stopped
#   status     show what is running and warm
# Models stay resident within a mode; a switch costs one container start
# (~35 s pool warm-up) and keeps the LLM, backend and frontend up throughout.
set -euo pipefail
cd "$(dirname "$0")/.."
C=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.llm.yml -f docker-compose.avatar.yml -f docker-compose.quality.yml)
A=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.llm.yml -f docker-compose.assistant.yml)
case "${1:-status}" in
  translate)
    "${C[@]}" stop backend-avatar musetalk-avatar >/dev/null 2>&1 || true
    # --no-deps: the avatar overlay makes ingest depend on the avatar
    # services, which must stay down in translation mode.
    "${C[@]}" up -d --no-deps backend llm-gpu lipsync-latentsync frontend ingest-webrtc
    echo "waiting for the LatentSync pool warm-up"
    for _ in $(seq 1 40); do
      s=$(curl -s -m 5 "localhost:${LATENTSYNC_PORT:-8090}/health" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("warmup",{}).get("status"))' 2>/dev/null || true)
      [ "$s" = "done" ] && break; sleep 5
    done
    echo "translation mode: pool warm-up=$s" ;;
  avatar)
    "${C[@]}" stop lipsync-latentsync >/dev/null 2>&1 || true
    "${C[@]}" up -d backend llm-gpu backend-avatar musetalk-avatar frontend ingest-webrtc
    echo "avatar mode: pool stopped; avatar services starting (admission opens after their warm-up)" ;;
  assistant)
    "${C[@]}" stop lipsync-latentsync backend-avatar musetalk-avatar >/dev/null 2>&1 || true
    R="${ASSISTANT_RENDERER_URL:-http://localhost:${FLASHHEAD_PORT:-8094}}"
    if curl -s -m 3 "$R/health" >/dev/null 2>&1; then
      echo "renderer already answering at $R (lab container); not starting the compose flashhead service"
      "${A[@]}" up -d --no-deps backend llm-gpu frontend ingest-webrtc
    else
      "${A[@]}" up -d --no-deps backend llm-gpu frontend flashhead ingest-webrtc
    fi
    echo "waiting for the FlashHead warm-up"
    for _ in $(seq 1 60); do
      s=$(curl -s -m 5 "$R/health" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("done" if d.get("warm") else d.get("status"))' 2>/dev/null || true)
      [ "$s" = "done" ] && break; sleep 5
    done
    echo "assistant mode: renderer=$s; open the frontend at /assistant" ;;
  status)
    docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'polyglot|avatar' || true
    for p in "${LATENTSYNC_PORT:-8090}" 8092 8093 "${FLASHHEAD_PORT:-8094}"; do printf '%s: ' "$p"; curl -s -m 3 "localhost:$p/health" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("warmup") or {k:d[k] for k in ("model_warm","avatar_inference_warm","warm","chunks_rendered") if k in d})' 2>/dev/null || echo down; done ;;
  *) echo "usage: $0 translate|avatar|assistant|status" >&2; exit 2 ;;
esac
