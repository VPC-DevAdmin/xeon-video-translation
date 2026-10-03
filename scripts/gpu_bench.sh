#!/usr/bin/env bash
# Submit one job, wait for it, print per-stage wall-clock. GPU-track bench
# harness; works against any running backend.
#
#   scripts/gpu_bench.sh [--fixture F] [--target es] [--lipsync none|wav2lip|musetalk|latentsync]
#                        [--tts auto|xtts|f5tts|indicf5] [--label name] [--extra "k=v k=v"]
#
# Output: one JSON line per job appended to $BENCH_LOG (default
# artifacts/bench/results.jsonl) plus a human table on stdout.
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
FIXTURE="${FIXTURE:-artifacts/inputs/IMG_7228.MOV}"
TARGET="${TARGET:-es}"
LIPSYNC="${LIPSYNC:-none}"
TTS="${TTS:-auto}"
LABEL=""
EXTRA=""
POLL="${POLL:-3}"
TIMEOUT_SECS="${TIMEOUT_SECS:-7200}"
BENCH_LOG="${BENCH_LOG:-artifacts/bench/results.jsonl}"

while [ $# -gt 0 ]; do
  case "$1" in
    --fixture) FIXTURE="$2"; shift 2 ;;
    --target)  TARGET="$2"; shift 2 ;;
    --lipsync) LIPSYNC="$2"; shift 2 ;;
    --tts)     TTS="$2"; shift 2 ;;
    --label)   LABEL="$2"; shift 2 ;;
    --extra)   EXTRA="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
[ -f "$FIXTURE" ] || { echo "fixture not found: $FIXTURE" >&2; exit 1; }
mkdir -p "$(dirname "$BENCH_LOG")"

form=(-F "video=@${FIXTURE}" -F "target_language=${TARGET}" -F "lipsync_backend=${LIPSYNC}" -F "tts_backend=${TTS}")
for kv in $EXTRA; do form+=(-F "$kv"); done

t0=$(date +%s.%N)
created=$(curl -sS -X POST "$API_BASE/jobs" "${form[@]}")
job_id=$(echo "$created" | jq -r '.job_id // empty')
[ -n "$job_id" ] || { echo "submit failed: $created" >&2; exit 1; }
echo "job $job_id  fixture=$(basename "$FIXTURE") target=$TARGET lipsync=$LIPSYNC tts=$TTS ${LABEL:+label=$LABEL}"

deadline=$(( $(date +%s) + TIMEOUT_SECS ))
while :; do
  rec=$(curl -sS "$API_BASE/jobs/$job_id")
  status=$(echo "$rec" | jq -r '.status')
  stage=$(echo "$rec" | jq -r '.current_stage // "-"')
  case "$status" in
    completed|failed) break ;;
  esac
  [ "$(date +%s)" -lt "$deadline" ] || { echo "timed out waiting for $job_id (last stage $stage)" >&2; break; }
  printf '\r  %-10s %-12s' "$status" "$stage" >&2
  sleep "$POLL"
done
t1=$(date +%s.%N)
printf '\r%40s\r' "" >&2

wall=$(echo "$t1 - $t0" | bc)
src=$(echo "$rec" | jq -r '.source_duration_seconds // 0')
echo "status=$status  wall=${wall}s  source=${src}s"
echo "$rec" | jq -r '
  "  stage        status    secs",
  (.stages[] | "  \(.name | . + "            " | .[:12]) \(.status | . + "        " | .[:8]) \(if .duration_ms then (.duration_ms/1000|tostring) else "-" end)")'
if [ "$status" = "failed" ]; then
  echo "$rec" | jq -r '.stages[] | select(.error) | "  ERROR \(.name): \(.error)"'
fi

echo "$rec" | jq -c --arg label "$LABEL" --arg fixture "$(basename "$FIXTURE")" --arg wall "$wall" \
  --arg lipsync "$LIPSYNC" --arg tts "$TTS" --arg target "$TARGET" --arg ts "$(date -u +%FT%TZ)" '
  {ts:$ts, label:$label, job_id:.job_id, status:.status, fixture:$fixture, target:$target,
   lipsync:$lipsync, tts:$tts, source_s:.source_duration_seconds, wall_s:($wall|tonumber),
   stages:(.stages | map({(.name): (if .duration_ms then .duration_ms/1000 else null end)}) | add)}' \
  >> "$BENCH_LOG"
echo "logged to $BENCH_LOG"
