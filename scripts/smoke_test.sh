#!/usr/bin/env bash
# End-to-end smoke test for stages 1–6 (M1 + M2 + M3 + M4).
# Uploads a fixture clip, polls the job until done, prints the transcript,
# translation, and a downloadable URL for the final watermarked MP4.
#
# Requires: curl, jq, a running backend at $API_BASE, and a clip at the fixture path.
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
FIXTURE="${FIXTURE:-artifacts/inputs/IMG_7228.MOV}"
TARGET="${TARGET:-es}"
LIPSYNC="${LIPSYNC:-none}"   # none | wav2lip | musetalk | latentsync

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq / apt-get install jq)" >&2
  exit 1
fi

if [ ! -f "$FIXTURE" ]; then
  cat >&2 <<EOF
Missing fixture: $FIXTURE
Drop a short single-speaker clip under artifacts/inputs/, e.g.:

  cp your_clip.mov artifacts/inputs/

Or override the path:
  FIXTURE=/path/to/clip.mp4 $0
  make run FIXTURE=/path/to/clip.mp4
EOF
  exit 1
fi

echo "==> POST $API_BASE/jobs (target=$TARGET lipsync=$LIPSYNC)"
JOB_JSON="$(curl -sS -X POST "$API_BASE/jobs" \
  -F "video=@$FIXTURE" \
  -F "target_language=$TARGET" \
  -F "lipsync_backend=$LIPSYNC")"
JOB_ID="$(echo "$JOB_JSON" | jq -r '.job_id')"
echo "    job_id=$JOB_ID"

echo "==> polling /jobs/$JOB_ID"
for _ in $(seq 1 240); do
  STATUS_JSON="$(curl -sS "$API_BASE/jobs/$JOB_ID")"
  STATUS="$(echo "$STATUS_JSON" | jq -r '.status')"
  CURRENT="$(echo "$STATUS_JSON" | jq -r '.current_stage // "—"')"
  printf "\r    status=%s stage=%s         " "$STATUS" "$CURRENT"
  case "$STATUS" in
    completed) echo; break ;;
    failed)    echo; echo "JOB FAILED:"; echo "$STATUS_JSON" | jq .; exit 1 ;;
  esac
  sleep 1
done

echo
echo "==> Transcript:"
curl -sS "$API_BASE/jobs/$JOB_ID/artifacts/transcript.json" | jq -r '.text'
echo
echo "==> Translation ($TARGET):"
curl -sS "$API_BASE/jobs/$JOB_ID/artifacts/translation.json" | jq -r '.text'
echo
TTS_STATUS="$(echo "$STATUS_JSON" | jq -r '.stages[] | select(.name=="tts") | .status')"
if [ "$TTS_STATUS" = "done" ]; then
  echo "==> Cloned voice ($TARGET):"
  echo "    $API_BASE/jobs/$JOB_ID/artifacts/translated_audio.wav"
  echo
fi
MUX_STATUS="$(echo "$STATUS_JSON" | jq -r '.stages[] | select(.name=="mux") | .status')"
if [ "$MUX_STATUS" = "done" ]; then
  echo "==> Final video (watermarked):"
  echo "    $API_BASE/jobs/$JOB_ID/artifacts/final.mp4"
  echo
fi
echo "==> Per-stage timing:"
echo "$STATUS_JSON" | jq -r '.stages[] | "  \(.name): \(.status) (\(.duration_ms // "—") ms)"'
