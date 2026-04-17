#!/usr/bin/env bash
# End-to-end smoke test for stages 1–3 (M1 + M2).
# Uploads a fixture clip, polls the job until done, prints the transcript and translation.
#
# Requires: curl, jq, a running backend at $API_BASE, and a clip at the fixture path.
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
FIXTURE="${FIXTURE:-backend/tests/fixtures/sample_5s.mp4}"
TARGET="${TARGET:-es}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq / apt-get install jq)" >&2
  exit 1
fi

if [ ! -f "$FIXTURE" ]; then
  cat >&2 <<EOF
Missing fixture: $FIXTURE
Drop a short single-speaker clip there, e.g.:

  ffmpeg -i your_clip.mov -t 5 -ac 1 backend/tests/fixtures/sample_5s.mp4

Or override with FIXTURE=/path/to/clip.mp4 $0
EOF
  exit 1
fi

echo "==> POST $API_BASE/jobs (target=$TARGET)"
JOB_JSON="$(curl -sS -X POST "$API_BASE/jobs" \
  -F "video=@$FIXTURE" \
  -F "target_language=$TARGET")"
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
echo "==> Per-stage timing:"
echo "$STATUS_JSON" | jq -r '.stages[] | "  \(.name): \(.status) (\(.duration_ms // "—") ms)"'
