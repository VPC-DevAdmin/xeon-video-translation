#!/usr/bin/env bash
# End-to-end smoke test for stages 1–6 (M1 + M2 + M3 + M4).
# Uploads a fixture clip, polls the job until done, prints the transcript,
# translation, and a downloadable URL for the final watermarked MP4.
#
# Also supports reconnect mode (RECONNECT=<job-id> or --reconnect <id>):
# skip the upload and poll an existing job. Useful when the previous
# invocation hit the SMOKE_POLL_TIMEOUT_SECS client-side timeout and
# the job is still running on the backend.
#
# Requires: curl, jq, a running backend at $API_BASE, and a clip at the fixture path.
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
FIXTURE="${FIXTURE:-artifacts/inputs/IMG_7228.MOV}"
TARGET="${TARGET:-es}"
LIPSYNC="${LIPSYNC:-none}"   # none | wav2lip | musetalk | latentsync
RECONNECT="${RECONNECT:-}"

# Parse a --reconnect <id> flag in addition to env var.
while [ $# -gt 0 ]; do
  case "$1" in
    --reconnect)
      shift
      RECONNECT="${1:-}"
      shift || true
      ;;
    *)
      shift
      ;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq / apt-get install jq)" >&2
  exit 1
fi

if [ -z "$RECONNECT" ] && [ ! -f "$FIXTURE" ]; then
  cat >&2 <<EOF
Missing fixture: $FIXTURE
Drop a short single-speaker clip under artifacts/inputs/, e.g.:

  cp your_clip.mov artifacts/inputs/

Or override the path:
  FIXTURE=/path/to/clip.mp4 $0
  make run FIXTURE=/path/to/clip.mp4

Or, if a previous invocation timed out client-side while the job is
still running on the backend, reconnect instead of submitting a new
job:
  ./scripts/smoke_test.sh --reconnect <job-id>
  make reconnect JOB=<job-id>
EOF
  exit 1
fi

if [ -n "$RECONNECT" ]; then
  JOB_ID="$RECONNECT"
  echo "==> reconnecting to $API_BASE/jobs/$JOB_ID (no new submission)"
  # Validate the job exists before we start polling.
  if ! curl -sS -o /dev/null -w "%{http_code}" "$API_BASE/jobs/$JOB_ID" \
      | grep -q "^2"; then
    echo "    job $JOB_ID not found on $API_BASE" >&2
    exit 1
  fi
else
  echo "==> POST $API_BASE/jobs (target=$TARGET lipsync=$LIPSYNC)"

  # Optional per-request quality knobs. Only forward when set, so the
  # service's env defaults still apply otherwise. And filter the
  # MUSETALK_* fields out when LIPSYNC isn't musetalk — they're
  # silently dropped server-side but showing them in the log makes
  # non-MuseTalk runs read as if they were hybrid.
  _extra_fields=()
  if [ "$LIPSYNC" = "musetalk" ]; then
    [ -n "${MUSETALK_BLEND_MODE:-}" ]            && _extra_fields+=(-F "musetalk_blend_mode=$MUSETALK_BLEND_MODE")
    [ -n "${MUSETALK_BLEND_FEATHER:-}" ]         && _extra_fields+=(-F "musetalk_blend_feather=$MUSETALK_BLEND_FEATHER")
    [ -n "${MUSETALK_FACE_RESTORE:-}" ]          && _extra_fields+=(-F "musetalk_face_restore=$MUSETALK_FACE_RESTORE")
    [ -n "${MUSETALK_FACE_RESTORE_FIDELITY:-}" ] && _extra_fields+=(-F "musetalk_face_restore_fidelity=$MUSETALK_FACE_RESTORE_FIDELITY")
    [ -n "${MUSETALK_FACE_RESTORE_BLEND:-}" ]    && _extra_fields+=(-F "musetalk_face_restore_blend=$MUSETALK_FACE_RESTORE_BLEND")
  fi
  [ -n "${TTS_BACKEND:-}" ]                    && _extra_fields+=(-F "tts_backend=$TTS_BACKEND")
  [ -n "${ENABLE_STABILIZATION:-}" ]           && _extra_fields+=(-F "enable_stabilization=$ENABLE_STABILIZATION")
  [ -n "${ENABLE_OUTPUT_STABILIZATION:-}" ]    && _extra_fields+=(-F "enable_output_stabilization=$ENABLE_OUTPUT_STABILIZATION")
  if [ ${#_extra_fields[@]} -gt 0 ]; then
    echo "    options: ${_extra_fields[*]}"
  fi

  JOB_JSON="$(curl -sS -X POST "$API_BASE/jobs" \
    -F "video=@$FIXTURE" \
    -F "target_language=$TARGET" \
    -F "lipsync_backend=$LIPSYNC" \
    "${_extra_fields[@]}")"
  JOB_ID="$(echo "$JOB_JSON" | jq -r '.job_id')"
  echo "    job_id=$JOB_ID"
fi

# Poll until the job reaches a terminal state. The 4-minute budget
# the old loop used was fine for the default lipsync=none path (a few
# seconds total) but wildly short for musetalk (~10 min) and latentsync
# (~25-60 min). When the old loop timed out it fell through silently
# and the script exited 0, which broke any caller that treated the
# exit code as "job succeeded" — notably scripts/run_latentsync_adaptive.sh.
#
# Poll every SMOKE_POLL_INTERVAL secs (default 5) up to
# SMOKE_POLL_TIMEOUT_SECS (default 3 h). This is a CLIENT-side wait.
# If we hit it, the job is still running on the backend — use the
# reconnect mode (--reconnect <id>) or `make watch JOB=<id>` to keep
# watching without re-submitting.
#
# SMOKE_TIMEOUT_SECS is still honored for backward compat; prefer the
# clearer SMOKE_POLL_TIMEOUT_SECS name for new config.
SMOKE_POLL_TIMEOUT_SECS="${SMOKE_POLL_TIMEOUT_SECS:-${SMOKE_TIMEOUT_SECS:-10800}}"
SMOKE_POLL_INTERVAL="${SMOKE_POLL_INTERVAL:-5}"
echo "==> polling /jobs/$JOB_ID (client wait ${SMOKE_POLL_TIMEOUT_SECS}s, every ${SMOKE_POLL_INTERVAL}s)"
_deadline=$(( $(date +%s) + SMOKE_POLL_TIMEOUT_SECS ))
STATUS=""
while [ "$(date +%s)" -lt "$_deadline" ]; do
  STATUS_JSON="$(curl -sS "$API_BASE/jobs/$JOB_ID")"
  STATUS="$(echo "$STATUS_JSON" | jq -r '.status')"
  CURRENT="$(echo "$STATUS_JSON" | jq -r '.current_stage // "—"')"
  printf "\r    status=%s stage=%s         " "$STATUS" "$CURRENT"
  case "$STATUS" in
    completed) echo; break ;;
    failed)    echo; echo "JOB FAILED:"; echo "$STATUS_JSON" | jq .; exit 1 ;;
    cancelled) echo; echo "JOB CANCELLED."; exit 3 ;;
  esac
  sleep "$SMOKE_POLL_INTERVAL"
done
if [ "$STATUS" != "completed" ]; then
  echo
  cat <<EOF
WATCHER GAVE UP after ${SMOKE_POLL_TIMEOUT_SECS}s — last status=${STATUS}.

Your job is still running on the backend. This is a client-side wait
only; no cancellation was requested. Options:

  # Keep watching without re-submitting:
  ./scripts/smoke_test.sh --reconnect $JOB_ID
  # or
  make watch JOB=$JOB_ID

  # Cancel the job (frees the queue for anything waiting behind it):
  make cancel JOB=$JOB_ID

  # Or just bump the wait:
  SMOKE_POLL_TIMEOUT_SECS=$((SMOKE_POLL_TIMEOUT_SECS * 2)) \\
    ./scripts/smoke_test.sh --reconnect $JOB_ID
EOF
  exit 2
fi

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
