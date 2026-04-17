#!/usr/bin/env bash
# Download a job's artifacts locally. By default grabs the most recent job.
#
# Usage:
#   ./scripts/fetch.sh                        # latest job
#   ./scripts/fetch.sh <job-id-or-prefix>     # specific job (prefix ok if unique)
#   ./scripts/fetch.sh --list                 # show recent jobs without downloading
#
# Env:
#   API_BASE   backend URL (default: http://localhost:8088)
#   OUT_ROOT   where to write (default: ./artifacts)
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
OUT_ROOT="${OUT_ROOT:-./artifacts}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq / apt-get install jq)" >&2
  exit 1
fi

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
  curl -sS "$API_BASE/jobs?limit=20" | jq -r '
    "Recent jobs (newest first):",
    "",
    (.jobs[] |
      "  \(.job_id[:8])  \(.status)  \(.target_language // "?")  lipsync=\(.lipsync_backend // "?")  \(.input_filename // "?")  (\(.created_at // "?"))")
  '
  exit 0
fi

JOB_ID="${1:-}"

# Resolve job id: latest, full, or unique prefix.
if [ -z "$JOB_ID" ]; then
  JOB_ID="$(curl -sS "$API_BASE/jobs?limit=1" | jq -r '.jobs[0].job_id // empty')"
  if [ -z "$JOB_ID" ]; then
    echo "No jobs found on $API_BASE" >&2
    echo "(run a job first or point API_BASE at the right host)" >&2
    exit 1
  fi
  echo "==> latest job: $JOB_ID"
else
  # Allow partial IDs: find unique match against the recent-jobs list.
  if [ ${#JOB_ID} -lt 32 ]; then
    MATCHES="$(curl -sS "$API_BASE/jobs?limit=200" | jq -r --arg p "$JOB_ID" '
      .jobs[] | select(.job_id | startswith($p)) | .job_id')"
    N=$(echo "$MATCHES" | grep -c . || true)
    if [ "$N" -eq 0 ]; then
      echo "No job matches prefix '$JOB_ID'" >&2; exit 1
    elif [ "$N" -gt 1 ]; then
      echo "Prefix '$JOB_ID' is ambiguous; matches:" >&2
      echo "$MATCHES" >&2
      exit 1
    fi
    JOB_ID="$MATCHES"
    echo "==> resolved to: $JOB_ID"
  fi
fi

SHORT="${JOB_ID:0:8}"
OUT="$OUT_ROOT/$SHORT"
mkdir -p "$OUT"

echo "==> downloading artifacts to $OUT/"

LISTING="$(curl -sS --fail "$API_BASE/jobs/$JOB_ID/artifacts")" || {
  echo "Could not list artifacts for $JOB_ID (is the backend reachable at $API_BASE?)" >&2
  exit 1
}

COUNT=0
while IFS=$'\t' read -r name size; do
  [ -z "$name" ] && continue
  url="$API_BASE/jobs/$JOB_ID/artifacts/$name"
  if curl -sS --fail -o "$OUT/$name" "$url"; then
    printf "  %-28s %10s bytes\n" "$name" "$size"
    COUNT=$((COUNT+1))
  else
    echo "  ! failed to fetch $name" >&2
  fi
done < <(echo "$LISTING" | jq -r '.artifacts[] | [.name, .size_bytes] | @tsv')

echo "==> $COUNT file(s) saved under $OUT/"
