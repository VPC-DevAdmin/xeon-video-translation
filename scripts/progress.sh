#!/usr/bin/env bash
# Show job status. Default is one-shot; --watch polls until terminal state.
#
# Usage:
#   ./scripts/progress.sh                      # one-shot on latest job
#   ./scripts/progress.sh <job-id-or-prefix>   # one-shot on specific job
#   ./scripts/progress.sh --watch              # poll latest until done
#   ./scripts/progress.sh --watch <prefix>     # poll specific
#
# Env:
#   API_BASE   backend URL (default: http://localhost:8088)
#   INTERVAL   polling interval in seconds (default: 5)
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8088}"
INTERVAL="${INTERVAL:-5}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq / apt-get install jq)" >&2
  exit 1
fi

WATCH=0
JOB_ID=""
for arg in "$@"; do
  case "$arg" in
    --watch|-w) WATCH=1 ;;
    --interval=*) INTERVAL="${arg#--interval=}" ;;
    -h|--help)
      sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) JOB_ID="$arg" ;;
  esac
done

# ---- resolve job id (latest, full, or unique prefix) ------------------------

resolve_job_id() {
  local raw="$1"
  if [ -z "$raw" ]; then
    curl -sS "$API_BASE/jobs?limit=1" | jq -r '.jobs[0].job_id // empty'
    return
  fi
  if [ ${#raw} -ge 32 ]; then
    echo "$raw"
    return
  fi
  # prefix match
  local matches
  matches="$(curl -sS "$API_BASE/jobs?limit=200" | jq -r --arg p "$raw" \
    '.jobs[] | select(.job_id | startswith($p)) | .job_id')"
  local n
  n=$(echo "$matches" | grep -c . || true)
  if [ "$n" -eq 0 ]; then
    echo "no job matches prefix '$raw'" >&2
    return 1
  elif [ "$n" -gt 1 ]; then
    echo "prefix '$raw' is ambiguous; matches:" >&2
    echo "$matches" >&2
    return 1
  fi
  echo "$matches"
}

JOB_ID="$(resolve_job_id "$JOB_ID")" || exit 1
if [ -z "$JOB_ID" ]; then
  echo "no jobs found on $API_BASE" >&2
  exit 1
fi

# ---- reporting --------------------------------------------------------------

# Fetch once, cache decoded JSON. Returns via stdout.
fetch_job() {
  curl -sS --fail "$API_BASE/jobs/$JOB_ID" 2>/dev/null
}

# Observed-rate ETA. The static `eta_seconds` from the backend is
# `source_duration * realtime_factor` — that factor is a fixed guess
# in orchestrator.py that can be off by 2× or more for LatentSync on
# varying CPUs, resolutions, and feature-flag combinations. Instead
# we time how long the current running stage took to reach its
# current progress and linearly extrapolate remaining work:
#   remaining ≈ elapsed * (1 - p) / p
# Only kicks in once progress > 2% AND elapsed > 3s, so we don't
# divide by near-zero and produce a nonsense ETA in the first couple
# of ticks. Falls back to the static estimate until then.
_RUN_STAGE_NAME=""
_RUN_STAGE_STARTED_AT=""

# Compute observed-remaining in seconds (or "null" for "no observation yet").
# Emits ALWAYS as valid JSON so jq --argjson can consume it.
compute_observed_remaining() {
  local json="$1"
  local now running_stage progress elapsed
  now=$(date +%s)
  running_stage=$(echo "$json" | jq -r '(.stages[] | select(.status=="running") | .name) // empty' | head -1)
  progress=$(echo "$json" | jq -r '(.stages[] | select(.status=="running") | .progress) // 0' | head -1)

  # Stage changed (or no stage running) — reset the anchor.
  if [ "$running_stage" != "$_RUN_STAGE_NAME" ]; then
    _RUN_STAGE_NAME="$running_stage"
    _RUN_STAGE_STARTED_AT="$now"
  fi

  if [ -z "$running_stage" ] || [ -z "$_RUN_STAGE_STARTED_AT" ]; then
    echo "null"
    return
  fi

  elapsed=$((now - _RUN_STAGE_STARTED_AT))
  # Only trust the observation once we have enough signal.
  if [ "$elapsed" -lt 3 ]; then
    echo "null"
    return
  fi
  awk -v p="$progress" -v e="$elapsed" '
    BEGIN {
      if (p + 0 > 0.02) {
        printf "%.0f", (1 - p) * e / p;
      } else {
        print "null";
      }
    }'
}

render_report() {
  local json="$1"
  local observed="${2:-null}"
  local running_name="${3:-}"
  echo "$json" | jq -r --argjson obs "$observed" --arg running "$running_name" '
    def pad(s; n): (s | tostring) as $x | $x + (" " * (n - ($x | length)));
    def icon(st):
      if   st == "done"     then "✓"
      elif st == "running"  then "⟳"
      elif st == "failed"   then "✗"
      elif st == "skipped"  then "-"
      elif st == "pending"  then "·"
      else "?" end;
    def color(st):
      if   st == "done"     then "\u001b[32m"    # green
      elif st == "running"  then "\u001b[33m"    # yellow
      elif st == "failed"   then "\u001b[31m"    # red
      elif st == "skipped"  then "\u001b[90m"    # grey
      else "\u001b[90m" end;
    def reset: "\u001b[0m";
    def bar(p; width):
      if p == null then ""
      else
        ((p * width) | floor) as $f |
        "[" + ("█" * $f) + ("░" * (width - $f)) + "] " +
        ((p * 100) | floor | tostring) + "%"
      end;
    # Running-stage ETA. Three-tier fallback:
    #
    #   1. Observed rate (preferred): $obs is computed in shell from
    #      elapsed-since-stage-started vs. live progress; linear
    #      extrapolation of remaining work. Only available after ~3s
    #      and >2% progress (see compute_observed_remaining).
    #   2. Static estimate * (1 - p): the backend's guess-factor
    #      `eta_seconds` is the stage's total — show the remaining
    #      portion of it until observed data kicks in.
    #   3. Raw total `eta_seconds`: when progress is 0 or null.
    #
    # $obs is a single number for the current running stage only (the
    # caller knows which one via $running), so we gate on s.name
    # matching before using it.
    def stage_right(s):
      if   s.status == "done" and s.duration_ms then
        ((s.duration_ms / 1000) | floor | tostring) + "s"
      elif s.status == "running" then
        (bar(s.progress; 16)) +
        (if $obs != null and s.name == $running then
           "  ~" + ($obs | tostring) + "s left"
         elif s.eta_seconds and s.eta_seconds > 0 then
           (if (s.progress // 0) > 0 then
              ((s.eta_seconds * (1 - (s.progress // 0))) | floor) as $rem |
              (if $rem < 0 then 0 else $rem end) as $rem |
              "  ~" + ($rem | tostring) + "s left*"
            else
              "  ~" + (s.eta_seconds | floor | tostring) + "s ETA*"
            end)
         else "" end)
      elif s.status == "pending" and s.eta_seconds then
        "~" + (s.eta_seconds | floor | tostring) + "s ETA"
      else "" end;

    "",
    "Job:      \(.job_id[:8])   status: " + color(.status) + .status + reset,
    "Stage:    \(.current_stage // "—")",
    "Lang:     \(.source_language // "?") → \(.target_language)    Lipsync: \(.lipsync_backend // "default")",
    "Input:    \(.input_filename // "?")   source duration: \((.source_duration_seconds // 0) | tostring)s",
    "Created:  \(.created_at // "?")",
    (if .completed_at then "Finished: \(.completed_at)" else empty end),
    "",
    (.stages[] |
      color(.status) + "  " + icon(.status) + " " + pad(.name; 12) + reset +
      pad(.status; 10) +
      stage_right(.)),
    "",
    (if .error then "ERROR: " + color("failed") + .error + reset else empty end)
  '
}

# ---- main loop --------------------------------------------------------------

if [ "$WATCH" = "0" ]; then
  json="$(fetch_job)" || { echo "job not found: $JOB_ID" >&2; exit 1; }
  # One-shot: no history, so no observed rate available.
  render_report "$json" "null" ""
  status=$(echo "$json" | jq -r '.status')
  [ "$status" = "failed" ] && exit 1 || exit 0
fi

# watch mode
trap 'tput cnorm 2>/dev/null; echo; exit 130' INT
tput civis 2>/dev/null || true

while :; do
  json="$(fetch_job)" || { echo "job not found: $JOB_ID" >&2; exit 1; }
  observed="$(compute_observed_remaining "$json")"
  clear
  render_report "$json" "$observed" "$_RUN_STAGE_NAME"
  status=$(echo "$json" | jq -r '.status')
  if [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
    break
  fi
  sleep "$INTERVAL"
done
tput cnorm 2>/dev/null || true

[ "$status" = "failed" ] && exit 1 || exit 0
