#!/usr/bin/env bash
# Adaptive memory ladder for LatentSync runs.
#
# Why this exists
# ---------------
# LatentSync's peak memory is dominated by two things: the resident
# models (UNet ~5 GB + VAE ~5 GB + Whisper) and the final restore_video
# step, which allocates full-resolution kornia warp/blur tensors per
# frame. Peak RSS scales with clip length and resolution — what fits in
# 48 GB for a 30 s 1080p clip will OOM for a 90 s or 4K clip.
#
# A static memory ceiling is always wrong for some workload. This script
# picks a working one per-run:
#
#   1. Kick off the job at the current cgroup limit (typically 48 g from
#      docker-compose.yml).
#   2. If the container restarts mid-run — a reliable OOM signal for
#      this service, since the Python code doesn't crash for other
#      reasons — call `docker update --memory=<next>` live and retry.
#   3. The retry hits the resume cache (content-addressable, populated
#      by the first run before restore_video started), so it only
#      redoes the cheap half (~3 min) instead of the whole pipeline.
#   4. Cap escalation at 128 g. If 128 g isn't enough, something is
#      wrong with the workload, not the limit.
#
# Env overrides
#   CONTAINER    Docker container name (default: polyglot-lipsync-latentsync)
#   LADDER       Space-separated memory rungs (default: "64g 96g 128g")
#   FIXTURE/...  Passed through to smoke_test.sh same as make run-latentsync

set -euo pipefail

CONTAINER="${CONTAINER:-polyglot-lipsync-latentsync}"
read -r -a LADDER <<< "${LADDER:-64g 96g 128g}"

_restart_count() {
  docker inspect --format '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo 0
}

_current_memory() {
  # HostConfig.Memory is in bytes; format as `<n>g` for readability.
  local bytes
  bytes="$(docker inspect --format '{{.HostConfig.Memory}}' "$CONTAINER" 2>/dev/null || echo 0)"
  if [ "$bytes" -eq 0 ]; then
    echo "unlimited"
  else
    echo "$(( bytes / 1024 / 1024 / 1024 ))g"
  fi
}

_wait_for_running() {
  # After `docker update` or a container restart, give the service a
  # moment to come fully online before hammering it with a new request.
  local max=30 i=0
  while [ "$i" -lt "$max" ]; do
    if docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
      # Also check the health endpoint — container up != service ready.
      if curl -sf "http://localhost:${LATENTSYNC_PORT:-8090}/health" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "!! Service didn't come back online within ${max}s" >&2
  return 1
}

_set_memory() {
  local mem="$1"
  echo "==> docker update --memory=$mem $CONTAINER"
  # --memory-swap must be >= --memory; setting both to the same value
  # disables swap (which is already disabled on this host per free -h).
  docker update --memory="$mem" --memory-swap="$mem" "$CONTAINER" >/dev/null
  _wait_for_running
}

_run_once() {
  # Returns: 0 = success, 2 = OOM (container restarted), other = real failure.
  local rc_before rc_after status=0
  rc_before="$(_restart_count)"

  # scripts/smoke_test.sh is what `make run-latentsync` drives. Pass
  # through LIPSYNC=latentsync and whatever other env the caller set.
  LIPSYNC=latentsync ./scripts/smoke_test.sh || status=$?

  rc_after="$(_restart_count)"
  if [ "$status" -eq 0 ]; then
    return 0
  fi
  if [ "$rc_after" -gt "$rc_before" ]; then
    echo "==> Container restarted during job (RestartCount $rc_before -> $rc_after)."
    echo "    Treating as OOM — the lipsync service doesn't crash for other reasons."
    return 2
  fi
  echo "==> Job failed but container stayed alive — not an OOM, won't escalate memory."
  return "$status"
}

echo "==> Adaptive LatentSync run"
echo "    container: $CONTAINER"
echo "    starting memory: $(_current_memory)"
echo "    ladder: ${LADDER[*]}"

if _run_once; then
  echo "==> Success at $(_current_memory)."
  exit 0
fi
rc=$?
if [ "$rc" -ne 2 ]; then
  exit "$rc"
fi

for mem in "${LADDER[@]}"; do
  _set_memory "$mem"
  echo "==> Retrying at $mem (resume cache should skip the denoising loop)..."
  if _run_once; then
    echo "==> Success at $mem."
    exit 0
  fi
  rc=$?
  if [ "$rc" -ne 2 ]; then
    exit "$rc"
  fi
done

cat >&2 <<EOF
!! Exhausted memory ladder. Job still failing at ${LADDER[-1]}.

   If the last failure was another OOM, the clip is pathologically
   large for this pipeline (4K + very long, or a resolution the
   service never sees). Options:
     - Shrink the clip / re-encode at 1080p
     - Extend the ladder: LADDER="64g 128g 256g" $0
     - Investigate the actual RSS with: docker stats --no-stream $CONTAINER

   If the last failure was not an OOM (container stayed alive),
   check \`make logs-latentsync\` for the real error.
EOF
exit 1
