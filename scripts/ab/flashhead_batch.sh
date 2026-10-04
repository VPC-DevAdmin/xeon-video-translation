#!/bin/bash
# FlashHead A/B batch, run inside the FlashHead lab container from /experiment.
set -u
cd /experiment
export PYTHONPATH=.
I=/jobs/ab/inputs
O=/jobs/ab/out
run() { label=$1; shift; python stream_render.py "$@" --image $I/portrait.png --label "$label" --output $O > "$O/$label.log" 2>&1; echo "$label exit $?" >> $O/flashhead-batch.status; }
run flashhead-pro-compiled-14s --model pro --compile --audio $I/reply14-16k.wav --playback-audio $I/reply14-master24k.wav
run flashhead-pro-compiled-60s --model pro --compile --audio $I/reply60-16k.wav --playback-audio $I/reply60-master24k.wav
run flashhead-lite-14s         --model lite          --audio $I/reply14-16k.wav --playback-audio $I/reply14-master24k.wav
run flashhead-lite-60s         --model lite          --audio $I/reply60-16k.wav --playback-audio $I/reply60-master24k.wav
run flashhead-pro-eager-14s    --model pro           --audio $I/reply14-16k.wav --playback-audio $I/reply14-master24k.wav
touch $O/flashhead-batch.done
