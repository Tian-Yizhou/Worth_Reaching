#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: bash $0 DATASET CHECKPOINT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi

dataset=$1
checkpoint=$2
out=$3
mkdir "$out"

run() {
    local name=$1 repeats=$2
    shift 2
    JAX_PLATFORMS=cpu timeout 90s uv run --no-sync wr waypoint-pilot \
        --dataset "$dataset" --checkpoint "$checkpoint" --out "$out/$name" \
        --split dev --offer-expert --seed 1729 --repeats "$repeats" "$@" \
        > "$out/$name.log" 2>&1 || {
        local status=$?
        printf '%s exited with status %s. See %s\n' "$name" "$status" "$out/$name.log" >&2
        return "$status"
    }
}

pids=()
for slip in 0 0.25 0.5; do
    (
        for skip in 12 18 24 30; do
            run "near-$slip-$skip" 8 --worlds 6 --skip-worlds "$skip" \
                --horizon 32 --slip-probability "$slip" || exit
        done
    ) &
    pids+=("$!")
done

(
    for slip in 0 0.25; do
        for skip in 4 6 8 10; do
            run "early-success-$slip-$skip" 8 --worlds 2 --skip-worlds "$skip" \
                --prefix-steps 96 --horizon 128 --source-outcome success \
                --slip-probability "$slip" || exit
        done
        for batch in "0 2" "2 2" "4 1" "5 1"; do
            read -r skip count <<< "$batch"
            run "early-death-$slip-$skip" 8 --worlds "$count" --skip-worlds "$skip" \
                --prefix-steps 96 --horizon 128 --source-outcome death \
                --slip-probability "$slip" || exit
        done
    done
) &
pids+=("$!")

(
    run near-short-check 8 --worlds 6 --skip-worlds 12 --horizon 16 \
        --slip-probability 0.25 || exit
    for radius in 0.001 0.004; do
        run "radius-$radius" 8 --worlds 6 --skip-worlds 12 --horizon 32 \
            --slip-probability 0.25 --region-radius "$radius" || exit
    done
) &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    exit "$status"
fi

pids=()
for first in 12 24; do
    (
        for skip in "$first" "$((first + 6))"; do
            run "near-calibration-$skip" 24 --worlds 6 --skip-worlds "$skip" \
                --horizon 16 --slip-probability 0.25 || exit
        done
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
