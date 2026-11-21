#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/run_craftax_collection.sh expert|random [episodes] [budget] [seed_start]' >&2
  exit 2
fi

policy="$1"
episodes="${2:-1000}"
budget="${3:-1000}"
seed_start="${4:-20000}"
artifact_root="$PWD/results"
args=()
case "$policy" in
  expert)
    checkpoint="$artifact_root/checkpoints/craftax-classic-ppo-rnn"
    if [[ ! -d "$checkpoint" ]]; then
      echo 'Fetch the expert first: python3 scripts/fetch_craftax_expert.py' >&2
      exit 1
    fi
    args+=(--expert-checkpoint "$checkpoint")
    ;;
  random) ;;
  *) echo 'Choose expert or random.' >&2; exit 2 ;;
esac

mkdir -p -- "$artifact_root"
run_dir="$artifact_root/craftax-$policy-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'Writing results to %s\n' "$run_dir"
JAX_PLATFORMS=cpu uv run --no-sync wr collect --config configs/craftax/iron_pickaxe.toml \
  --episodes "$episodes" --budget "$budget" --seed-start "$seed_start" \
  --out "$run_dir" "${args[@]}" 2>&1 | tee "$run_dir.log"

printf 'Summary at %s\n' "$run_dir/summary.json"
