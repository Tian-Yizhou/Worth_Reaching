#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../../../.."

if [[ $# -lt 1 || $# -gt 5 ]]; then
  printf 'Usage: bash %q expert|random [episodes] [budget] [seed_start] [task]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

policy="$1"
episodes="${2:-1000}"
budget="${3:-1000}"
seed_start="${4:-20000}"
task="${5:-make_iron_pickaxe}"
artifact_root="$PWD/results"
dataset_root="$artifact_root/datasets"
args=()
case "$policy" in
  expert)
    checkpoint="$artifact_root/checkpoints/craftax-classic-ppo-rnn"
    if [[ ! -d "$checkpoint" ]]; then
      printf 'Fetch the expert first: python3 %q\n' "$script_dir/fetch_craftax_expert.py" >&2
      exit 1
    fi
    args+=(--expert-checkpoint "$checkpoint")
    ;;
  random) ;;
  *) echo 'Choose expert or random.' >&2; exit 2 ;;
esac

mkdir -p -- "$dataset_root"
run_dir="$dataset_root/craftax-$policy-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'Writing results to %s\n' "$run_dir"
uv run --no-sync wr collect --config configs/craftax/iron_pickaxe.toml \
  --episodes "$episodes" --budget "$budget" --seed-start "$seed_start" --task "$task" \
  --out "$run_dir" "${args[@]}" 2>&1 | tee "$run_dir.log"

printf 'Summary at %s\n' "$run_dir/summary.json"
