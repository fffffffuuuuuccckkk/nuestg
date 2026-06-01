#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:-pems08}"
SEEDS="${SEEDS:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

run_if_exists() {
  local cfg="$1"
  local seed="$2"
  if [[ -f "$cfg" ]]; then
    "$PYTHON" train.py --config "$cfg" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
  else
    echo "skip missing config: $cfg"
  fi
}

for seed in $SEEDS; do
  run_if_exists "configs/baselines/${DATASET}/stid.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/graphwavenet.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/agcrn.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/stgcn.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/stnorm.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/d2stgnn.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/stid_mlp.py" "$seed"
done
