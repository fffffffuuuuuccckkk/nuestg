#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-pems08}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SEEDS="${SEEDS:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
echo "ST-OOD baselines for ${DATASET}: runnable adapters are CaST-fixed-node, STONE-fixed-node, and STOP."
echo "Remaining external_required methods use results/external_import_templates/*.csv."
"$PYTHON" experiments/run_plan.py --dataset "$DATASET" --kind external

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
  run_if_exists "configs/baselines/${DATASET}/cast.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/stone.py" "$seed"
  run_if_exists "configs/baselines/${DATASET}/stop.py" "$seed"
done
