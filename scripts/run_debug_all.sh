#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:-pems08}"
DEBUG_BATCH_SIZE="${DEBUG_BATCH_SIZE:-4}"
COMMON_SET=(--set "TRAIN.batch_size=${DEBUG_BATCH_SIZE}" --set "TRAIN.num_workers=0")

run_if_exists() {
  local cfg="$1"
  if [[ -f "$cfg" ]]; then
    "$PYTHON" train.py --config "$cfg" --debug_batch "${COMMON_SET[@]}"
  else
    echo "skip missing config: $cfg"
  fi
}

run_if_exists "configs/ours/${DATASET}/nuestg_stid_mlp.py"
run_if_exists "configs/ours/${DATASET}/nuestg_graphwavenet.py"
run_if_exists "configs/ours/${DATASET}/nuestg_agcrn.py"

run_if_exists "configs/baselines/${DATASET}/stid.py"
run_if_exists "configs/baselines/${DATASET}/graphwavenet.py"
run_if_exists "configs/baselines/${DATASET}/agcrn.py"
run_if_exists "configs/baselines/${DATASET}/stgcn.py"
run_if_exists "configs/baselines/${DATASET}/stnorm.py"
run_if_exists "configs/baselines/${DATASET}/d2stgnn.py"
run_if_exists "configs/baselines/${DATASET}/cast.py"
run_if_exists "configs/baselines/${DATASET}/stone.py"
run_if_exists "configs/baselines/${DATASET}/stop.py"
run_if_exists "configs/baselines/${DATASET}/stid_mlp.py"
