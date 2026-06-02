#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

run_config() {
  local label="$1"
  local config="$2"
  echo "[run] ${label}: ${config}"
  "$PYTHON" train.py --config "$config" $EXTRA_ARGS
}

run_config "STGCN" "configs/baselines/pems08/stgcn.py"
run_config "GraphWaveNet" "configs/baselines/pems08/graphwavenet.py"
run_config "AGCRN" "configs/baselines/pems08/agcrn.py"
run_config "ST-Norm" "configs/baselines/pems08/stnorm.py"
run_config "D2STGNN" "configs/baselines/pems08/d2stgnn.py"
run_config "STID" "configs/baselines/pems08/stid.py"
run_config "CaST-adapter" "configs/baselines/pems08/cast.py"
run_config "STONE-adapter" "configs/baselines/pems08/stone.py"
run_config "STOP-adapter" "configs/baselines/pems08/stop.py"
