#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:-pems08}"
SEEDS="${SEEDS:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for seed in $SEEDS; do
  "$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_stid_mlp.py" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
  "$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_graphwavenet.py" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
  "$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_agcrn.py" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
done
