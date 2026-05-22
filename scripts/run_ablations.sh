#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:-pems08}"
SEEDS="${SEEDS:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ABLATIONS="${ABLATIONS:-no_env no_gate no_swap no_persistence no_separation global_env shuffled_env no_kl no_ind no_sparse}"

for seed in $SEEDS; do
  "$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_stid_mlp.py" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
  for ablation in $ABLATIONS; do
    "$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_stid_mlp.py" --ablation "$ablation" --set "TRAIN.seed=${seed}" $EXTRA_ARGS
  done
done
