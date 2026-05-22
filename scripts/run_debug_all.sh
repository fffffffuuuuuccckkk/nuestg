#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:-pems08}"
DEBUG_BATCH_SIZE="${DEBUG_BATCH_SIZE:-4}"
COMMON_SET=(--set "TRAIN.batch_size=${DEBUG_BATCH_SIZE}" --set "TRAIN.num_workers=0")

"$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_stid_mlp.py" --debug_batch "${COMMON_SET[@]}"
"$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_graphwavenet.py" --debug_batch "${COMMON_SET[@]}"
"$PYTHON" train.py --config "configs/ours/${DATASET}/nuestg_agcrn.py" --debug_batch "${COMMON_SET[@]}"

"$PYTHON" train.py --config "configs/baselines/${DATASET}/stid_mlp.py" --debug_batch "${COMMON_SET[@]}"
"$PYTHON" train.py --config "configs/baselines/${DATASET}/graphwavenet.py" --debug_batch "${COMMON_SET[@]}"
"$PYTHON" train.py --config "configs/baselines/${DATASET}/agcrn.py" --debug_batch "${COMMON_SET[@]}"
