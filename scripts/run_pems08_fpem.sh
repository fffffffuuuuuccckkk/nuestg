#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

"$PYTHON" train.py --config configs/ours/pems08/fpem_stid_mlp.py $EXTRA_ARGS
