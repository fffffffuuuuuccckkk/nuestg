#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-pems08}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
echo "ST-OOD baselines for ${DATASET} are external_required in this repository."
echo "Use results/external_import_templates/*.csv and place completed imports under results/raw/."
"$PYTHON" experiments/run_plan.py --dataset "$DATASET" --kind external
