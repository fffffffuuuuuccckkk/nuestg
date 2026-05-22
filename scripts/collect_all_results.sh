#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"

"$PYTHON" experiments/collect_results.py --results_dir "$RESULTS_DIR" --checkpoints_dir "$CHECKPOINTS_DIR" --out "$RESULTS_DIR/tables/all_results.csv"
"$PYTHON" experiments/make_tables.py --input "$RESULTS_DIR/tables/all_results.csv" --out_dir "$RESULTS_DIR/tables"
