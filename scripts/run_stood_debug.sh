#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DEVICE="${DEVICE:-cuda:0}"
DEBUG_BATCH_SIZE="${DEBUG_BATCH_SIZE:-4}"
RUN_PREPARE="${RUN_PREPARE:-1}"

cd "$PROJECT_DIR"

if [[ "$RUN_PREPARE" == "1" ]]; then
  "$PYTHON" scripts/prepare_stood_datasets.py \
    --datasets newbike_chicago taxi_chicago speed_nyc
fi

"$PYTHON" -m py_compile \
  train.py \
  configs/*.py \
  scripts/prepare_stood_datasets.py \
  models/*.py \
  models/backbones/*.py \
  models/separation/*.py \
  losses/*.py \
  utils/*.py

for config in \
  configs/newbike_chicago_nuestg.py \
  configs/taxi_chicago_nuestg.py \
  configs/speed_nyc_nuestg.py
do
  echo "[ST-OOD debug] $config"
  "$PYTHON" train.py \
    --config "$config" \
    --debug_batch \
    --set "TRAIN.device=${DEVICE}" \
    --set "TRAIN.batch_size=${DEBUG_BATCH_SIZE}" \
    --set "TRAIN.num_workers=0"
done

echo "[ST-OOD debug] done"
