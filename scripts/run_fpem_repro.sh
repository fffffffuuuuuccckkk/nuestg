#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"

RUN_COMPILE="${RUN_COMPILE:-1}"
RUN_OLD_NUE="${RUN_OLD_NUE:-1}"
RUN_SET_VARIANT="${RUN_SET_VARIANT:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"

cd "$PROJECT_DIR"

COMMON_SET=(
  --set "TRAIN.seed=${SEED}"
  --set "TRAIN.batch_size=${BATCH_SIZE}"
  --set "TRAIN.num_workers=${NUM_WORKERS}"
  --set "TRAIN.device=${DEVICE}"
)

echo "[FPEM repro] project: $PROJECT_DIR"
echo "[FPEM repro] python:  $PYTHON"
echo "[FPEM repro] seed=${SEED} batch_size=${BATCH_SIZE} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

if [[ "$RUN_COMPILE" == "1" ]]; then
  echo "[FPEM repro] py_compile"
  "$PYTHON" -m py_compile \
    train.py \
    configs/*.py \
    configs/ours/*.py \
    configs/ours/pems08/*.py \
    models/*.py \
    models/backbones/*.py \
    models/separation/*.py \
    losses/*.py \
    utils/*.py
fi

echo "[FPEM repro] debug: configs/ours/pems08_fpem.py"
"$PYTHON" train.py \
  --config configs/ours/pems08_fpem.py \
  --debug_batch \
  "${COMMON_SET[@]}"

if [[ "$RUN_SET_VARIANT" == "1" ]]; then
  echo "[FPEM repro] debug: base config with --set MODEL.method_variant=fpem"
  "$PYTHON" train.py \
    --config configs/pems08_nuestg.py \
    --debug_batch \
    --set MODEL.method_variant=fpem \
    "${COMMON_SET[@]}"
fi

if [[ "$RUN_OLD_NUE" == "1" ]]; then
  echo "[FPEM repro] debug: old NUE-STG compatibility"
  "$PYTHON" train.py \
    --config configs/pems08_nuestg.py \
    --debug_batch \
    "${COMMON_SET[@]}"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  EPOCHS="${EPOCHS:-100}"
  CKPT_DIR="${CKPT_DIR:-./checkpoints/pems08_fpem_seed${SEED}}"
  echo "[FPEM repro] train: configs/ours/pems08_fpem.py epochs=${EPOCHS} ckpt_dir=${CKPT_DIR}"
  "$PYTHON" train.py \
    --config configs/ours/pems08_fpem.py \
    "${COMMON_SET[@]}" \
    --set "TRAIN.epochs=${EPOCHS}" \
    --set "TRAIN.ckpt_dir=${CKPT_DIR}"
fi

echo "[FPEM repro] done"
