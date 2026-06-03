#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/ours/pems08_fpem.py}"
SEED="${SEED:-2026}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/pems08_fpem_seed${SEED}}"
LR_MILESTONES="${LR_MILESTONES:-[30,60,80]}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${SEED}"

cd "${PROJECT_DIR}"

echo "[FPEM] project=${PROJECT_DIR}"
echo "[FPEM] config=${CONFIG}"
echo "[FPEM] seed=${SEED} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM] ckpt_dir=${CKPT_DIR}"

# Original-scale loss smoke/run can be tried with:
# EXTRA_ARGS="${EXTRA_ARGS} --set LOSS.train_loss_scale=original"

"${PYTHON}" train.py \
  --config "${CONFIG}" \
  --set "TRAIN.seed=${SEED}" \
  --set "TRAIN.device=${DEVICE}" \
  --set "TRAIN.ckpt_dir=${CKPT_DIR}" \
  --set "TRAIN.lr_scheduler=multistep" \
  --set "TRAIN.lr_milestones=${LR_MILESTONES}" \
  --set "TRAIN.lr_gamma=0.3" \
  --set "TRAIN.optimizer=adamw" \
  --set "TRAIN.weight_decay=1e-5" \
  --set "TRAIN.no_decay_for_bias_norm_emb=true" \
  --set "TRAIN.grad_clip=3.0" \
  --set "METRICS.mape_threshold=1.0" \
  --set "METRICS.mape_eps=1e-5" \
  --set "METRICS.mape_as_percent=true" \
  ${EXTRA_ARGS}
