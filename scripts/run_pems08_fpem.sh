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
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${SEED}"

cd "${PROJECT_DIR}"

echo "[FPEM] project=${PROJECT_DIR}"
echo "[FPEM] config=${CONFIG}"
echo "[FPEM] seed=${SEED} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM] ckpt_dir=${CKPT_DIR}"
echo "[FPEM] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER}"
echo "[FPEM] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"
echo "[FPEM] swap_detach_env=${SWAP_DETACH_ENV}"

if [[ "${SKIP_COMPLETED}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ && -f "${CKPT_DIR}/${COMPLETION_MARKER}" ]]; then
  echo "[FPEM] skip completed: ${CKPT_DIR}/${COMPLETION_MARKER}"
  exit 0
fi

# Original-scale loss smoke/run can be tried with:
# EXTRA_ARGS="${EXTRA_ARGS} --set LOSS.train_loss_scale=original"

"${PYTHON}" train.py \
  --config "${CONFIG}" \
  --set "TRAIN.seed=${SEED}" \
  --set "TRAIN.device=${DEVICE}" \
  --set "TRAIN.ckpt_dir=${CKPT_DIR}" \
  --set "TRAIN.auto_resume=${AUTO_RESUME}" \
  --set "TRAIN.resume_from=${RESUME_FROM}" \
  --set "TRAIN.best_select_split=${BEST_SELECT_SPLIT}" \
  --set "TRAIN.best_select_metric=${BEST_SELECT_METRIC}" \
  --set "TRAIN.lr_scheduler=multistep" \
  --set "TRAIN.lr_milestones=${LR_MILESTONES}" \
  --set "TRAIN.lr_gamma=0.3" \
  --set "TRAIN.optimizer=adamw" \
  --set "TRAIN.weight_decay=1e-5" \
  --set "TRAIN.no_decay_for_bias_norm_emb=true" \
  --set "TRAIN.grad_clip=3.0" \
  --set "LOSS.swap_detach_env=${SWAP_DETACH_ENV}" \
  --set "METRICS.mape_threshold=1.0" \
  --set "METRICS.mape_eps=1e-5" \
  --set "METRICS.mape_as_percent=true" \
  ${EXTRA_ARGS}
