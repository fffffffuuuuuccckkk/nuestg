#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/ours/pems08_ood_fpem_graphwavenet.py}"
SEEDS="${SEEDS:-2026}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/checkpoints/pems08_ood}"
CKPT_NAME_PREFIX="${CKPT_NAME_PREFIX:-fpem_graphwavenet_full}"
BACKBONE_NAME="${BACKBONE_NAME:-graphwavenet_full}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DROPOUT="${DROPOUT:-0.3}"
LR_SCHEDULER="${LR_SCHEDULER:-multistep}"
LR_MILESTONES="${LR_MILESTONES:-[30,60,80]}"
LR_GAMMA="${LR_GAMMA:-0.5}"
LAMBDA_ENVPRED="${LAMBDA_ENVPRED:-0.05}"
LAMBDA_FUTURE_MI="${LAMBDA_FUTURE_MI:-0.01}"
LAMBDA_SWAP="${LAMBDA_SWAP:-0.05}"
LAMBDA_MASK_SPARSE="${LAMBDA_MASK_SPARSE:-1e-3}"
SPARSE_TARGET="${SPARSE_TARGET:-0.3}"
TRAIN_LOSS_SCALE="${TRAIN_LOSS_SCALE:-normalized}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"

echo "[FPEM-GraphWaveNet-OOD] project=${PROJECT_DIR}"
echo "[FPEM-GraphWaveNet-OOD] config=${CONFIG}"
echo "[FPEM-GraphWaveNet-OOD] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM-GraphWaveNet-OOD] backbone=${BACKBONE_NAME} batch_size=${BATCH_SIZE} lr=${LR} optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[FPEM-GraphWaveNet-OOD] lambdas envpred=${LAMBDA_ENVPRED} future_mi=${LAMBDA_FUTURE_MI} swap=${LAMBDA_SWAP} mask_sparse=${LAMBDA_MASK_SPARSE} sparse_target=${SPARSE_TARGET}"
echo "[FPEM-GraphWaveNet-OOD] train_loss_scale=${TRAIN_LOSS_SCALE} scheduler=${LR_SCHEDULER} milestones=${LR_MILESTONES} gamma=${LR_GAMMA}"

for seed in ${SEEDS}; do
  export PYTHONHASHSEED="${seed}"
  run_ckpt_dir="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_NAME_PREFIX}_seed${seed}}"
  echo "[FPEM-GraphWaveNet-OOD] run seed=${seed} ckpt_dir=${run_ckpt_dir}"

  "${PYTHON}" train.py \
    --config "${CONFIG}" \
    --set "MODEL.backbone_name=${BACKBONE_NAME}" \
    --set "MODEL.backbone.name=${BACKBONE_NAME}" \
    --set "MODEL.backbone.graph_wavenet.dropout=${DROPOUT}" \
    --set "MODEL.backbone.graph_wavenet_full.dropout=${DROPOUT}" \
    --set "TRAIN.seed=${seed}" \
    --set "TRAIN.device=${DEVICE}" \
    --set "TRAIN.ckpt_dir=${run_ckpt_dir}" \
    --set "TRAIN.batch_size=${BATCH_SIZE}" \
    --set "TRAIN.learning_rate=${LR}" \
    --set "TRAIN.optimizer=${OPTIMIZER}" \
    --set "TRAIN.weight_decay=${WEIGHT_DECAY}" \
    --set "TRAIN.no_decay_for_bias_norm_emb=true" \
    --set "TRAIN.grad_clip=${GRAD_CLIP}" \
    --set "TRAIN.lr_scheduler=${LR_SCHEDULER}" \
    --set "TRAIN.lr_milestones=${LR_MILESTONES}" \
    --set "TRAIN.lr_gamma=${LR_GAMMA}" \
    --set "LOSS.lambda_envpred=${LAMBDA_ENVPRED}" \
    --set "LOSS.lambda_future_mi=${LAMBDA_FUTURE_MI}" \
    --set "LOSS.lambda_swap=${LAMBDA_SWAP}" \
    --set "LOSS.lambda_mask_sparse=${LAMBDA_MASK_SPARSE}" \
    --set "LOSS.sparse_target=${SPARSE_TARGET}" \
    --set "LOSS.train_loss_scale=${TRAIN_LOSS_SCALE}" \
    --set "METRICS.mape_threshold=1.0" \
    --set "METRICS.mape_eps=1e-5" \
    --set "METRICS.mape_as_percent=true" \
    ${EXTRA_ARGS}
done
