#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/baselines/pems08_ood/graphwavenet.py}"
SEEDS="${SEEDS:-2026}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/checkpoints/pems08_ood}"
CKPT_NAME_PREFIX="${CKPT_NAME_PREFIX:-baseline_graphwavenet_full}"
BACKBONE_NAME="${BACKBONE_NAME:-graphwavenet_full}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DROPOUT="${DROPOUT:-0.3}"
TRAIN_LOSS_SCALE="${TRAIN_LOSS_SCALE:-original}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"

echo "[PureGraphWaveNet-OOD] project=${PROJECT_DIR}"
echo "[PureGraphWaveNet-OOD] config=${CONFIG}"
echo "[PureGraphWaveNet-OOD] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[PureGraphWaveNet-OOD] backbone=${BACKBONE_NAME} batch_size=${BATCH_SIZE} lr=${LR} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[PureGraphWaveNet-OOD] train_loss_scale=${TRAIN_LOSS_SCALE}"

for seed in ${SEEDS}; do
  export PYTHONHASHSEED="${seed}"
  run_ckpt_dir="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_NAME_PREFIX}_seed${seed}}"
  echo "[PureGraphWaveNet-OOD] run seed=${seed} ckpt_dir=${run_ckpt_dir}"

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
    --set "TRAIN.optimizer=adam" \
    --set "TRAIN.weight_decay=${WEIGHT_DECAY}" \
    --set "TRAIN.grad_clip=${GRAD_CLIP}" \
    --set "LOSS.train_loss_scale=${TRAIN_LOSS_SCALE}" \
    --set "LOSS.use_inv=false" \
    --set "LOSS.lambda_inv=0.0" \
    --set "METRICS.mape_threshold=1.0" \
    --set "METRICS.mape_eps=1e-5" \
    --set "METRICS.mape_as_percent=true" \
    ${EXTRA_ARGS}
done
