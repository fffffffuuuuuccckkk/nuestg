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
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"

echo "[PureGraphWaveNet-OOD] project=${PROJECT_DIR}"
echo "[PureGraphWaveNet-OOD] config=${CONFIG}"
echo "[PureGraphWaveNet-OOD] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[PureGraphWaveNet-OOD] backbone=${BACKBONE_NAME} batch_size=${BATCH_SIZE} lr=${LR} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[PureGraphWaveNet-OOD] train_loss_scale=${TRAIN_LOSS_SCALE}"
echo "[PureGraphWaveNet-OOD] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"
echo "[PureGraphWaveNet-OOD] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"

for seed in ${SEEDS}; do
  export PYTHONHASHSEED="${seed}"
  run_ckpt_dir="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_NAME_PREFIX}_seed${seed}}"
  if [[ "${SKIP_COMPLETED}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ && -f "${run_ckpt_dir}/${COMPLETION_MARKER}" ]]; then
    echo "[PureGraphWaveNet-OOD] skip completed seed=${seed}: ${run_ckpt_dir}/${COMPLETION_MARKER}"
    continue
  fi
  echo "[PureGraphWaveNet-OOD] run seed=${seed} ckpt_dir=${run_ckpt_dir}"

  set +e
  "${PYTHON}" train.py \
    --config "${CONFIG}" \
    --set "MODEL.backbone_name=${BACKBONE_NAME}" \
    --set "MODEL.backbone.name=${BACKBONE_NAME}" \
    --set "MODEL.backbone.graph_wavenet.dropout=${DROPOUT}" \
    --set "MODEL.backbone.graph_wavenet_full.dropout=${DROPOUT}" \
    --set "TRAIN.seed=${seed}" \
    --set "TRAIN.device=${DEVICE}" \
    --set "TRAIN.ckpt_dir=${run_ckpt_dir}" \
    --set "TRAIN.auto_resume=${AUTO_RESUME}" \
    --set "TRAIN.resume_from=${RESUME_FROM}" \
    --set "TRAIN.best_select_split=${BEST_SELECT_SPLIT}" \
    --set "TRAIN.best_select_metric=${BEST_SELECT_METRIC}" \
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
  exit_code=$?
  set -e
  mkdir -p "${run_ckpt_dir}"
  printf "exit_code=%s\n" "${exit_code}" > "${run_ckpt_dir}/last_exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[PureGraphWaveNet-OOD] seed=${seed} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
    if [[ ! "${CONTINUE_ON_FAILURE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ ]]; then
      exit "${exit_code}"
    fi
  fi
done
