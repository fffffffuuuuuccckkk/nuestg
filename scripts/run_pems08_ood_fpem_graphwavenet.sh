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
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"
SWAP_FREEZE_PREDICTOR="${SWAP_FREEZE_PREDICTOR:-true}"
LAMBDA_MASK_SPARSE="${LAMBDA_MASK_SPARSE:-1e-3}"
SPARSE_TARGET="${SPARSE_TARGET:-0.3}"
TRAIN_LOSS_SCALE="${TRAIN_LOSS_SCALE:-normalized}"
GRAD_CONSENSUS_ENABLED="${GRAD_CONSENSUS_ENABLED:-false}"
GRAD_CONSENSUS_TARGET="${GRAD_CONSENSUS_TARGET:-z_seq}"
GRAD_CONSENSUS_APPLY_TO="${GRAD_CONSENSUS_APPLY_TO:-inv_branch}"
GRAD_CONSENSUS_MODE="${GRAD_CONSENSUS_MODE:-time_channel}"
GRAD_CONSENSUS_RHO_MAX="${GRAD_CONSENSUS_RHO_MAX:-0.1}"
GRAD_CONSENSUS_GAMMA="${GRAD_CONSENSUS_GAMMA:-1.0}"
GRAD_CONSENSUS_EMA_BETA="${GRAD_CONSENSUS_EMA_BETA:-0.95}"
GRAD_CONSENSUS_WARMUP_EPOCHS="${GRAD_CONSENSUS_WARMUP_EPOCHS:-10}"
GRAD_CONSENSUS_EPS="${GRAD_CONSENSUS_EPS:-1e-8}"
GRAD_CONSENSUS_USE_EMA="${GRAD_CONSENSUS_USE_EMA:-true}"
GRAD_CONSENSUS_LOSS_TYPE="${GRAD_CONSENSUS_LOSS_TYPE:-mse}"
GRAD_CONSENSUS_LOG_STATS="${GRAD_CONSENSUS_LOG_STATS:-true}"
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

echo "[FPEM-GraphWaveNet-OOD] project=${PROJECT_DIR}"
echo "[FPEM-GraphWaveNet-OOD] config=${CONFIG}"
echo "[FPEM-GraphWaveNet-OOD] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM-GraphWaveNet-OOD] backbone=${BACKBONE_NAME} batch_size=${BATCH_SIZE} lr=${LR} optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[FPEM-GraphWaveNet-OOD] lambdas envpred=${LAMBDA_ENVPRED} future_mi=${LAMBDA_FUTURE_MI} swap=${LAMBDA_SWAP} mask_sparse=${LAMBDA_MASK_SPARSE} sparse_target=${SPARSE_TARGET}"
echo "[FPEM-GraphWaveNet-OOD] swap_detach_env=${SWAP_DETACH_ENV} swap_freeze_predictor=${SWAP_FREEZE_PREDICTOR}"
echo "[FPEM-GraphWaveNet-OOD] train_loss_scale=${TRAIN_LOSS_SCALE} scheduler=${LR_SCHEDULER} milestones=${LR_MILESTONES} gamma=${LR_GAMMA}"
echo "[FPEM-GraphWaveNet-OOD] grad_consensus enabled=${GRAD_CONSENSUS_ENABLED} target=${GRAD_CONSENSUS_TARGET} mode=${GRAD_CONSENSUS_MODE} rho_max=${GRAD_CONSENSUS_RHO_MAX} gamma=${GRAD_CONSENSUS_GAMMA} ema_beta=${GRAD_CONSENSUS_EMA_BETA} warmup=${GRAD_CONSENSUS_WARMUP_EPOCHS}"
echo "[FPEM-GraphWaveNet-OOD] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"
echo "[FPEM-GraphWaveNet-OOD] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"

for seed in ${SEEDS}; do
  export PYTHONHASHSEED="${seed}"
  run_ckpt_dir="${CKPT_DIR:-${CKPT_ROOT}/${CKPT_NAME_PREFIX}_seed${seed}}"
  if [[ "${SKIP_COMPLETED}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ && -f "${run_ckpt_dir}/${COMPLETION_MARKER}" ]]; then
    echo "[FPEM-GraphWaveNet-OOD] skip completed seed=${seed}: ${run_ckpt_dir}/${COMPLETION_MARKER}"
    continue
  fi
  echo "[FPEM-GraphWaveNet-OOD] run seed=${seed} ckpt_dir=${run_ckpt_dir}"

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
    --set "LOSS.swap_detach_env=${SWAP_DETACH_ENV}" \
    --set "SWAP.freeze_predictor=${SWAP_FREEZE_PREDICTOR}" \
    --set "LOSS.lambda_mask_sparse=${LAMBDA_MASK_SPARSE}" \
    --set "LOSS.sparse_target=${SPARSE_TARGET}" \
    --set "LOSS.train_loss_scale=${TRAIN_LOSS_SCALE}" \
    --set "LOSS.grad_consensus.enabled=${GRAD_CONSENSUS_ENABLED}" \
    --set "LOSS.grad_consensus.target=${GRAD_CONSENSUS_TARGET}" \
    --set "LOSS.grad_consensus.apply_to=${GRAD_CONSENSUS_APPLY_TO}" \
    --set "LOSS.grad_consensus.mode=${GRAD_CONSENSUS_MODE}" \
    --set "LOSS.grad_consensus.rho_max=${GRAD_CONSENSUS_RHO_MAX}" \
    --set "LOSS.grad_consensus.gamma=${GRAD_CONSENSUS_GAMMA}" \
    --set "LOSS.grad_consensus.ema_beta=${GRAD_CONSENSUS_EMA_BETA}" \
    --set "LOSS.grad_consensus.warmup_epochs=${GRAD_CONSENSUS_WARMUP_EPOCHS}" \
    --set "LOSS.grad_consensus.eps=${GRAD_CONSENSUS_EPS}" \
    --set "LOSS.grad_consensus.use_ema=${GRAD_CONSENSUS_USE_EMA}" \
    --set "LOSS.grad_consensus.loss_type=${GRAD_CONSENSUS_LOSS_TYPE}" \
    --set "LOSS.grad_consensus.log_stats=${GRAD_CONSENSUS_LOG_STATS}" \
    --set "METRICS.mape_threshold=1.0" \
    --set "METRICS.mape_eps=1e-5" \
    --set "METRICS.mape_as_percent=true" \
    ${EXTRA_ARGS}
  exit_code=$?
  set -e
  mkdir -p "${run_ckpt_dir}"
  printf "exit_code=%s\n" "${exit_code}" > "${run_ckpt_dir}/last_exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[FPEM-GraphWaveNet-OOD] seed=${seed} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
    if [[ ! "${CONTINUE_ON_FAILURE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ ]]; then
      exit "${exit_code}"
    fi
  fi
done
