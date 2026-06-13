#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/ours/pems08_fpem.py}"
SWEEP_NAME="${SWEEP_NAME:-pems08_fpem_perturb_consistency_ablation}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/${SWEEP_NAME}}"
BACKBONE="${BACKBONE:-stid_mlp}"
SEED="${SEED:-2026}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-true}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS_BASE="${EXTRA_ARGS_BASE:-}"
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"

LR="${LR:-0.001}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-3.0}"
LR_SCHEDULER="${LR_SCHEDULER:-multistep}"
LR_MILESTONES="${LR_MILESTONES:-[30,60,80]}"
LR_GAMMA="${LR_GAMMA:-0.5}"

DEFAULT_ABLATIONS=(no_perturb_cons z_cons_only y_cons_only z_y_cons strong_z_y_cons)

truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

append_common_args() {
  local -n out_args=$1
  out_args+=(
    "--set" "MODEL.backbone_name=${BACKBONE}"
    "--set" "MODEL.backbone.name=${BACKBONE}"
    "--set" "TRAIN.seed=${SEED}"
    "--set" "TRAIN.device=${DEVICE}"
    "--set" "TRAIN.auto_resume=${AUTO_RESUME}"
    "--set" "TRAIN.resume_from=${RESUME_FROM}"
    "--set" "TRAIN.best_select_split=${BEST_SELECT_SPLIT}"
    "--set" "TRAIN.best_select_metric=${BEST_SELECT_METRIC}"
    "--set" "TRAIN.learning_rate=${LR}"
    "--set" "TRAIN.optimizer=${OPTIMIZER}"
    "--set" "TRAIN.weight_decay=${WEIGHT_DECAY}"
    "--set" "TRAIN.grad_clip=${GRAD_CLIP}"
    "--set" "TRAIN.lr_scheduler=${LR_SCHEDULER}"
    "--set" "TRAIN.lr_milestones=${LR_MILESTONES}"
    "--set" "TRAIN.lr_gamma=${LR_GAMMA}"
    "--set" "LOSS.swap_detach_env=${SWAP_DETACH_ENV}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    out_args+=("--set" "TRAIN.epochs=${EPOCHS}")
  fi
}

append_ablation_args() {
  local ablation="$1"
  local -n out_args=$2
  case "${ablation}" in
    no_perturb_cons)
      out_args+=(
        "--set" "MODEL.perturb_enabled=false"
        "--set" "LOSS.lambda_z_cons=0"
        "--set" "LOSS.lambda_y_cons=0"
      )
      ;;
    z_cons_only)
      out_args+=(
        "--set" "MODEL.perturb_enabled=true"
        "--set" "LOSS.lambda_z_cons=0.01"
        "--set" "LOSS.lambda_y_cons=0"
      )
      ;;
    y_cons_only)
      out_args+=(
        "--set" "MODEL.perturb_enabled=true"
        "--set" "LOSS.lambda_z_cons=0"
        "--set" "LOSS.lambda_y_cons=0.01"
      )
      ;;
    z_y_cons)
      out_args+=(
        "--set" "MODEL.perturb_enabled=true"
        "--set" "LOSS.lambda_z_cons=0.01"
        "--set" "LOSS.lambda_y_cons=0.01"
      )
      ;;
    strong_z_y_cons)
      out_args+=(
        "--set" "MODEL.perturb_enabled=true"
        "--set" "LOSS.lambda_z_cons=0.05"
        "--set" "LOSS.lambda_y_cons=0.05"
      )
      ;;
    *)
      echo "[perturb-ablation] unsupported ablation: ${ablation}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local ablation="$1"
  local ckpt_dir="${PROJECT_DIR}/${CKPT_ROOT}/${ablation}"
  local complete_path="${ckpt_dir}/${COMPLETION_MARKER}"
  if truthy "${SKIP_COMPLETED}" && [[ -f "${complete_path}" ]]; then
    echo
    echo "[perturb-ablation] skip completed ${ablation}: ${complete_path}"
    return 0
  fi
  local args=("--config" "${CONFIG}")
  append_common_args args
  append_ablation_args "${ablation}" args
  args+=("--set" "TRAIN.ckpt_dir=${ckpt_dir}")
  if [[ -n "${EXTRA_ARGS_BASE}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_ARGS_BASE} )
    args+=("${extra[@]}")
  fi

  mkdir -p "${ckpt_dir}"
  local quoted
  quoted="$(printf "%q " "${args[@]}")"
  {
    printf "ablation=%s\n" "${ablation}"
    printf "backbone=%s\n" "${BACKBONE}"
    printf "seed=%s\n" "${SEED}"
    printf "swap_detach_env=%s\n" "${SWAP_DETACH_ENV}"
    printf "command=%q train.py %s\n" "${PYTHON}" "${quoted}"
  } > "${ckpt_dir}/ablation_command.env"

  echo
  echo "[perturb-ablation] ${ablation}"
  echo "[perturb-ablation] ckpt_dir=${ckpt_dir}"
  echo "[cmd] CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} train.py ${quoted}"
  if truthy "${DRY_RUN}"; then
    return 0
  fi
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONHASHSEED="${SEED}" "${PYTHON}" train.py "${args[@]}"
  local exit_code=$?
  set -e
  printf "exit_code=%s\n" "${exit_code}" > "${ckpt_dir}/last_exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[perturb-ablation] ${ablation} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
    if ! truthy "${CONTINUE_ON_FAILURE}"; then
      exit "${exit_code}"
    fi
  fi
}

cd "${PROJECT_DIR}"
if [[ -n "${ABLATIONS:-}" ]]; then
  # shellcheck disable=SC2206
  selected_ablations=( ${ABLATIONS} )
else
  selected_ablations=("${DEFAULT_ABLATIONS[@]}")
fi

echo "[perturb-ablation] project=${PROJECT_DIR}"
echo "[perturb-ablation] config=${CONFIG}"
echo "[perturb-ablation] dry_run=${DRY_RUN} backbone=${BACKBONE} seed=${SEED} device=${DEVICE}"
echo "[perturb-ablation] swap_detach_env=${SWAP_DETACH_ENV}"
echo "[perturb-ablation] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"
echo "[perturb-ablation] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"
echo "[perturb-ablation] selected=${selected_ablations[*]}"
if ! truthy "${DRY_RUN}" && [[ -z "${EPOCHS}" ]]; then
  echo "[perturb-ablation] WARNING: DRY_RUN=false with EPOCHS unset will use config TRAIN.epochs." >&2
fi

for ablation in "${selected_ablations[@]}"; do
  run_one "${ablation}"
done
