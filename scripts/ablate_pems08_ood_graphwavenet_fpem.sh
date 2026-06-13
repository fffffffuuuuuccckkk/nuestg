#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"

# Kept as the requested default fallback. For the PEMS08-OOD protocol, this
# script prefers FPEM_CONFIG below when it exists.
DATASET_CONFIG="${DATASET_CONFIG:-configs/ours/pems08_fpem.py}"
FPEM_CONFIG="${FPEM_CONFIG:-configs/ours/pems08_ood_fpem_graphwavenet.py}"
PURE_GRAPHWAVENET_CONFIG="${PURE_GRAPHWAVENET_CONFIG:-configs/baselines/pems08_ood/graphwavenet.py}"

BACKBONE="${BACKBONE:-graphwavenet}"
SWEEP_NAME="${SWEEP_NAME:-pems08_ood_graphwavenet_ablation}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/${SWEEP_NAME}}"
SEED="${SEED:-2026}"
DRY_RUN="${DRY_RUN:-true}"
EPOCHS="${EPOCHS:-}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"

# AdamW is intentionally kept as the default to match the current FPEM sweep
# profile. Set OPTIMIZER=adam to compare against the original Graph WaveNet
# optimizer more directly.
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DROPOUT="${DROPOUT:-0.3}"
LR_SCHEDULER="${LR_SCHEDULER:-multistep}"
LR_MILESTONES="${LR_MILESTONES:-[30,60,80]}"
LR_GAMMA="${LR_GAMMA:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"

DEFAULT_DRY_RUN_ABLATIONS=(
  pure_graphwavenet
  fpem_no_env
  fpem_full
  no_future_mi
  no_envpred
  no_future_constraints
  no_swap
  weak_swap
  no_sparse
  sparse_target_0_5
  all_env_or_no_mask
  no_env_neighbor
  env_neighbor_none
  no_separation
  no_kl
  fusion_film
  fusion_concat
  fusion_gated_add
  loss_normalized
  loss_original
  peak_weight_0_2
)
DEFAULT_RUN_ABLATIONS=(pure_graphwavenet fpem_no_env fpem_full)

truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_ablation() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    full) printf "fpem_full" ;;
    no_env) printf "fpem_no_env" ;;
    *) printf "%s" "$1" ;;
  esac
}

append_base_profile_args() {
  local -n out_args_ref=$1
  out_args_ref+=(
    "--set" "MODEL.backbone_name=${BACKBONE}"
    "--set" "MODEL.backbone.name=${BACKBONE}"
    "--set" "TRAIN.auto_resume=${AUTO_RESUME}"
    "--set" "TRAIN.resume_from=${RESUME_FROM}"
    "--set" "TRAIN.best_select_split=${BEST_SELECT_SPLIT}"
    "--set" "TRAIN.best_select_metric=${BEST_SELECT_METRIC}"
    "--set" "TRAIN.optimizer=${OPTIMIZER}"
    "--set" "TRAIN.learning_rate=${LR}"
    "--set" "TRAIN.weight_decay=${WEIGHT_DECAY}"
    "--set" "TRAIN.grad_clip=${GRAD_CLIP}"
    "--set" "TRAIN.lr_scheduler=${LR_SCHEDULER}"
    "--set" "TRAIN.lr_milestones=${LR_MILESTONES}"
    "--set" "TRAIN.lr_gamma=${LR_GAMMA}"
    "--set" "LOSS.swap_detach_env=${SWAP_DETACH_ENV}"
    # Current code consumes graph_wavenet, not graphwavenet.
    "--set" "MODEL.backbone.graph_wavenet.dropout=${DROPOUT}"
    "--set" "MODEL.GWNET.dropout=${DROPOUT}"
    "--set" "METRICS.mape_threshold=1.0"
    "--set" "METRICS.mape_eps=1e-5"
    "--set" "METRICS.mape_as_percent=true"
  )
  if [[ -n "${BATCH_SIZE}" ]]; then
    out_args_ref+=("--set" "TRAIN.batch_size=${BATCH_SIZE}")
  fi
  if [[ -n "${EPOCHS}" ]]; then
    out_args_ref+=("--set" "TRAIN.epochs=${EPOCHS}")
  fi
}

append_ablation_args() {
  local ablation="$1"
  local -n ablation_args_ref=$2
  local -n notes_ref=$3
  case "${ablation}" in
    pure_graphwavenet)
      notes_ref="pure GraphWaveNet baseline; no FPEM environment branch"
      ablation_args_ref+=(
        "--set" "LOSS.train_loss_scale=original"
        "--set" "LOSS.use_inv=false"
        "--set" "LOSS.lambda_inv=0.0"
      )
      ;;
    fpem_no_env)
      notes_ref="FPEM backbone only: no_env ablation plus force_mask_value=0 so env_plus is zero"
      ablation_args_ref+=(
        "--ablation" "no_env"
        "--set" "MODEL.force_mask_value=0.0"
        "--set" "LOSS.lambda_envpred=0.0"
        "--set" "LOSS.lambda_future_mi=0.0"
        "--set" "LOSS.lambda_swap=0.0"
        "--set" "LOSS.lambda_kl=0.0"
        "--set" "LOSS.lambda_mask_sparse=0.0"
      )
      ;;
    fpem_full)
      notes_ref="full FPEM with GraphWaveNet backbone"
      ;;
    no_future_mi)
      notes_ref="disable future MI BA/NLL term"
      ablation_args_ref+=("--set" "LOSS.lambda_future_mi=0.0")
      ;;
    no_envpred)
      notes_ref="disable environment future-prediction auxiliary term"
      ablation_args_ref+=("--set" "LOSS.lambda_envpred=0.0")
      ;;
    no_future_constraints)
      notes_ref="disable both future MI and environment future-prediction terms"
      ablation_args_ref+=(
        "--set" "LOSS.lambda_future_mi=0.0"
        "--set" "LOSS.lambda_envpred=0.0"
      )
      ;;
    no_swap)
      notes_ref="disable swap regularization"
      ablation_args_ref+=("--set" "LOSS.lambda_swap=0.0")
      ;;
    weak_swap)
      notes_ref="weaken swap regularization"
      ablation_args_ref+=("--set" "LOSS.lambda_swap=0.01")
      ;;
    no_sparse)
      notes_ref="disable mask sparse penalty"
      ablation_args_ref+=("--set" "LOSS.lambda_mask_sparse=0.0")
      ;;
    sparse_target_0_5)
      notes_ref="move mask sparse target from default toward denser masks"
      ablation_args_ref+=("--set" "LOSS.sparse_target=0.5")
      ;;
    all_env_or_no_mask)
      notes_ref="supported: force mask to 1.0, so all history-env tokens are selected; sparse penalty disabled"
      ablation_args_ref+=(
        "--set" "MODEL.force_mask_value=1.0"
        "--set" "LOSS.lambda_mask_sparse=0.0"
      )
      ;;
    no_env_neighbor)
      notes_ref="disable neighbor aggregation in environment encoders"
      ablation_args_ref+=("--set" "MODEL.env_use_neighbor=false")
      ;;
    env_neighbor_none)
      notes_ref="supported: env_neighbor_mix accepts None; env_use_neighbor=false is added because the encoder switch is env_use_neighbor"
      ablation_args_ref+=(
        "--set" "MODEL.env_neighbor_mix=none"
        "--set" "MODEL.env_use_neighbor=false"
      )
      ;;
    no_separation)
      notes_ref="disable separation module"
      ablation_args_ref+=(
        "--set" "MODEL.separation.enabled=false"
        "--set" "MODEL.separation.mode=none"
      )
      ;;
    no_kl)
      notes_ref="disable KL bottleneck loss"
      ablation_args_ref+=("--set" "LOSS.lambda_kl=0.0")
      ;;
    fusion_film)
      notes_ref="use FiLM latent fusion"
      ablation_args_ref+=("--set" "MODEL.fusion_type=film")
      ;;
    fusion_concat)
      notes_ref="use concat latent fusion"
      ablation_args_ref+=("--set" "MODEL.fusion_type=concat")
      ;;
    fusion_gated_add)
      notes_ref="supported: use gated_add latent fusion"
      ablation_args_ref+=("--set" "MODEL.fusion_type=gated_add")
      ;;
    loss_normalized)
      notes_ref="train prediction loss on normalized scale"
      ablation_args_ref+=("--set" "LOSS.train_loss_scale=normalized")
      ;;
    loss_original)
      notes_ref="train prediction loss on original/raw scale"
      ablation_args_ref+=("--set" "LOSS.train_loss_scale=original")
      ;;
    peak_weight_0_2)
      notes_ref="enable peak-aware training loss with +0.2 weight above 0.75 quantile"
      ablation_args_ref+=(
        "--set" "LOSS.peak_weight_enabled=true"
        "--set" "LOSS.peak_weight=0.2"
        "--set" "LOSS.peak_quantile=0.75"
      )
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

run_one() {
  local raw_ablation="$1"
  local ablation
  ablation="$(normalize_ablation "${raw_ablation}")"

  local config_path="${FPEM_CONFIG}"
  local notes=""
  local args=()
  if [[ "${ablation}" == "pure_graphwavenet" ]]; then
    config_path="${PURE_GRAPHWAVENET_CONFIG}"
    if [[ ! -f "${config_path}" ]]; then
      echo "[WARN] skip pure_graphwavenet: OOD GraphWaveNet config not found: ${config_path}" >&2
      return 0
    fi
  elif [[ ! -f "${config_path}" ]]; then
    if [[ -f "${DATASET_CONFIG}" ]]; then
      echo "[WARN] ${FPEM_CONFIG} not found; falling back to DATASET_CONFIG=${DATASET_CONFIG}" >&2
      config_path="${DATASET_CONFIG}"
    else
      echo "[WARN] skip ${ablation}: FPEM config not found: ${FPEM_CONFIG}; fallback missing: ${DATASET_CONFIG}" >&2
      return 0
    fi
  fi

  append_base_profile_args args
  if ! append_ablation_args "${ablation}" args notes; then
    echo "[WARN] skip unsupported ablation: ${raw_ablation}" >&2
    return 0
  fi

  local ckpt_dir="${PROJECT_DIR}/${CKPT_ROOT}/${ablation}"
  local complete_path="${ckpt_dir}/${COMPLETION_MARKER}"
  if truthy "${SKIP_COMPLETED}" && [[ -f "${complete_path}" ]]; then
    echo
    echo "[ablation] skip completed ${ablation}: ${complete_path}"
    return 0
  fi
  args+=(
    "--set" "TRAIN.seed=${SEED}"
    "--set" "TRAIN.device=${DEVICE}"
    "--set" "TRAIN.ckpt_dir=${ckpt_dir}"
  )
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_ARGS} )
    args+=("${extra[@]}")
  fi

  local quoted_args
  quoted_args="$(printf "%q " "${args[@]}")"
  mkdir -p "${ckpt_dir}"
  {
    printf "ablation=%s\n" "${ablation}"
    printf "raw_ablation=%s\n" "${raw_ablation}"
    printf "config=%s\n" "${config_path}"
    printf "backbone=%s\n" "${BACKBONE}"
    printf "seed=%s\n" "${SEED}"
    printf "swap_detach_env=%s\n" "${SWAP_DETACH_ENV}"
    printf "notes=%s\n" "${notes}"
    printf "command=%q train.py --config %q %s\n" "${PYTHON}" "${config_path}" "${quoted_args}"
  } > "${ckpt_dir}/ablation_command.env"

  echo
  echo "[ablation] ${ablation}"
  echo "[ablation] config=${config_path}"
  echo "[ablation] ckpt_dir=${ckpt_dir}"
  echo "[ablation] notes=${notes}"
  echo "[cmd] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON} train.py --config ${config_path} ${quoted_args}"

  if truthy "${DRY_RUN}"; then
    return 0
  fi
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONHASHSEED="${SEED}" \
    "${PYTHON}" train.py --config "${config_path}" "${args[@]}"
  local exit_code=$?
  set -e
  printf "exit_code=%s\n" "${exit_code}" > "${ckpt_dir}/last_exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[ablation] ${ablation} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
    if ! truthy "${CONTINUE_ON_FAILURE}"; then
      exit "${exit_code}"
    fi
  fi
}

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES

if [[ -n "${ABLATIONS:-}" ]]; then
  # shellcheck disable=SC2206
  selected_ablations=( ${ABLATIONS} )
elif truthy "${DRY_RUN}"; then
  selected_ablations=("${DEFAULT_DRY_RUN_ABLATIONS[@]}")
else
  selected_ablations=("${DEFAULT_RUN_ABLATIONS[@]}")
  echo "[WARN] DRY_RUN=false and ABLATIONS is empty; running only core ablations: ${selected_ablations[*]}" >&2
fi

echo "[ablation] project=${PROJECT_DIR}"
echo "[ablation] dry_run=${DRY_RUN} seed=${SEED} device=${DEVICE} gpu=${GPU} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[ablation] ckpt_root=${PROJECT_DIR}/${CKPT_ROOT}"
echo "[ablation] fpm_config=${FPEM_CONFIG} pure_config=${PURE_GRAPHWAVENET_CONFIG} fallback_dataset_config=${DATASET_CONFIG}"
echo "[ablation] backbone=${BACKBONE} optimizer=${OPTIMIZER} lr=${LR} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[ablation] swap_detach_env=${SWAP_DETACH_ENV}"
echo "[ablation] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"
echo "[ablation] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"
echo "[ablation] selected=${selected_ablations[*]}"
if ! truthy "${DRY_RUN}" && [[ -z "${EPOCHS}" ]]; then
  echo "[WARN] DRY_RUN=false with EPOCHS unset will use config TRAIN.epochs. Set EPOCHS=20 for a bounded smoke run." >&2
fi

for ablation in "${selected_ablations[@]}"; do
  run_one "${ablation}"
done

echo
echo "[ablation] done. Collect with:"
echo "  ${PYTHON} scripts/collect_pems08_ood_graphwavenet_ablation.py --root ${CKPT_ROOT}"
