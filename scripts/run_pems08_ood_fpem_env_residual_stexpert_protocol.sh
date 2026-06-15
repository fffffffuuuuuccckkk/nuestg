#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/ours/pems08_ood_fpem_graphwavenet.py}"
SEEDS="${SEEDS:-2023}"
ABLATIONS="${ABLATIONS:-env_residual_grad_surgery env_residual_z_inv_ib_vib}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/checkpoints/pems08_ood_graphwavenet_ablation}"
CKPT_DIR="${CKPT_DIR:-}"
BACKBONE_NAME="${BACKBONE_NAME:-graphwavenet_full}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
OPTIMIZER="${OPTIMIZER:-adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DROPOUT="${DROPOUT:-0.3}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-30}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-3}"

LR_SCHEDULER="${LR_SCHEDULER:-none}"
LR_MILESTONES="${LR_MILESTONES:-[30,60,80]}"
LR_GAMMA="${LR_GAMMA:-0.5}"
TRAIN_LOSS_SCALE="${TRAIN_LOSS_SCALE:-original}"
MASK_VALUE_MODE="${MASK_VALUE_MODE:-stexpert_min}"
MAPE_THRESHOLD="${MAPE_THRESHOLD:-0.0}"
MAPE_EPS="${MAPE_EPS:-1e-5}"
MAPE_AS_PERCENT="${MAPE_AS_PERCENT:-false}"
SCALER_MEAN="${SCALER_MEAN:-229.78659010357995}"
SCALER_STD="${SCALER_STD:-145.61977053909104}"

IN_DIM="${IN_DIM:-3}"
USE_TIME_OF_DAY_CHANNEL="${USE_TIME_OF_DAY_CHANNEL:-true}"
USE_DAY_OF_WEEK_CHANNEL="${USE_DAY_OF_WEEK_CHANNEL:-true}"
ENGINE_PAD_INPUT="${ENGINE_PAD_INPUT:-false}"
ADDAPTADJ="${ADDAPTADJ:-true}"
RANDOMADJ="${RANDOMADJ:-true}"
USE_STATIC_ADJ="${USE_STATIC_ADJ:-true}"
SUPPORTS_LEN="${SUPPORTS_LEN:-2}"
ADJTYPE="${ADJTYPE:-doubletransition}"
ADJ_PATH="${ADJ_PATH:-/data/OuXiaoyu/mystg/datasets/PEMS08-OOD/adj_mx.pkl}"

ENV_FUSION_SCALE="${ENV_FUSION_SCALE:-0.1}"
FUSION_ZERO_INIT="${FUSION_ZERO_INIT:-true}"
LAMBDA_ENVPRED="${LAMBDA_ENVPRED:-0.05}"
LAMBDA_FUTURE_MI="${LAMBDA_FUTURE_MI:-0.01}"
LAMBDA_SWAP="${LAMBDA_SWAP:-0.05}"
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"
SWAP_FREEZE_PREDICTOR="${SWAP_FREEZE_PREDICTOR:-true}"
LAMBDA_MASK_SPARSE="${LAMBDA_MASK_SPARSE:-1e-3}"
SPARSE_TARGET="${SPARSE_TARGET:-0.3}"

GRAD_SURGERY_METHOD="${GRAD_SURGERY_METHOD:-pcgrad}"
Z_INV_IB_BETA="${Z_INV_IB_BETA:-1e-4}"
Z_INV_IB_NOISE_STD="${Z_INV_IB_NOISE_STD:-0.05}"
Z_INV_IB_KL_FREE_BITS="${Z_INV_IB_KL_FREE_BITS:-0.0}"
Z_INV_IB_PREDICT_FROM_SAMPLED_Z="${Z_INV_IB_PREDICT_FROM_SAMPLED_Z:-true}"

DROP_LAST_TRAIN="${DROP_LAST_TRAIN:-true}"
DROP_LAST_VAL="${DROP_LAST_VAL:-true}"
DROP_LAST_TEST="${DROP_LAST_TEST:-true}"
VAL_BATCHES="${VAL_BATCHES:--1}"
TEST_BATCHES="${TEST_BATCHES:--1}"
CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-false}"
NO_DECAY_FOR_BIAS_NORM_EMB="${NO_DECAY_FOR_BIAS_NORM_EMB:-false}"
EVAL_TEST_ON_BEST="${EVAL_TEST_ON_BEST:-true}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"

AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"

seed_count="$(awk '{print NF}' <<< "${SEEDS}")"
ablation_count="$(awk '{print NF}' <<< "${ABLATIONS}")"

echo "[FPEM-EnvResidual-STExpertProtocol] project=${PROJECT_DIR}"
echo "[FPEM-EnvResidual-STExpertProtocol] config=${CONFIG}"
echo "[FPEM-EnvResidual-STExpertProtocol] ablations=${ABLATIONS}"
echo "[FPEM-EnvResidual-STExpertProtocol] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM-EnvResidual-STExpertProtocol] backbone=${BACKBONE_NAME} in_dim=${IN_DIM} tod=${USE_TIME_OF_DAY_CHANNEL} dow=${USE_DAY_OF_WEEK_CHANNEL} engine_pad=${ENGINE_PAD_INPUT}"
echo "[FPEM-EnvResidual-STExpertProtocol] supports=${ADJTYPE}/${SUPPORTS_LEN} adj_path=${ADJ_PATH} addaptadj=${ADDAPTADJ} randomadj=${RANDOMADJ} static_adj=${USE_STATIC_ADJ}"
echo "[FPEM-EnvResidual-STExpertProtocol] fusion=env_residual env_fusion_scale=${ENV_FUSION_SCALE} zero_init=${FUSION_ZERO_INIT}"
echo "[FPEM-EnvResidual-STExpertProtocol] batch_size=${BATCH_SIZE} lr=${LR} optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[FPEM-EnvResidual-STExpertProtocol] epochs=${EPOCHS} patience=${EARLY_STOP_PATIENCE} scheduler=${LR_SCHEDULER} torch_threads=${TORCH_NUM_THREADS}"
echo "[FPEM-EnvResidual-STExpertProtocol] loss_scale=${TRAIN_LOSS_SCALE} mask_value_mode=${MASK_VALUE_MODE} mape_threshold=${MAPE_THRESHOLD} mape_as_percent=${MAPE_AS_PERCENT}"
echo "[FPEM-EnvResidual-STExpertProtocol] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC} eval_test_on_best=${EVAL_TEST_ON_BEST}"
echo "[FPEM-EnvResidual-STExpertProtocol] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"

ablation_settings() {
  local ablation="$1"
  grad_surgery_enabled="false"
  z_inv_ib_enabled="false"
  z_inv_ib_type="vib"
  case "${ablation}" in
    env_residual_grad_surgery|grad_surgery)
      grad_surgery_enabled="true"
      run_ablation="env_residual_grad_surgery"
      ;;
    env_residual_z_inv_ib_vib|z_inv_ib_vib)
      z_inv_ib_enabled="true"
      z_inv_ib_type="vib"
      run_ablation="env_residual_z_inv_ib_vib"
      ;;
    *)
      echo "[FPEM-EnvResidual-STExpertProtocol] unsupported ablation=${ablation}" >&2
      return 2
      ;;
  esac
}

write_metadata() {
  local path="$1"
  local exit_code="$2"
  local seed="$3"
  local ablation="$4"
  ALIGNMENT_METADATA_PATH="${path}/alignment_metadata.json" \
  EXIT_CODE="${exit_code}" \
  SEED_VALUE="${seed}" \
  ABLATION_VALUE="${ablation}" \
  ENV_FUSION_SCALE_VALUE="${ENV_FUSION_SCALE}" \
  FUSION_ZERO_INIT_VALUE="${FUSION_ZERO_INIT}" \
  IN_DIM_VALUE="${IN_DIM}" \
  USE_TOD_VALUE="${USE_TIME_OF_DAY_CHANNEL}" \
  USE_DOW_VALUE="${USE_DAY_OF_WEEK_CHANNEL}" \
  ADDAPTADJ_VALUE="${ADDAPTADJ}" \
  RANDOMADJ_VALUE="${RANDOMADJ}" \
  USE_STATIC_ADJ_VALUE="${USE_STATIC_ADJ}" \
  SUPPORTS_LEN_VALUE="${SUPPORTS_LEN}" \
  ADJTYPE_VALUE="${ADJTYPE}" \
  ADJ_PATH_VALUE="${ADJ_PATH}" \
  LAMBDA_ENVPRED_VALUE="${LAMBDA_ENVPRED}" \
  LAMBDA_FUTURE_MI_VALUE="${LAMBDA_FUTURE_MI}" \
  LAMBDA_SWAP_VALUE="${LAMBDA_SWAP}" \
  LAMBDA_MASK_SPARSE_VALUE="${LAMBDA_MASK_SPARSE}" \
  GRAD_SURGERY_ENABLED_VALUE="${grad_surgery_enabled}" \
  GRAD_SURGERY_METHOD_VALUE="${GRAD_SURGERY_METHOD}" \
  Z_INV_IB_ENABLED_VALUE="${z_inv_ib_enabled}" \
  Z_INV_IB_TYPE_VALUE="${z_inv_ib_type}" \
  Z_INV_IB_BETA_VALUE="${Z_INV_IB_BETA}" \
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

def as_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "on"}

path = Path(os.environ["ALIGNMENT_METADATA_PATH"])
metadata = {
    "protocol": "fpem_env_residual_with_stexpert_training_protocol",
    "ablation": os.environ["ABLATION_VALUE"],
    "fusion": {
        "fusion_type": "env_residual",
        "env_fusion_scale": float(os.environ["ENV_FUSION_SCALE_VALUE"]),
        "fusion_zero_init": as_bool(os.environ["FUSION_ZERO_INIT_VALUE"]),
    },
    "input": {
        "in_dim": int(os.environ["IN_DIM_VALUE"]),
        "use_time_of_day": as_bool(os.environ["USE_TOD_VALUE"]),
        "use_day_of_week": as_bool(os.environ["USE_DOW_VALUE"]),
        "semantics": "traffic_plus_time_of_day_plus_day_of_week",
    },
    "adjacency": {
        "type": os.environ["ADJTYPE_VALUE"],
        "supports_len": int(os.environ["SUPPORTS_LEN_VALUE"]),
        "adj_path": os.environ["ADJ_PATH_VALUE"],
        "addaptadj": as_bool(os.environ["ADDAPTADJ_VALUE"]),
        "randomadj": as_bool(os.environ["RANDOMADJ_VALUE"]),
        "use_static_adj": as_bool(os.environ["USE_STATIC_ADJ_VALUE"]),
        "semantics": "pure_graphwavenet_adaptive_adjacency_not_stexpert_expert_graph",
        "stexpert_expert_graph_equivalent": False,
    },
    "loss_ablation": {
        "lambda_envpred": float(os.environ["LAMBDA_ENVPRED_VALUE"]),
        "lambda_future_mi": float(os.environ["LAMBDA_FUTURE_MI_VALUE"]),
        "lambda_swap": float(os.environ["LAMBDA_SWAP_VALUE"]),
        "lambda_mask_sparse": float(os.environ["LAMBDA_MASK_SPARSE_VALUE"]),
    },
    "grad_surgery": {
        "enabled": as_bool(os.environ["GRAD_SURGERY_ENABLED_VALUE"]),
        "method": os.environ["GRAD_SURGERY_METHOD_VALUE"],
        "apply_to": "inv_encoder",
    },
    "z_inv_bottleneck": {
        "enabled": as_bool(os.environ["Z_INV_IB_ENABLED_VALUE"]),
        "type": os.environ["Z_INV_IB_TYPE_VALUE"],
        "beta": float(os.environ["Z_INV_IB_BETA_VALUE"]),
    },
    "seed": int(os.environ["SEED_VALUE"]),
    "exit_code": int(os.environ["EXIT_CODE"]),
}
path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
PY
}

for ablation in ${ABLATIONS}; do
  ablation_settings "${ablation}"
  for seed in ${SEEDS}; do
    export PYTHONHASHSEED="${seed}"
    if [[ -n "${CKPT_DIR}" && "${ablation_count}" -eq 1 && "${seed_count}" -eq 1 ]]; then
      run_ckpt_dir="${CKPT_DIR}"
    elif [[ -n "${CKPT_DIR}" ]]; then
      run_ckpt_dir="${CKPT_DIR}/${run_ablation}"
      if [[ "${seed_count}" -gt 1 ]]; then
        run_ckpt_dir="${run_ckpt_dir}/seed${seed}"
      fi
    else
      run_ckpt_dir="${CKPT_ROOT}/fpem_${run_ablation}_stexpert_protocol"
      if [[ "${seed_count}" -gt 1 ]]; then
        run_ckpt_dir="${run_ckpt_dir}/seed${seed}"
      fi
    fi

    if [[ "${SKIP_COMPLETED}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ && -f "${run_ckpt_dir}/${COMPLETION_MARKER}" ]]; then
      echo "[FPEM-EnvResidual-STExpertProtocol] skip completed ablation=${run_ablation} seed=${seed}: ${run_ckpt_dir}/${COMPLETION_MARKER}"
      continue
    fi

    echo "[FPEM-EnvResidual-STExpertProtocol] run ablation=${run_ablation} seed=${seed} ckpt_dir=${run_ckpt_dir}"
    echo "[FPEM-EnvResidual-STExpertProtocol] grad_surgery=${grad_surgery_enabled} z_inv_ib=${z_inv_ib_enabled}/${z_inv_ib_type}"

    set +e
    "${PYTHON}" train.py \
      --config "${CONFIG}" \
      --set "RUN.method=FPEM-EnvResidual-GraphWaveNet-STExpertTrainingProtocol-OOD" \
      --set "RUN.ablation=${run_ablation}" \
      --set "RUN.reference_status=fpem_env_residual_stexpert_training_protocol_no_expert_graph" \
      --set "RUN.adjacency_semantics=pure_graphwavenet_adaptive_adjacency_not_stexpert_expert_graph" \
      --set "RUN.input_semantics=traffic_plus_time_of_day_plus_day_of_week" \
      --set "MODEL.reference_status=fpem_env_residual_stexpert_training_protocol_no_expert_graph" \
      --set "MODEL.backbone_name=${BACKBONE_NAME}" \
      --set "MODEL.backbone.name=${BACKBONE_NAME}" \
      --set "MODEL.fusion_type=env_residual" \
      --set "MODEL.env_fusion_scale=${ENV_FUSION_SCALE}" \
      --set "MODEL.fusion_zero_init=${FUSION_ZERO_INIT}" \
      --set "MODEL.backbone.graph_wavenet_full.dropout=${DROPOUT}" \
      --set "MODEL.backbone.graph_wavenet_full.blocks=4" \
      --set "MODEL.backbone.graph_wavenet_full.layers=2" \
      --set "MODEL.backbone.graph_wavenet_full.kernel_size=2" \
      --set "MODEL.backbone.graph_wavenet_full.residual_channels=32" \
      --set "MODEL.backbone.graph_wavenet_full.dilation_channels=32" \
      --set "MODEL.backbone.graph_wavenet_full.skip_channels=256" \
      --set "MODEL.backbone.graph_wavenet_full.end_channels=512" \
      --set "MODEL.backbone.graph_wavenet_full.gcn_bool=true" \
      --set "MODEL.backbone.graph_wavenet_full.addaptadj=${ADDAPTADJ}" \
      --set "MODEL.backbone.graph_wavenet_full.randomadj=${RANDOMADJ}" \
      --set "MODEL.backbone.graph_wavenet_full.use_static_adj=${USE_STATIC_ADJ}" \
      --set "MODEL.backbone.graph_wavenet_full.supports_len=${SUPPORTS_LEN}" \
      --set "MODEL.backbone.graph_wavenet_full.adj_path=${ADJ_PATH}" \
      --set "MODEL.backbone.graph_wavenet_full.adjtype=${ADJTYPE}" \
      --set "MODEL.backbone.graph_wavenet_full.support_add_self_loop=false" \
      --set "MODEL.backbone.graph_wavenet_full.aptonly=false" \
      --set "MODEL.backbone.graph_wavenet_full.in_dim=${IN_DIM}" \
      --set "MODEL.backbone.graph_wavenet_full.engine_pad_input=${ENGINE_PAD_INPUT}" \
      --set "MODEL.backbone.graph_wavenet_full.use_time_of_day_channel=${USE_TIME_OF_DAY_CHANNEL}" \
      --set "MODEL.backbone.graph_wavenet_full.use_day_of_week_channel=${USE_DAY_OF_WEEK_CHANNEL}" \
      --set "TRAIN.seed=${seed}" \
      --set "TRAIN.device=${DEVICE}" \
      --set "TRAIN.ckpt_dir=${run_ckpt_dir}" \
      --set "TRAIN.auto_resume=${AUTO_RESUME}" \
      --set "TRAIN.resume_from=${RESUME_FROM}" \
      --set "TRAIN.best_select_split=${BEST_SELECT_SPLIT}" \
      --set "TRAIN.best_select_metric=${BEST_SELECT_METRIC}" \
      --set "TRAIN.eval_test_on_best=${EVAL_TEST_ON_BEST}" \
      --set "TRAIN.batch_size=${BATCH_SIZE}" \
      --set "TRAIN.epochs=${EPOCHS}" \
      --set "TRAIN.early_stop_patience=${EARLY_STOP_PATIENCE}" \
      --set "TRAIN.learning_rate=${LR}" \
      --set "TRAIN.optimizer=${OPTIMIZER}" \
      --set "TRAIN.weight_decay=${WEIGHT_DECAY}" \
      --set "TRAIN.no_decay_for_bias_norm_emb=${NO_DECAY_FOR_BIAS_NORM_EMB}" \
      --set "TRAIN.grad_clip=${GRAD_CLIP}" \
      --set "TRAIN.lr_scheduler=${LR_SCHEDULER}" \
      --set "TRAIN.lr_milestones=${LR_MILESTONES}" \
      --set "TRAIN.lr_gamma=${LR_GAMMA}" \
      --set "TRAIN.torch_num_threads=${TORCH_NUM_THREADS}" \
      --set "TRAIN.drop_last_train=${DROP_LAST_TRAIN}" \
      --set "TRAIN.drop_last_val=${DROP_LAST_VAL}" \
      --set "TRAIN.drop_last_test=${DROP_LAST_TEST}" \
      --set "TRAIN.val_batches=${VAL_BATCHES}" \
      --set "TRAIN.test_batches=${TEST_BATCHES}" \
      --set "TRAIN.curriculum_enabled=${CURRICULUM_ENABLED}" \
      --set "TRAIN.save_completion_marker=true" \
      --set "LOSS.lambda_envpred=${LAMBDA_ENVPRED}" \
      --set "LOSS.lambda_future_mi=${LAMBDA_FUTURE_MI}" \
      --set "LOSS.lambda_swap=${LAMBDA_SWAP}" \
      --set "LOSS.swap_detach_env=${SWAP_DETACH_ENV}" \
      --set "SWAP.freeze_predictor=${SWAP_FREEZE_PREDICTOR}" \
      --set "LOSS.lambda_mask_sparse=${LAMBDA_MASK_SPARSE}" \
      --set "LOSS.sparse_target=${SPARSE_TARGET}" \
      --set "LOSS.train_loss_scale=${TRAIN_LOSS_SCALE}" \
      --set "LOSS.mask_value_mode=${MASK_VALUE_MODE}" \
      --set "LOSS.grad_surgery.enabled=${grad_surgery_enabled}" \
      --set "LOSS.grad_surgery.method=${GRAD_SURGERY_METHOD}" \
      --set "LOSS.grad_surgery.apply_to=inv_encoder" \
      --set "LOSS.z_inv_bottleneck.enabled=${z_inv_ib_enabled}" \
      --set "LOSS.z_inv_bottleneck.type=${z_inv_ib_type}" \
      --set "LOSS.z_inv_bottleneck.beta=${Z_INV_IB_BETA}" \
      --set "LOSS.z_inv_bottleneck.noise_std=${Z_INV_IB_NOISE_STD}" \
      --set "LOSS.z_inv_bottleneck.kl_free_bits=${Z_INV_IB_KL_FREE_BITS}" \
      --set "LOSS.z_inv_bottleneck.apply_to=z_inv" \
      --set "LOSS.z_inv_bottleneck.predict_from_sampled_z=${Z_INV_IB_PREDICT_FROM_SAMPLED_Z}" \
      --set "METRICS.mape_threshold=${MAPE_THRESHOLD}" \
      --set "METRICS.mape_eps=${MAPE_EPS}" \
      --set "METRICS.mape_as_percent=${MAPE_AS_PERCENT}" \
      --set "EVAL.metric_aggregation=concat" \
      --set "EVAL.save_test_diagnostics=true" \
      --set "SCALER.enabled=true" \
      --set "SCALER.type=zscore" \
      --set "SCALER.norm_each_channel=false" \
      --set "SCALER.mean=${SCALER_MEAN}" \
      --set "SCALER.std=${SCALER_STD}" \
      ${EXTRA_ARGS}
    exit_code=$?
    set -e

    mkdir -p "${run_ckpt_dir}"
    write_metadata "${run_ckpt_dir}" "${exit_code}" "${seed}" "${run_ablation}"
    if [[ "${exit_code}" -eq 0 ]]; then
      rm -f "${run_ckpt_dir}/last_exit_code.txt"
      if [[ ! -f "${run_ckpt_dir}/run_complete.json" ]]; then
        "${PYTHON}" - <<PY
import json
from pathlib import Path
path = Path("${run_ckpt_dir}") / "run_complete.json"
path.write_text(json.dumps({"status": "complete", "seed": int("${seed}"), "ablation": "${run_ablation}"}, indent=2), encoding="utf-8")
PY
      fi
    else
      printf "exit_code=%s\n" "${exit_code}" > "${run_ckpt_dir}/last_exit_code.txt"
      echo "[FPEM-EnvResidual-STExpertProtocol] ablation=${run_ablation} seed=${seed} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
      if [[ ! "${CONTINUE_ON_FAILURE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ ]]; then
        exit "${exit_code}"
      fi
    fi
  done
done
