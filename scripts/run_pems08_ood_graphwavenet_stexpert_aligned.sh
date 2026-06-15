#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/baselines/pems08_ood/graphwavenet.py}"
SEEDS="${SEEDS:-2023}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/pems08_ood_graphwavenet_ablation/stexpert_aligned_pure_graphwavenet}"
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
PROTOCOL_NAME="${PROTOCOL_NAME:-pure_gwnet_backbone_only_stexpert_training_protocol}"
ADJACENCY_SEMANTICS="${ADJACENCY_SEMANTICS:-pure_graphwavenet_adaptive_adjacency_not_stexpert_expert_graph}"
STEXPERT_EXPERT_GRAPH_EQUIVALENT="${STEXPERT_EXPERT_GRAPH_EQUIVALENT:-false}"

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

echo "[PureGraphWaveNet-STExpertAligned] project=${PROJECT_DIR}"
echo "[PureGraphWaveNet-STExpertAligned] config=${CONFIG}"
echo "[PureGraphWaveNet-STExpertAligned] seeds=${SEEDS} device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[PureGraphWaveNet-STExpertAligned] ckpt_dir=${CKPT_DIR}"
echo "[PureGraphWaveNet-STExpertAligned] backbone=${BACKBONE_NAME} in_dim=${IN_DIM} tod=${USE_TIME_OF_DAY_CHANNEL} dow=${USE_DAY_OF_WEEK_CHANNEL} engine_pad=${ENGINE_PAD_INPUT}"
echo "[PureGraphWaveNet-STExpertAligned] supports=${ADJTYPE}/${SUPPORTS_LEN} addaptadj=${ADDAPTADJ} randomadj=${RANDOMADJ} static_adj=${USE_STATIC_ADJ}"
echo "[PureGraphWaveNet-STExpertAligned] adj_path=${ADJ_PATH}"
echo "[PureGraphWaveNet-STExpertAligned] protocol=${PROTOCOL_NAME} adjacency_semantics=${ADJACENCY_SEMANTICS} stexpert_expert_graph_equivalent=${STEXPERT_EXPERT_GRAPH_EQUIVALENT}"
echo "[PureGraphWaveNet-STExpertAligned] batch_size=${BATCH_SIZE} lr=${LR} optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} clip=${GRAD_CLIP} dropout=${DROPOUT}"
echo "[PureGraphWaveNet-STExpertAligned] epochs=${EPOCHS} patience=${EARLY_STOP_PATIENCE} scheduler=${LR_SCHEDULER} torch_threads=${TORCH_NUM_THREADS}"
echo "[PureGraphWaveNet-STExpertAligned] loss_scale=${TRAIN_LOSS_SCALE} mask_value_mode=${MASK_VALUE_MODE} mape_threshold=${MAPE_THRESHOLD} mape_as_percent=${MAPE_AS_PERCENT}"
echo "[PureGraphWaveNet-STExpertAligned] scaler_mean=${SCALER_MEAN} scaler_std=${SCALER_STD}"
echo "[PureGraphWaveNet-STExpertAligned] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC} eval_test_on_best=${EVAL_TEST_ON_BEST}"
echo "[PureGraphWaveNet-STExpertAligned] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} skip_completed=${SKIP_COMPLETED} completion_marker=${COMPLETION_MARKER} continue_on_failure=${CONTINUE_ON_FAILURE}"

for seed in ${SEEDS}; do
  export PYTHONHASHSEED="${seed}"
  if [[ "${seed_count}" -gt 1 ]]; then
    run_ckpt_dir="${CKPT_DIR}/seed${seed}"
  else
    run_ckpt_dir="${CKPT_DIR}"
  fi
  if [[ "${SKIP_COMPLETED}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ && -f "${run_ckpt_dir}/${COMPLETION_MARKER}" ]]; then
    echo "[PureGraphWaveNet-STExpertAligned] skip completed seed=${seed}: ${run_ckpt_dir}/${COMPLETION_MARKER}"
    continue
  fi
  echo "[PureGraphWaveNet-STExpertAligned] run seed=${seed} ckpt_dir=${run_ckpt_dir}"

  set +e
  "${PYTHON}" train.py \
    --config "${CONFIG}" \
    --set "RUN.method=PureGraphWaveNet-BackboneOnly-STExpertTrainingProtocol-OOD" \
    --set "RUN.ablation=stexpert_aligned_pure_graphwavenet" \
    --set "RUN.reference_status=${PROTOCOL_NAME}" \
    --set "RUN.adjacency_semantics=${ADJACENCY_SEMANTICS}" \
    --set "RUN.stexpert_expert_graph_equivalent=${STEXPERT_EXPERT_GRAPH_EQUIVALENT}" \
    --set "RUN.input_semantics=traffic_plus_time_of_day_plus_day_of_week" \
    --set "MODEL.method_variant=backbone_only" \
    --set "MODEL.reference_status=${PROTOCOL_NAME}" \
    --set "MODEL.adjacency_semantics=${ADJACENCY_SEMANTICS}" \
    --set "MODEL.stexpert_expert_graph_equivalent=${STEXPERT_EXPERT_GRAPH_EQUIVALENT}" \
    --set "MODEL.backbone_name=${BACKBONE_NAME}" \
    --set "MODEL.backbone.name=${BACKBONE_NAME}" \
    --set "MODEL.use_timestamp=true" \
    --set "MODEL.required_timestamp=false" \
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
    --set "LOSS.train_loss_scale=${TRAIN_LOSS_SCALE}" \
    --set "LOSS.mask_value_mode=${MASK_VALUE_MODE}" \
    --set "LOSS.lambda_pred=1.0" \
    --set "LOSS.use_inv=false" \
    --set "LOSS.lambda_inv=0.0" \
    --set "LOSS.use_gate=false" \
    --set "LOSS.lambda_gate=0.0" \
    --set "LOSS.use_swap=false" \
    --set "LOSS.lambda_swap=0.0" \
    --set "LOSS.use_kl=false" \
    --set "LOSS.lambda_kl=0.0" \
    --set "LOSS.use_ind=false" \
    --set "LOSS.lambda_ind=0.0" \
    --set "LOSS.use_sparse=false" \
    --set "LOSS.lambda_sparse=0.0" \
    --set "LOSS.use_envpred=false" \
    --set "LOSS.lambda_envpred=0.0" \
    --set "LOSS.use_future_mi=false" \
    --set "LOSS.lambda_future_mi=0.0" \
    --set "LOSS.use_mask_sparse=false" \
    --set "LOSS.lambda_mask_sparse=0.0" \
    --set "SWAP.enabled=false" \
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
  ALIGNMENT_METADATA_PATH="${run_ckpt_dir}/alignment_metadata.json" \
  EXIT_CODE="${exit_code}" \
  SEED_VALUE="${seed}" \
  IN_DIM_VALUE="${IN_DIM}" \
  USE_TOD_VALUE="${USE_TIME_OF_DAY_CHANNEL}" \
  USE_DOW_VALUE="${USE_DAY_OF_WEEK_CHANNEL}" \
  ADDAPTADJ_VALUE="${ADDAPTADJ}" \
  RANDOMADJ_VALUE="${RANDOMADJ}" \
  USE_STATIC_ADJ_VALUE="${USE_STATIC_ADJ}" \
  SUPPORTS_LEN_VALUE="${SUPPORTS_LEN}" \
  ADJTYPE_VALUE="${ADJTYPE}" \
  ADJ_PATH_VALUE="${ADJ_PATH}" \
  PROTOCOL_NAME_VALUE="${PROTOCOL_NAME}" \
  ADJACENCY_SEMANTICS_VALUE="${ADJACENCY_SEMANTICS}" \
  STEXPERT_EXPERT_GRAPH_EQUIVALENT_VALUE="${STEXPERT_EXPERT_GRAPH_EQUIVALENT}" \
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

def as_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "on"}

path = Path(os.environ["ALIGNMENT_METADATA_PATH"])
metadata = {
    "protocol": os.environ["PROTOCOL_NAME_VALUE"],
    "ablation": "stexpert_aligned_pure_graphwavenet",
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
        "semantics": os.environ["ADJACENCY_SEMANTICS_VALUE"],
        "stexpert_expert_graph_equivalent": as_bool(os.environ["STEXPERT_EXPERT_GRAPH_EQUIVALENT_VALUE"]),
        "note": (
            "This run uses the local pure GraphWaveNet backbone adaptive adjacency. "
            "It does not implement ST-Expert expert_embedding/gumbel expert graph selection."
        ),
    },
    "seed": int(os.environ["SEED_VALUE"]),
    "exit_code": int(os.environ["EXIT_CODE"]),
}
path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
PY
  if [[ "${exit_code}" -eq 0 ]]; then
    rm -f "${run_ckpt_dir}/last_exit_code.txt"
    if [[ ! -f "${run_ckpt_dir}/run_complete.json" ]]; then
      "${PYTHON}" - <<PY
import json
from pathlib import Path
path = Path("${run_ckpt_dir}") / "run_complete.json"
path.write_text(json.dumps({"status": "complete", "seed": int("${seed}")}, indent=2), encoding="utf-8")
PY
    fi
  else
    printf "exit_code=%s\n" "${exit_code}" > "${run_ckpt_dir}/last_exit_code.txt"
    echo "[PureGraphWaveNet-STExpertAligned] seed=${seed} failed with exit_code=${exit_code}; rerun the same script to auto-resume." >&2
    if [[ ! "${CONTINUE_ON_FAILURE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|y|Y|on|ON)$ ]]; then
      exit "${exit_code}"
    fi
  fi
done
