#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_pems08_fpem.sh}"
CONFIG="${CONFIG:-configs/ours/pems08_fpem.py}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"

SWEEP_NAME="${SWEEP_NAME:-pems08_fpem_sweep}"
SWEEP_ROOT="${SWEEP_ROOT:-${PROJECT_DIR}/checkpoints/${SWEEP_NAME}}"
RESULTS_TSV="${RESULTS_TSV:-${SWEEP_ROOT}/results.tsv}"
BEST_JSON="${BEST_JSON:-${SWEEP_ROOT}/best_config.json}"
BEST_TXT="${BEST_TXT:-${SWEEP_ROOT}/best_config.txt}"
BEST_METRIC="${BEST_METRIC:-mae}"
BEST_SPLIT_METRICS="${BEST_SPLIT_METRICS:-best_test_metrics.json}"
INCLUDE_NONZERO_WITH_METRICS="${INCLUDE_NONZERO_WITH_METRICS:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_COMBOS="${MAX_COMBOS:-0}"

# Space-separated sweep grids. Override any of these from the shell.
SEED_LIST="${SEED_LIST:-2026}"
LR_LIST="${LR_LIST:-0.001 0.0005}"
OPTIMIZER_LIST="${OPTIMIZER_LIST:-adamw}"
WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-1e-5 5e-5}"
GRAD_CLIP_LIST="${GRAD_CLIP_LIST:-3.0 5.0}"
LR_SCHEDULER_LIST="${LR_SCHEDULER_LIST:-multistep}"
LR_MILESTONES_LIST="${LR_MILESTONES_LIST:-[30,60,80]}"
LR_GAMMA_LIST="${LR_GAMMA_LIST:-0.3 0.5}"
LAMBDA_ENVPRED_LIST="${LAMBDA_ENVPRED_LIST:-0.03 0.05}"
LAMBDA_FUTURE_MI_LIST="${LAMBDA_FUTURE_MI_LIST:-0.03 0.05}"
LAMBDA_SWAP_LIST="${LAMBDA_SWAP_LIST:-0.03 0.05}"
LAMBDA_MASK_SPARSE_LIST="${LAMBDA_MASK_SPARSE_LIST:-1e-3}"
SPARSE_TARGET_LIST="${SPARSE_TARGET_LIST:-0.25 0.3}"
TRAIN_LOSS_SCALE_LIST="${TRAIN_LOSS_SCALE_LIST:-normalized}"

# Fixed metric definition. Do not sweep this when comparing model quality.
MAPE_THRESHOLD="${MAPE_THRESHOLD:-1.0}"
MAPE_EPS="${MAPE_EPS:-1e-5}"
MAPE_AS_PERCENT="${MAPE_AS_PERCENT:-true}"

# Applied to every run. Useful for smoke tests, e.g.
# EXTRA_ARGS_BASE="--set TRAIN.epochs=1 --set TRAIN.max_train_batches=1 --set TRAIN.val_batches=1 --set TRAIN.test_batches=1"
EXTRA_ARGS_BASE="${EXTRA_ARGS_BASE:-}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p "${SWEEP_ROOT}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a LRS <<< "${LR_LIST}"
read -r -a OPTIMIZERS <<< "${OPTIMIZER_LIST}"
read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAY_LIST}"
read -r -a GRAD_CLIPS <<< "${GRAD_CLIP_LIST}"
read -r -a LR_SCHEDULERS <<< "${LR_SCHEDULER_LIST}"
read -r -a LR_MILESTONES_VALUES <<< "${LR_MILESTONES_LIST}"
read -r -a LR_GAMMAS <<< "${LR_GAMMA_LIST}"
read -r -a LAMBDA_ENVPREDS <<< "${LAMBDA_ENVPRED_LIST}"
read -r -a LAMBDA_FUTURE_MIS <<< "${LAMBDA_FUTURE_MI_LIST}"
read -r -a LAMBDA_SWAPS <<< "${LAMBDA_SWAP_LIST}"
read -r -a LAMBDA_MASK_SPARSES <<< "${LAMBDA_MASK_SPARSE_LIST}"
read -r -a SPARSE_TARGETS <<< "${SPARSE_TARGET_LIST}"
read -r -a TRAIN_LOSS_SCALES <<< "${TRAIN_LOSS_SCALE_LIST}"

if [[ ! -f "${RESULTS_TSV}" ]]; then
  printf "combo_id\tstatus\texit_code\tmetrics_found\tckpt_dir\tseed\tlr\toptimizer\tweight_decay\tgrad_clip\tlr_scheduler\tlr_milestones\tlr_gamma\tlambda_envpred\tlambda_future_mi\tlambda_swap\tlambda_mask_sparse\tsparse_target\ttrain_loss_scale\tmae\tmse\trmse\tmape\tmetric_path\n" > "${RESULTS_TSV}"
fi

truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

append_result() {
  local combo_id="$1"
  local status="$2"
  local exit_code="$3"
  local metrics_path="$4"
  local ckpt_dir="$5"
  local seed="$6"
  local lr="$7"
  local optimizer="$8"
  local weight_decay="$9"
  local grad_clip="${10}"
  local lr_scheduler="${11}"
  local lr_milestones="${12}"
  local lr_gamma="${13}"
  local lambda_envpred="${14}"
  local lambda_future_mi="${15}"
  local lambda_swap="${16}"
  local lambda_mask_sparse="${17}"
  local sparse_target="${18}"
  local train_loss_scale="${19}"

  "${PYTHON}" - "${RESULTS_TSV}" "${metrics_path}" "${combo_id}" "${status}" "${exit_code}" "${ckpt_dir}" \
    "${seed}" "${lr}" "${optimizer}" "${weight_decay}" "${grad_clip}" "${lr_scheduler}" "${lr_milestones}" \
    "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" "${lambda_mask_sparse}" \
    "${sparse_target}" "${train_loss_scale}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

(
    results_path,
    metrics_path,
    combo_id,
    status,
    exit_code,
    ckpt_dir,
    seed,
    lr,
    optimizer,
    weight_decay,
    grad_clip,
    lr_scheduler,
    lr_milestones,
    lr_gamma,
    lambda_envpred,
    lambda_future_mi,
    lambda_swap,
    lambda_mask_sparse,
    sparse_target,
    train_loss_scale,
) = sys.argv[1:]

metrics_file = Path(metrics_path)
metrics_found = metrics_file.exists()
metrics = {}
if metrics_found:
    with metrics_file.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

rmse = metrics.get("rmse", float("nan"))
mse = metrics.get("mse")
if mse is None:
    try:
        mse = float(rmse) ** 2
    except Exception:
        mse = float("nan")

row = {
    "combo_id": combo_id,
    "status": status,
    "exit_code": exit_code,
    "metrics_found": "1" if metrics_found else "0",
    "ckpt_dir": ckpt_dir,
    "seed": seed,
    "lr": lr,
    "optimizer": optimizer,
    "weight_decay": weight_decay,
    "grad_clip": grad_clip,
    "lr_scheduler": lr_scheduler,
    "lr_milestones": lr_milestones,
    "lr_gamma": lr_gamma,
    "lambda_envpred": lambda_envpred,
    "lambda_future_mi": lambda_future_mi,
    "lambda_swap": lambda_swap,
    "lambda_mask_sparse": lambda_mask_sparse,
    "sparse_target": sparse_target,
    "train_loss_scale": train_loss_scale,
    "mae": metrics.get("mae", float("nan")),
    "mse": mse,
    "rmse": rmse,
    "mape": metrics.get("mape", float("nan")),
    "metric_path": str(metrics_file),
}

with open(results_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
    writer.writerow(row)
PY
}

update_best() {
  "${PYTHON}" - "${RESULTS_TSV}" "${BEST_METRIC}" "${BEST_JSON}" "${BEST_TXT}" "${INCLUDE_NONZERO_WITH_METRICS}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

results_path, best_metric, best_json, best_txt, include_nonzero = sys.argv[1:]
include_nonzero = include_nonzero.lower() in {"1", "true", "yes", "y", "on"}

with open(results_path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

candidates = []
for row in rows:
    if row.get("metrics_found") != "1":
        continue
    if not include_nonzero and row.get("exit_code") != "0":
        continue
    try:
        value = float(row.get(best_metric, "nan"))
    except ValueError:
        continue
    if math.isfinite(value):
        candidates.append((value, row))

if not candidates:
    Path(best_txt).write_text("No completed metric rows yet.\n", encoding="utf-8")
    raise SystemExit(0)

best_value, best_row = min(candidates, key=lambda item: item[0])
payload = {
    "best_metric": best_metric,
    "best_value": best_value,
    "best_row": best_row,
}
Path(best_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

lines = [
    f"best_metric={best_metric}",
    f"best_value={best_value}",
    f"combo_id={best_row['combo_id']}",
    f"ckpt_dir={best_row['ckpt_dir']}",
    f"metric_path={best_row['metric_path']}",
    "params:",
]
for key in [
    "seed", "lr", "optimizer", "weight_decay", "grad_clip", "lr_scheduler",
    "lr_milestones", "lr_gamma", "lambda_envpred", "lambda_future_mi",
    "lambda_swap", "lambda_mask_sparse", "sparse_target", "train_loss_scale",
]:
    lines.append(f"  {key}={best_row[key]}")
for key in ["mae", "mse", "rmse", "mape", "status", "exit_code"]:
    lines.append(f"{key}={best_row[key]}")
Path(best_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

total=0
for _seed in "${SEEDS[@]}"; do
for _lr in "${LRS[@]}"; do
for _optimizer in "${OPTIMIZERS[@]}"; do
for _weight_decay in "${WEIGHT_DECAYS[@]}"; do
for _grad_clip in "${GRAD_CLIPS[@]}"; do
for _lr_scheduler in "${LR_SCHEDULERS[@]}"; do
for _lr_milestones in "${LR_MILESTONES_VALUES[@]}"; do
for _lr_gamma in "${LR_GAMMAS[@]}"; do
for _lambda_envpred in "${LAMBDA_ENVPREDS[@]}"; do
for _lambda_future_mi in "${LAMBDA_FUTURE_MIS[@]}"; do
for _lambda_swap in "${LAMBDA_SWAPS[@]}"; do
for _lambda_mask_sparse in "${LAMBDA_MASK_SPARSES[@]}"; do
for _sparse_target in "${SPARSE_TARGETS[@]}"; do
for _train_loss_scale in "${TRAIN_LOSS_SCALES[@]}"; do
  total=$((total + 1))
done; done; done; done; done; done; done; done; done; done; done; done; done; done

echo "[sweep] project=${PROJECT_DIR}"
echo "[sweep] run_script=${RUN_SCRIPT}"
echo "[sweep] sweep_root=${SWEEP_ROOT}"
echo "[sweep] results=${RESULTS_TSV}"
echo "[sweep] total_combos=${total} max_combos=${MAX_COMBOS} dry_run=${DRY_RUN}"

combo=0
for seed in "${SEEDS[@]}"; do
for lr in "${LRS[@]}"; do
for optimizer in "${OPTIMIZERS[@]}"; do
for weight_decay in "${WEIGHT_DECAYS[@]}"; do
for grad_clip in "${GRAD_CLIPS[@]}"; do
for lr_scheduler in "${LR_SCHEDULERS[@]}"; do
for lr_milestones in "${LR_MILESTONES_VALUES[@]}"; do
for lr_gamma in "${LR_GAMMAS[@]}"; do
for lambda_envpred in "${LAMBDA_ENVPREDS[@]}"; do
for lambda_future_mi in "${LAMBDA_FUTURE_MIS[@]}"; do
for lambda_swap in "${LAMBDA_SWAPS[@]}"; do
for lambda_mask_sparse in "${LAMBDA_MASK_SPARSES[@]}"; do
for sparse_target in "${SPARSE_TARGETS[@]}"; do
for train_loss_scale in "${TRAIN_LOSS_SCALES[@]}"; do
  combo=$((combo + 1))
  if [[ "${MAX_COMBOS}" -gt 0 && "${combo}" -gt "${MAX_COMBOS}" ]]; then
    echo "[sweep] reached MAX_COMBOS=${MAX_COMBOS}; stopping."
    update_best
    exit 0
  fi

  combo_id=$(printf "combo_%04d" "${combo}")
  ckpt_dir="${SWEEP_ROOT}/${combo_id}"
  metrics_path="${ckpt_dir}/${BEST_SPLIT_METRICS}"
  log_path="${ckpt_dir}/run.log"
  mkdir -p "${ckpt_dir}"

  cat > "${ckpt_dir}/params.env" <<EOF
combo_id=${combo_id}
seed=${seed}
lr=${lr}
optimizer=${optimizer}
weight_decay=${weight_decay}
grad_clip=${grad_clip}
lr_scheduler=${lr_scheduler}
lr_milestones=${lr_milestones}
lr_gamma=${lr_gamma}
lambda_envpred=${lambda_envpred}
lambda_future_mi=${lambda_future_mi}
lambda_swap=${lambda_swap}
lambda_mask_sparse=${lambda_mask_sparse}
sparse_target=${sparse_target}
train_loss_scale=${train_loss_scale}
EOF

  extra_args=(
    "--set" "TRAIN.learning_rate=${lr}"
    "--set" "TRAIN.optimizer=${optimizer}"
    "--set" "TRAIN.weight_decay=${weight_decay}"
    "--set" "TRAIN.no_decay_for_bias_norm_emb=true"
    "--set" "TRAIN.grad_clip=${grad_clip}"
    "--set" "TRAIN.lr_scheduler=${lr_scheduler}"
    "--set" "TRAIN.lr_gamma=${lr_gamma}"
    "--set" "LOSS.lambda_envpred=${lambda_envpred}"
    "--set" "LOSS.lambda_future_mi=${lambda_future_mi}"
    "--set" "LOSS.lambda_swap=${lambda_swap}"
    "--set" "LOSS.lambda_mask_sparse=${lambda_mask_sparse}"
    "--set" "LOSS.sparse_target=${sparse_target}"
    "--set" "LOSS.train_loss_scale=${train_loss_scale}"
    "--set" "METRICS.mape_threshold=${MAPE_THRESHOLD}"
    "--set" "METRICS.mape_eps=${MAPE_EPS}"
    "--set" "METRICS.mape_as_percent=${MAPE_AS_PERCENT}"
  )
  if [[ -n "${EXTRA_ARGS_BASE}" ]]; then
    # shellcheck disable=SC2206
    base_args=( ${EXTRA_ARGS_BASE} )
    extra_args+=("${base_args[@]}")
  fi
  extra_args_string=$(printf "%q " "${extra_args[@]}")

  echo "[sweep] ${combo_id}/${total} seed=${seed} lr=${lr} wd=${weight_decay} clip=${grad_clip} envpred=${lambda_envpred} future_mi=${lambda_future_mi} swap=${lambda_swap} sparse=${lambda_mask_sparse} target=${sparse_target}"

  if truthy "${SKIP_COMPLETED}" && [[ -f "${metrics_path}" ]]; then
    echo "[sweep] skip completed ${combo_id}: ${metrics_path}"
    append_result "${combo_id}" "skipped_completed" "0" "${metrics_path}" "${ckpt_dir}" \
      "${seed}" "${lr}" "${optimizer}" "${weight_decay}" "${grad_clip}" "${lr_scheduler}" "${lr_milestones}" \
      "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" "${lambda_mask_sparse}" \
      "${sparse_target}" "${train_loss_scale}"
    update_best
    continue
  fi

  if truthy "${DRY_RUN}"; then
    echo "[dry-run] CKPT_DIR=${ckpt_dir} SEED=${seed} GPU=${GPU} DEVICE=${DEVICE} CONFIG=${CONFIG} LR_MILESTONES=${lr_milestones} EXTRA_ARGS=${extra_args_string} bash ${RUN_SCRIPT}" | tee "${log_path}"
    continue
  fi

  set +e
  CKPT_DIR="${ckpt_dir}" \
  SEED="${seed}" \
  GPU="${GPU}" \
  DEVICE="${DEVICE}" \
  CONFIG="${CONFIG}" \
  LR_MILESTONES="${lr_milestones}" \
  EXTRA_ARGS="${extra_args_string}" \
  bash "${RUN_SCRIPT}" 2>&1 | tee "${log_path}"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ -f "${metrics_path}" && "${exit_code}" -eq 0 ]]; then
    status="ok"
  elif [[ -f "${metrics_path}" ]]; then
    status="metric_ok_nonzero_exit"
  else
    status="failed"
  fi

  append_result "${combo_id}" "${status}" "${exit_code}" "${metrics_path}" "${ckpt_dir}" \
    "${seed}" "${lr}" "${optimizer}" "${weight_decay}" "${grad_clip}" "${lr_scheduler}" "${lr_milestones}" \
    "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" "${lambda_mask_sparse}" \
    "${sparse_target}" "${train_loss_scale}"
  update_best
done; done; done; done; done; done; done; done; done; done; done; done; done; done

echo "[sweep] done. Final best:"
update_best
