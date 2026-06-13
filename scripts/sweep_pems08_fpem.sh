#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_pems08_fpem.sh}"
CONFIG="${CONFIG:-configs/ours/pems08_fpem.py}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"

SWEEP_NAME="${SWEEP_NAME:-pems08_fpem_backbone_sweep}"
SWEEP_ROOT="${SWEEP_ROOT:-${PROJECT_DIR}/checkpoints/${SWEEP_NAME}}"
RESULTS_TSV="${RESULTS_TSV:-${SWEEP_ROOT}/results.tsv}"
BEST_JSON="${BEST_JSON:-${SWEEP_ROOT}/best_config.json}"
BEST_TXT="${BEST_TXT:-${SWEEP_ROOT}/best_config.txt}"
BEST_BY_BACKBONE_TSV="${BEST_BY_BACKBONE_TSV:-${SWEEP_ROOT}/best_by_backbone.tsv}"
BEST_METRIC="${BEST_METRIC:-mae}"
BEST_SPLIT_METRICS="${BEST_SPLIT_METRICS:-best_test_metrics.json}"
INCLUDE_NONZERO_WITH_METRICS="${INCLUDE_NONZERO_WITH_METRICS:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_COMBOS="${MAX_COMBOS:-0}"
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME_FROM="${RESUME_FROM:-auto}"
COMPLETION_MARKER="${COMPLETION_MARKER:-run_complete.json}"
BEST_SELECT_SPLIT="${BEST_SELECT_SPLIT:-test}"
BEST_SELECT_METRIC="${BEST_SELECT_METRIC:-mae}"

SWEEP_PROFILE="${SWEEP_PROFILE:-backbone_pruned}"
BACKBONE_PROFILE="${BACKBONE_PROFILE:-official}"
ANALYZE_PREVIOUS="${ANALYZE_PREVIOUS:-true}"
PREVIOUS_RESULTS_TSV="${PREVIOUS_RESULTS_TSV:-${PROJECT_DIR}/checkpoints/pems08_fpem_sweep/results.tsv}"
PREVIOUS_ANALYSIS_TXT="${PREVIOUS_ANALYSIS_TXT:-${SWEEP_ROOT}/previous_sweep_analysis.txt}"
PREVIOUS_TOP_K="${PREVIOUS_TOP_K:-10}"

# Default backbone candidates verified through the FPEM path. Extended
# candidates are documented but intentionally not enabled by default.
BACKBONE_LIST="${BACKBONE_LIST:-stid_mlp graphwavenet agcrn}"
BACKBONE_LIST_EXTENDED="${BACKBONE_LIST_EXTENDED:-graphwavenet_full stid stgcn stnorm}"

# Space-separated sweep grids. Profiles only set defaults; explicit environment
# variables still win.
SEED_LIST="${SEED_LIST:-2026}"
case "${SWEEP_PROFILE}" in
  full)
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
    ;;
  fine)
    LR_LIST="${LR_LIST:-0.001 0.0008 0.0005}"
    OPTIMIZER_LIST="${OPTIMIZER_LIST:-adamw}"
    WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-1e-5}"
    GRAD_CLIP_LIST="${GRAD_CLIP_LIST:-3.0}"
    LR_SCHEDULER_LIST="${LR_SCHEDULER_LIST:-multistep}"
    LR_MILESTONES_LIST="${LR_MILESTONES_LIST:-[30,60,80]}"
    LR_GAMMA_LIST="${LR_GAMMA_LIST:-0.5}"
    LAMBDA_ENVPRED_LIST="${LAMBDA_ENVPRED_LIST:-0.05}"
    LAMBDA_FUTURE_MI_LIST="${LAMBDA_FUTURE_MI_LIST:-0.01 0.02 0.03}"
    LAMBDA_SWAP_LIST="${LAMBDA_SWAP_LIST:-0.05}"
    LAMBDA_MASK_SPARSE_LIST="${LAMBDA_MASK_SPARSE_LIST:-1e-3}"
    SPARSE_TARGET_LIST="${SPARSE_TARGET_LIST:-0.3}"
    TRAIN_LOSS_SCALE_LIST="${TRAIN_LOSS_SCALE_LIST:-normalized}"
    ;;
  backbone_pruned)
    LR_LIST="${LR_LIST:-0.001 0.0005}"
    OPTIMIZER_LIST="${OPTIMIZER_LIST:-adamw}"
    WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-1e-5}"
    GRAD_CLIP_LIST="${GRAD_CLIP_LIST:-3.0}"
    LR_SCHEDULER_LIST="${LR_SCHEDULER_LIST:-multistep}"
    LR_MILESTONES_LIST="${LR_MILESTONES_LIST:-[30,60,80]}"
    LR_GAMMA_LIST="${LR_GAMMA_LIST:-0.5}"
    LAMBDA_ENVPRED_LIST="${LAMBDA_ENVPRED_LIST:-0.05}"
    LAMBDA_FUTURE_MI_LIST="${LAMBDA_FUTURE_MI_LIST:-0.01 0.03}"
    LAMBDA_SWAP_LIST="${LAMBDA_SWAP_LIST:-0.05}"
    LAMBDA_MASK_SPARSE_LIST="${LAMBDA_MASK_SPARSE_LIST:-1e-3}"
    SPARSE_TARGET_LIST="${SPARSE_TARGET_LIST:-0.3}"
    TRAIN_LOSS_SCALE_LIST="${TRAIN_LOSS_SCALE_LIST:-normalized}"
    ;;
  *)
    echo "[sweep] unknown SWEEP_PROFILE=${SWEEP_PROFILE}; expected backbone_pruned, fine, or full" >&2
    exit 2
    ;;
esac

PERTURB_ENABLED="${PERTURB_ENABLED:-false}"
SWAP_DETACH_ENV="${SWAP_DETACH_ENV:-false}"
LAMBDA_Z_CONS_LIST="${LAMBDA_Z_CONS_LIST:-0}"
LAMBDA_Y_CONS_LIST="${LAMBDA_Y_CONS_LIST:-0}"

# Fixed metric definition. Do not sweep this when comparing model quality.
MAPE_THRESHOLD="${MAPE_THRESHOLD:-1.0}"
MAPE_EPS="${MAPE_EPS:-1e-5}"
MAPE_AS_PERCENT="${MAPE_AS_PERCENT:-true}"

# Applied to every run. Useful for smoke tests, e.g.
# EXTRA_ARGS_BASE="--set TRAIN.epochs=1 --set TRAIN.max_train_batches=1 --set TRAIN.val_batches=1 --set TRAIN.test_batches=1"
EXTRA_ARGS_BASE="${EXTRA_ARGS_BASE:-}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p "${SWEEP_ROOT}"

read -r -a BACKBONES <<< "${BACKBONE_LIST}"
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
read -r -a LAMBDA_Z_CONS_VALUES <<< "${LAMBDA_Z_CONS_LIST}"
read -r -a LAMBDA_Y_CONS_VALUES <<< "${LAMBDA_Y_CONS_LIST}"

RESULTS_HEADER="combo_id	status	exit_code	metrics_found	ckpt_dir	backbone	seed	lr	optimizer	weight_decay	grad_clip	dropout	lr_scheduler	lr_milestones	lr_gamma	lambda_envpred	lambda_future_mi	lambda_swap	lambda_mask_sparse	sparse_target	train_loss_scale	perturb_enabled	lambda_z_cons	lambda_y_cons	mae	mse	rmse	mape	wape	metric_path"

truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

backbone_config_key() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    graphwavenet|gwnet|graph_wavenet) printf "graph_wavenet" ;;
    graphwavenet_full|gwnet_full|graph_wavenet_full|graphwavenet-full) printf "graph_wavenet_full" ;;
    stnorm|st_norm|stnorm_wavenet) printf "stnorm_wavenet" ;;
    *) printf "%s" "$1" ;;
  esac
}

profile_combo_enabled() {
  local candidate_backbone="$1"
  local candidate_lr="$2"
  if [[ "${BACKBONE_PROFILE}" != "official" ]]; then
    return 0
  fi
  case "$(printf '%s' "${candidate_backbone}" | tr '[:upper:]' '[:lower:]')" in
    graphwavenet|gwnet|graph_wavenet|graphwavenet_full|gwnet_full|graph_wavenet_full|graphwavenet-full)
      [[ "${candidate_lr}" == "0.001" || "${candidate_lr}" == "1e-3" ]]
      return
      ;;
  esac
  return 0
}

apply_backbone_profile() {
  effective_lr="${lr}"
  effective_weight_decay="${weight_decay}"
  effective_grad_clip="${grad_clip}"
  effective_dropout=""
  case "${BACKBONE_PROFILE}" in
    official)
      case "$(printf '%s' "${backbone}" | tr '[:upper:]' '[:lower:]')" in
        stid_mlp|mlp|stid_like|stid|official_stid)
          effective_lr="${lr}"
          effective_weight_decay="1e-5"
          effective_grad_clip="3.0"
          effective_dropout="0.1"
          ;;
        graphwavenet|gwnet|graph_wavenet|graphwavenet_full|gwnet_full|graph_wavenet_full|graphwavenet-full)
          effective_lr="0.001"
          effective_weight_decay="1e-4"
          effective_grad_clip="5.0"
          effective_dropout="0.3"
          ;;
        agcrn)
          effective_lr="${lr}"
          if [[ "${lr}" == "0.0005" || "${lr}" == "5e-4" ]]; then
            effective_lr="0.003"
          fi
          effective_weight_decay="1e-5"
          effective_grad_clip="5.0"
          effective_dropout="0.1"
          ;;
      esac
      ;;
    default|custom)
      ;;
    *)
      echo "[sweep] unknown BACKBONE_PROFILE=${BACKBONE_PROFILE}; expected official, default, or custom" >&2
      exit 2
      ;;
  esac
}

analyze_previous() {
  if ! truthy "${ANALYZE_PREVIOUS}"; then
    return 0
  fi
  "${PYTHON}" - "${PREVIOUS_RESULTS_TSV}" "${PREVIOUS_ANALYSIS_TXT}" "${PREVIOUS_TOP_K}" <<'PY'
import csv
import math
import sys
from collections import Counter
from pathlib import Path

results_path, out_path, top_k_raw = sys.argv[1:]
top_k = int(top_k_raw)
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
path = Path(results_path)
if not path.exists():
    text = f"Previous results not found: {path}\n"
    out.write_text(text, encoding="utf-8")
    print(text.rstrip())
    raise SystemExit(0)

with path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def finite_float(row, key):
    try:
        value = float(row.get(key, "nan"))
    except Exception:
        value = float("nan")
    return value if math.isfinite(value) else None

candidates = [row for row in rows if finite_float(row, "mae") is not None]
candidates.sort(key=lambda row: finite_float(row, "mae"))
top = candidates[:top_k]
columns = [
    "combo_id", "mae", "rmse", "mape", "lr", "optimizer", "weight_decay",
    "grad_clip", "lr_scheduler", "lr_milestones", "lr_gamma",
    "lambda_envpred", "lambda_future_mi", "lambda_swap",
    "lambda_mask_sparse", "sparse_target", "train_loss_scale",
]
param_keys = [
    "lr", "optimizer", "weight_decay", "grad_clip", "lr_scheduler",
    "lr_milestones", "lr_gamma", "lambda_envpred", "lambda_future_mi",
    "lambda_swap", "lambda_mask_sparse", "sparse_target",
    "train_loss_scale",
]
lines = [
    f"previous_results={path}",
    f"rows={len(rows)} metric_rows={len(candidates)} top_k={len(top)}",
    "",
    f"Top-{len(top)} by mae:",
    ",".join(columns),
]
for row in top:
    lines.append(",".join(str(row.get(col, "")) for col in columns))
lines.extend(["", "Top-k param frequency:"])
for key in param_keys:
    counts = Counter(str(row.get(key, "")) for row in top)
    pieces = [f"{value} -> {count}/{len(top)}" for value, count in counts.most_common()]
    lines.append(f"{key}: " + (", ".join(pieces) if pieces else "n/a"))

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[sweep] wrote previous analysis: {out}")
PY
}

ensure_results_header() {
  if [[ -f "${RESULTS_TSV}" ]]; then
    first_line="$(head -n 1 "${RESULTS_TSV}")"
    if [[ "${first_line}" != "${RESULTS_HEADER}" ]]; then
      backup="${RESULTS_TSV}.pre_backbone.$(date +%Y%m%d%H%M%S)"
      cp "${RESULTS_TSV}" "${backup}"
      printf "%s\n" "${RESULTS_HEADER}" > "${RESULTS_TSV}"
      echo "[sweep] backed up old results schema to ${backup}"
    fi
  else
    printf "%s\n" "${RESULTS_HEADER}" > "${RESULTS_TSV}"
  fi
}

append_result() {
  local combo_id="$1"
  local status="$2"
  local exit_code="$3"
  local metrics_path="$4"
  local ckpt_dir="$5"
  local backbone="$6"
  local seed="$7"
  local lr="$8"
  local optimizer="$9"
  local weight_decay="${10}"
  local grad_clip="${11}"
  local dropout="${12}"
  local lr_scheduler="${13}"
  local lr_milestones="${14}"
  local lr_gamma="${15}"
  local lambda_envpred="${16}"
  local lambda_future_mi="${17}"
  local lambda_swap="${18}"
  local lambda_mask_sparse="${19}"
  local sparse_target="${20}"
  local train_loss_scale="${21}"
  local perturb_enabled="${22}"
  local lambda_z_cons="${23}"
  local lambda_y_cons="${24}"

  "${PYTHON}" - "${RESULTS_TSV}" "${metrics_path}" "${combo_id}" "${status}" "${exit_code}" "${ckpt_dir}" \
    "${backbone}" "${seed}" "${lr}" "${optimizer}" "${weight_decay}" "${grad_clip}" "${dropout}" "${lr_scheduler}" \
    "${lr_milestones}" "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" \
    "${lambda_mask_sparse}" "${sparse_target}" "${train_loss_scale}" "${perturb_enabled}" "${lambda_z_cons}" "${lambda_y_cons}" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    results_path,
    metrics_path,
    combo_id,
    status,
    exit_code,
    ckpt_dir,
    backbone,
    seed,
    lr,
    optimizer,
    weight_decay,
    grad_clip,
    dropout,
    lr_scheduler,
    lr_milestones,
    lr_gamma,
    lambda_envpred,
    lambda_future_mi,
    lambda_swap,
    lambda_mask_sparse,
    sparse_target,
    train_loss_scale,
    perturb_enabled,
    lambda_z_cons,
    lambda_y_cons,
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
    "backbone": backbone,
    "seed": seed,
    "lr": lr,
    "optimizer": optimizer,
    "weight_decay": weight_decay,
    "grad_clip": grad_clip,
    "dropout": dropout,
    "lr_scheduler": lr_scheduler,
    "lr_milestones": lr_milestones,
    "lr_gamma": lr_gamma,
    "lambda_envpred": lambda_envpred,
    "lambda_future_mi": lambda_future_mi,
    "lambda_swap": lambda_swap,
    "lambda_mask_sparse": lambda_mask_sparse,
    "sparse_target": sparse_target,
    "train_loss_scale": train_loss_scale,
    "perturb_enabled": perturb_enabled,
    "lambda_z_cons": lambda_z_cons,
    "lambda_y_cons": lambda_y_cons,
    "mae": metrics.get("mae", float("nan")),
    "mse": mse,
    "rmse": rmse,
    "mape": metrics.get("mape", float("nan")),
    "wape": metrics.get("wape", float("nan")),
    "metric_path": str(metrics_file),
}

with open(results_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
    writer.writerow(row)
PY
}

update_best() {
  "${PYTHON}" - "${RESULTS_TSV}" "${BEST_METRIC}" "${BEST_JSON}" "${BEST_TXT}" "${BEST_BY_BACKBONE_TSV}" "${INCLUDE_NONZERO_WITH_METRICS}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

results_path, best_metric, best_json, best_txt, best_by_backbone, include_nonzero = sys.argv[1:]
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

by_fields = [
    "backbone", "best_metric", "best_value", "combo_id", "ckpt_dir",
    "mae", "mse", "rmse", "mape", "wape", "lr", "weight_decay", "grad_clip", "dropout",
    "lambda_envpred", "lambda_future_mi", "lambda_swap", "sparse_target",
    "perturb_enabled", "lambda_z_cons", "lambda_y_cons",
]
best_by_path = Path(best_by_backbone)
best_by_path.parent.mkdir(parents=True, exist_ok=True)

if not candidates:
    Path(best_txt).write_text("No completed metric rows yet.\n", encoding="utf-8")
    with best_by_path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=by_fields, delimiter="\t").writeheader()
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
    f"backbone={best_row.get('backbone', '')}",
    f"ckpt_dir={best_row['ckpt_dir']}",
    f"metric_path={best_row['metric_path']}",
    "params:",
]
for key in [
    "backbone", "seed", "lr", "optimizer", "weight_decay", "grad_clip",
    "dropout", "lr_scheduler", "lr_milestones", "lr_gamma", "lambda_envpred",
    "lambda_future_mi", "lambda_swap", "lambda_mask_sparse",
    "sparse_target", "train_loss_scale", "perturb_enabled",
    "lambda_z_cons", "lambda_y_cons",
]:
    lines.append(f"  {key}={best_row.get(key, '')}")
for key in ["mae", "mse", "rmse", "mape", "wape", "status", "exit_code"]:
    lines.append(f"{key}={best_row.get(key, '')}")
Path(best_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")

best_by = {}
for value, row in candidates:
    backbone = row.get("backbone", "")
    if backbone not in best_by or value < best_by[backbone][0]:
        best_by[backbone] = (value, row)
with best_by_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=by_fields, delimiter="\t")
    writer.writeheader()
    for backbone in sorted(best_by):
        value, row = best_by[backbone]
        writer.writerow({
            "backbone": backbone,
            "best_metric": best_metric,
            "best_value": value,
            "combo_id": row.get("combo_id", ""),
            "ckpt_dir": row.get("ckpt_dir", ""),
            "mae": row.get("mae", ""),
            "mse": row.get("mse", ""),
            "rmse": row.get("rmse", ""),
            "mape": row.get("mape", ""),
            "wape": row.get("wape", ""),
            "lr": row.get("lr", ""),
            "weight_decay": row.get("weight_decay", ""),
            "grad_clip": row.get("grad_clip", ""),
            "dropout": row.get("dropout", ""),
            "lambda_envpred": row.get("lambda_envpred", ""),
            "lambda_future_mi": row.get("lambda_future_mi", ""),
            "lambda_swap": row.get("lambda_swap", ""),
            "sparse_target": row.get("sparse_target", ""),
            "perturb_enabled": row.get("perturb_enabled", ""),
            "lambda_z_cons": row.get("lambda_z_cons", ""),
            "lambda_y_cons": row.get("lambda_y_cons", ""),
        })
print("\n".join(lines))
PY
}

analyze_previous
ensure_results_header

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
for _lambda_z_cons in "${LAMBDA_Z_CONS_VALUES[@]}"; do
for _lambda_y_cons in "${LAMBDA_Y_CONS_VALUES[@]}"; do
for _backbone in "${BACKBONES[@]}"; do
  if ! profile_combo_enabled "${_backbone}" "${_lr}"; then
    continue
  fi
  total=$((total + 1))
done; done; done; done; done; done; done; done; done; done; done; done; done; done; done; done; done

echo "[sweep] project=${PROJECT_DIR}"
echo "[sweep] run_script=${RUN_SCRIPT}"
echo "[sweep] config=${CONFIG}"
echo "[sweep] sweep_profile=${SWEEP_PROFILE}"
echo "[sweep] backbone_profile=${BACKBONE_PROFILE}"
echo "[sweep] sweep_root=${SWEEP_ROOT}"
echo "[sweep] results=${RESULTS_TSV}"
echo "[sweep] previous_analysis=${PREVIOUS_ANALYSIS_TXT}"
echo "[sweep] backbone_list=${BACKBONE_LIST}"
echo "[sweep] backbone_list_extended=${BACKBONE_LIST_EXTENDED} (not enabled by default)"
echo "[sweep] perturb_enabled=${PERTURB_ENABLED} lambda_z_cons_list=${LAMBDA_Z_CONS_LIST} lambda_y_cons_list=${LAMBDA_Y_CONS_LIST}"
echo "[sweep] auto_resume=${AUTO_RESUME} resume_from=${RESUME_FROM} completion_marker=${COMPLETION_MARKER}"
echo "[sweep] best_select_split=${BEST_SELECT_SPLIT} best_select_metric=${BEST_SELECT_METRIC}"
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
for lambda_z_cons in "${LAMBDA_Z_CONS_VALUES[@]}"; do
for lambda_y_cons in "${LAMBDA_Y_CONS_VALUES[@]}"; do
for backbone in "${BACKBONES[@]}"; do
  if ! profile_combo_enabled "${backbone}" "${lr}"; then
    continue
  fi
  combo=$((combo + 1))
  if [[ "${MAX_COMBOS}" -gt 0 && "${combo}" -gt "${MAX_COMBOS}" ]]; then
    echo "[sweep] reached MAX_COMBOS=${MAX_COMBOS}; stopping."
    update_best
    exit 0
  fi

  combo_short=$(printf "combo_%04d" "${combo}")
  combo_id=$(printf "%s_%s" "${backbone}" "${combo_short}")
  ckpt_dir="${SWEEP_ROOT}/${backbone}/${combo_short}"
  metrics_path="${ckpt_dir}/${BEST_SPLIT_METRICS}"
  complete_path="${ckpt_dir}/${COMPLETION_MARKER}"
  log_path="${ckpt_dir}/run.log"
  apply_backbone_profile
  backbone_cfg_key="$(backbone_config_key "${backbone}")"
  mkdir -p "${ckpt_dir}"

  cat > "${ckpt_dir}/params.env" <<EOF
combo_id=${combo_id}
backbone=${backbone}
backbone_profile=${BACKBONE_PROFILE}
seed=${seed}
lr=${effective_lr}
optimizer=${optimizer}
weight_decay=${effective_weight_decay}
grad_clip=${effective_grad_clip}
dropout=${effective_dropout}
lr_scheduler=${lr_scheduler}
lr_milestones=${lr_milestones}
lr_gamma=${lr_gamma}
lambda_envpred=${lambda_envpred}
lambda_future_mi=${lambda_future_mi}
lambda_swap=${lambda_swap}
lambda_mask_sparse=${lambda_mask_sparse}
sparse_target=${sparse_target}
train_loss_scale=${train_loss_scale}
perturb_enabled=${PERTURB_ENABLED}
swap_detach_env=${SWAP_DETACH_ENV}
lambda_z_cons=${lambda_z_cons}
lambda_y_cons=${lambda_y_cons}
auto_resume=${AUTO_RESUME}
resume_from=${RESUME_FROM}
completion_marker=${COMPLETION_MARKER}
best_select_split=${BEST_SELECT_SPLIT}
best_select_metric=${BEST_SELECT_METRIC}
EOF

  extra_args=(
    "--set" "MODEL.backbone_name=${backbone}"
    "--set" "TRAIN.auto_resume=${AUTO_RESUME}"
    "--set" "TRAIN.resume_from=${RESUME_FROM}"
    "--set" "TRAIN.best_select_split=${BEST_SELECT_SPLIT}"
    "--set" "TRAIN.best_select_metric=${BEST_SELECT_METRIC}"
    "--set" "TRAIN.learning_rate=${effective_lr}"
    "--set" "TRAIN.optimizer=${optimizer}"
    "--set" "TRAIN.weight_decay=${effective_weight_decay}"
    "--set" "TRAIN.no_decay_for_bias_norm_emb=true"
    "--set" "TRAIN.grad_clip=${effective_grad_clip}"
    "--set" "TRAIN.lr_scheduler=${lr_scheduler}"
    "--set" "TRAIN.lr_gamma=${lr_gamma}"
    "--set" "LOSS.lambda_envpred=${lambda_envpred}"
    "--set" "LOSS.lambda_future_mi=${lambda_future_mi}"
    "--set" "LOSS.lambda_swap=${lambda_swap}"
    "--set" "LOSS.lambda_mask_sparse=${lambda_mask_sparse}"
    "--set" "LOSS.sparse_target=${sparse_target}"
    "--set" "LOSS.train_loss_scale=${train_loss_scale}"
    "--set" "MODEL.perturb_enabled=${PERTURB_ENABLED}"
    "--set" "LOSS.swap_detach_env=${SWAP_DETACH_ENV}"
    "--set" "LOSS.lambda_z_cons=${lambda_z_cons}"
    "--set" "LOSS.lambda_y_cons=${lambda_y_cons}"
    "--set" "METRICS.mape_threshold=${MAPE_THRESHOLD}"
    "--set" "METRICS.mape_eps=${MAPE_EPS}"
    "--set" "METRICS.mape_as_percent=${MAPE_AS_PERCENT}"
  )
  if [[ -n "${effective_dropout}" ]]; then
    extra_args+=("--set" "MODEL.backbone.${backbone_cfg_key}.dropout=${effective_dropout}")
  fi
  if [[ -n "${EXTRA_ARGS_BASE}" ]]; then
    # shellcheck disable=SC2206
    base_args=( ${EXTRA_ARGS_BASE} )
    extra_args+=("${base_args[@]}")
  fi
  extra_args_string=$(printf "%q " "${extra_args[@]}")

  echo "[sweep] ${combo_id}/${total} backbone=${backbone} profile=${BACKBONE_PROFILE} seed=${seed} lr=${effective_lr} wd=${effective_weight_decay} clip=${effective_grad_clip} dropout=${effective_dropout:-na} envpred=${lambda_envpred} future_mi=${lambda_future_mi} swap=${lambda_swap} swap_detach_env=${SWAP_DETACH_ENV} sparse=${lambda_mask_sparse} target=${sparse_target} perturb=${PERTURB_ENABLED} z_cons=${lambda_z_cons} y_cons=${lambda_y_cons}"

  if truthy "${SKIP_COMPLETED}" && [[ -f "${complete_path}" ]]; then
    echo "[sweep] skip completed ${combo_id}: ${complete_path}"
    append_result "${combo_id}" "skipped_completed" "0" "${metrics_path}" "${ckpt_dir}" "${backbone}" \
      "${seed}" "${effective_lr}" "${optimizer}" "${effective_weight_decay}" "${effective_grad_clip}" "${effective_dropout}" "${lr_scheduler}" "${lr_milestones}" \
      "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" "${lambda_mask_sparse}" \
      "${sparse_target}" "${train_loss_scale}" "${PERTURB_ENABLED}" "${lambda_z_cons}" "${lambda_y_cons}"
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

  append_result "${combo_id}" "${status}" "${exit_code}" "${metrics_path}" "${ckpt_dir}" "${backbone}" \
    "${seed}" "${effective_lr}" "${optimizer}" "${effective_weight_decay}" "${effective_grad_clip}" "${effective_dropout}" "${lr_scheduler}" "${lr_milestones}" \
    "${lr_gamma}" "${lambda_envpred}" "${lambda_future_mi}" "${lambda_swap}" "${lambda_mask_sparse}" \
    "${sparse_target}" "${train_loss_scale}" "${PERTURB_ENABLED}" "${lambda_z_cons}" "${lambda_y_cons}"
  update_best
done; done; done; done; done; done; done; done; done; done; done; done; done; done; done; done; done

echo "[sweep] done. Final best:"
update_best
