#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/mystg/nue_stg_project}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CONFIG="${CONFIG:-configs/ours/pems08/fpem_stid_mlp.py}"

SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS="${EPOCHS:-100}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/pems08/fpem_once_seed${SEED}}"

export CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "${PROJECT_DIR}"

echo "[FPEM once] project=${PROJECT_DIR}"
echo "[FPEM once] config=${CONFIG}"
echo "[FPEM once] seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[FPEM once] device=${DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[FPEM once] ckpt_dir=${CKPT_DIR}"

"${PYTHON}" -m compileall models configs train.py >/dev/null

TRAIN_STATUS=0
set +e
"${PYTHON}" train.py \
  --config_file "${CONFIG}" \
  --set "TRAIN.seed=${SEED}" \
  --set "TRAIN.batch_size=${BATCH_SIZE}" \
  --set "TRAIN.num_workers=${NUM_WORKERS}" \
  --set "TRAIN.device=${DEVICE}" \
  --set "TRAIN.epochs=${EPOCHS}" \
  --set "TRAIN.ckpt_dir=${CKPT_DIR}"
TRAIN_STATUS=$?
set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "[FPEM once] train.py exited with status ${TRAIN_STATUS}" >&2
fi

METRICS_JSON="${CKPT_DIR}/best_test_metrics.json"
if [[ ! -f "${METRICS_JSON}" ]]; then
  echo "[FPEM once] missing metrics: ${METRICS_JSON}" >&2
  if [[ "${TRAIN_STATUS}" -eq 0 ]]; then
    exit 1
  fi
  exit "${TRAIN_STATUS}"
fi

"${PYTHON}" - "${METRICS_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

mae = float(metrics["mae"])
rmse = float(metrics["rmse"])
mse = float(metrics.get("mse", rmse * rmse))
mape = float(metrics["mape"])

print("[FPEM once] best test metrics")
print(f"MAE={mae:.6f}")
print(f"MSE={mse:.6f}")
print(f"RMSE={rmse:.6f}")
print(f"MAPE={mape:.6f}")
print(f"metrics_json={path}")
PY

exit "${TRAIN_STATUS}"
