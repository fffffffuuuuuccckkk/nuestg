from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "dataset",
    "split",
    "model_name",
    "reference_status",
    "seed",
    "mae",
    "rmse",
    "mape",
    "h3_mae",
    "h6_mae",
    "h12_mae",
    "train_time",
    "test_time",
    "config_path",
    "checkpoint_path",
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reference_status_from_config_path(config_path: str) -> str:
    if not config_path:
        return ""
    try:
        from utils.config_utils import load_config

        cfg = load_config(config_path)
        return str(cfg.get("MODEL", {}).get("reference_status") or cfg.get("RUN", {}).get("reference_status") or "")
    except Exception:
        return ""


def _reference_status_from_resolved(metrics_path: Path) -> str:
    resolved = metrics_path.parent / "resolved_config.json"
    if not resolved.exists():
        return ""
    try:
        cfg = _load_json(resolved)
    except Exception:
        return ""
    return str(cfg.get("MODEL", {}).get("reference_status") or cfg.get("RUN", {}).get("reference_status") or "")


def _row_from_metrics(path: Path) -> Dict:
    payload = _load_json(path)
    config_path = str(payload.get("config_path", ""))
    reference_status = (
        payload.get("reference_status")
        or _reference_status_from_resolved(path)
        or _reference_status_from_config_path(config_path)
    )
    return {
        "dataset": payload.get("dataset", ""),
        "split": payload.get("split", ""),
        "model_name": payload.get("method", payload.get("model_name", "")),
        "reference_status": reference_status,
        "seed": payload.get("seed", ""),
        "mae": payload.get("mae", ""),
        "rmse": payload.get("rmse", ""),
        "mape": payload.get("mape", ""),
        "h3_mae": payload.get("mae_h3", payload.get("h3_mae", "")),
        "h6_mae": payload.get("mae_h6", payload.get("h6_mae", "")),
        "h12_mae": payload.get("mae_h12", payload.get("h12_mae", "")),
        "train_time": payload.get("train_time", ""),
        "test_time": payload.get("test_time", ""),
        "config_path": config_path,
        "checkpoint_path": payload.get("ckpt_path", payload.get("checkpoint_path", str(path.parent))),
    }


def collect(checkpoints_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not checkpoints_dir.exists():
        return rows
    metric_names = ("best_test_metrics.json", "best_metrics.json", "last_metrics.json")
    for metrics_path in sorted(checkpoints_dir.rglob("*_metrics.json")):
        if metrics_path.name not in metric_names:
            continue
        row = _row_from_metrics(metrics_path)
        if row["reference_status"] and row["model_name"]:
            rows.append(row)
    return rows


def write_rows(rows: Iterable[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline metric JSONs into baseline_summary.csv.")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--out", default="results/tables/baseline_summary.csv")
    args = parser.parse_args()

    rows = collect((PROJECT_ROOT / args.checkpoints_dir).resolve())
    out_path = (PROJECT_ROOT / args.out).resolve()
    write_rows(rows, out_path)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
