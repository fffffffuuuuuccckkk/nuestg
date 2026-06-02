from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER_STATUSES = {
    "cast_fixed_node_simplified_adapter",
    "stone_fixed_node_simplified_adapter",
    "stop_architecture_adapter_without_sood_protocol",
}

MAIN_TABLE_SAFE_STATUSES = {
    "reference_native",
    "graphwavenet_native_adapter",
    "faithful_native_adapter",
    "stnorm_wavenet_adapter",
    "official_local_wrapper",
}

FIELDS = [
    "dataset",
    "split",
    "display_name",
    "model_name",
    "reference_status",
    "is_official",
    "is_adapter",
    "main_table_safe",
    "unsupported_reason",
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


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _resolve_config_path(config_path: str) -> Path | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def _metadata_from_config_dict(cfg: Dict) -> Dict:
    run_cfg = cfg.get("RUN", {})
    model_cfg = cfg.get("MODEL", {})
    reference_status = str(
        model_cfg.get("reference_status")
        or run_cfg.get("reference_status")
        or ""
    )
    unsupported_reason = str(run_cfg.get("unsupported_reason") or model_cfg.get("unsupported_reason") or "")

    explicit_official = _bool_or_none(run_cfg.get("is_official", model_cfg.get("is_official")))
    explicit_adapter = _bool_or_none(run_cfg.get("is_adapter", model_cfg.get("is_adapter")))
    explicit_safe = _bool_or_none(run_cfg.get("main_table_safe", model_cfg.get("main_table_safe")))

    is_adapter = explicit_adapter if explicit_adapter is not None else reference_status in ADAPTER_STATUSES
    is_official = explicit_official
    if is_official is None:
        is_official = reference_status == "official_local_wrapper" and not unsupported_reason
    if explicit_safe is None:
        main_table_safe = reference_status in MAIN_TABLE_SAFE_STATUSES and not is_adapter and not unsupported_reason
    else:
        main_table_safe = explicit_safe

    display_name = str(
        run_cfg.get("display_name")
        or run_cfg.get("method")
        or model_cfg.get("baseline_name")
        or model_cfg.get("name")
        or ""
    )
    return {
        "display_name": display_name,
        "reference_status": reference_status,
        "is_official": bool(is_official),
        "is_adapter": bool(is_adapter),
        "main_table_safe": bool(main_table_safe),
        "unsupported_reason": unsupported_reason,
    }


def _metadata_from_config_path(config_path: str) -> Dict:
    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return {}
    try:
        from utils.config_utils import load_config

        return _metadata_from_config_dict(load_config(str(resolved)))
    except Exception:
        return {}


def _metadata_from_resolved(metrics_path: Path) -> Dict:
    resolved = metrics_path.parent / "resolved_config.json"
    if not resolved.exists():
        return {}
    try:
        return _metadata_from_config_dict(_load_json(resolved))
    except Exception:
        return {}


def _merge_metadata(*items: Dict) -> Dict:
    merged: Dict = {}
    for item in items:
        for key, value in item.items():
            if value not in ("", None):
                merged[key] = value
    return merged


def _row_from_metrics(path: Path) -> Dict:
    payload = _load_json(path)
    config_path = str(payload.get("config_path", ""))
    metadata = _merge_metadata(
        _metadata_from_config_path(config_path),
        _metadata_from_resolved(path),
        {
            "display_name": payload.get("display_name", ""),
            "reference_status": payload.get("reference_status", ""),
            "is_official": payload.get("is_official", None),
            "is_adapter": payload.get("is_adapter", None),
            "main_table_safe": payload.get("main_table_safe", None),
            "unsupported_reason": payload.get("unsupported_reason", ""),
        },
    )
    model_name = payload.get("method", payload.get("model_name", ""))
    display_name = metadata.get("display_name") or model_name
    reference_status = metadata.get("reference_status", "")
    return {
        "dataset": payload.get("dataset", ""),
        "split": payload.get("split", ""),
        "display_name": display_name,
        "model_name": model_name,
        "reference_status": reference_status,
        "is_official": metadata.get("is_official", ""),
        "is_adapter": metadata.get("is_adapter", ""),
        "main_table_safe": metadata.get("main_table_safe", ""),
        "unsupported_reason": metadata.get("unsupported_reason", ""),
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
