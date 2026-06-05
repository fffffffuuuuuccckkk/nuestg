from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


FIELDS = [
    "ablation",
    "status",
    "ckpt_dir",
    "mae_id",
    "rmse_id",
    "mape_id",
    "wape_id",
    "mae_ood",
    "rmse_ood",
    "mape_ood",
    "wape_ood",
    "diag_mask_density",
    "diag_env_plus_future_nll",
    "diag_env_minus_future_nll",
    "diag_swap_delta_mae",
    "best_epoch",
    "notes",
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def finite_or_blank(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def classify_metric(path: Path, payload: Dict[str, Any]) -> str:
    name = path.name.lower()
    split = str(payload.get("split", "")).lower()
    if any(token in name for token in ("shift", "ood", "test")) or split in {"test", "ood", "shift"}:
        return "ood"
    if split in {"val", "valid", "validation", "id"}:
        return "id"
    if name == "best_metrics.json" or name == "last_metrics.json":
        return "id"
    return "unknown"


def score_payload(path: Path, payload: Dict[str, Any], target: str) -> tuple[int, str]:
    name = path.name.lower()
    split = str(payload.get("split", "")).lower()
    if target == "id":
        if name == "best_metrics.json":
            return (0, name)
        if split in {"val", "valid", "validation", "id"}:
            return (1, name)
        if name == "last_metrics.json":
            return (2, name)
    else:
        if name == "best_test_metrics.json":
            return (0, name)
        if "shift" in name:
            return (1, name)
        if "ood" in name:
            return (2, name)
        if split in {"test", "ood", "shift"}:
            return (3, name)
    return (99, name)


def choose_metric(items: Iterable[tuple[Path, Dict[str, Any]]], target: str) -> Optional[tuple[Path, Dict[str, Any]]]:
    candidates = [(path, payload) for path, payload in items if classify_metric(path, payload) == target]
    if not candidates:
        return None
    return min(candidates, key=lambda item: score_payload(item[0], item[1], target))


def read_command_notes(ckpt_dir: Path) -> str:
    env_path = ckpt_dir / "ablation_command.env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("notes="):
            return line.split("=", 1)[1]
    return ""


def collect_one(ckpt_dir: Path) -> Dict[str, str]:
    metric_items: list[tuple[Path, Dict[str, Any]]] = []
    for path in sorted(ckpt_dir.rglob("*metrics*.json")):
        payload = load_json(path)
        if payload is not None:
            metric_items.append((path, payload))

    id_item = choose_metric(metric_items, "id")
    ood_item = choose_metric(metric_items, "ood")
    id_payload = id_item[1] if id_item else {}
    ood_payload = ood_item[1] if ood_item else {}

    diagnostics = {}
    diag_path = ckpt_dir / "test_env_diagnostics.json"
    if diag_path.exists():
        diagnostics = load_json(diag_path) or {}

    def diag_value(key: str) -> Any:
        diag_key = f"diag_{key}"
        if diag_key in ood_payload:
            return ood_payload.get(diag_key)
        return diagnostics.get(key, diagnostics.get(diag_key))

    status = "ok" if id_payload or ood_payload else "missing_metrics"
    notes = str(ood_payload.get("notes") or id_payload.get("notes") or read_command_notes(ckpt_dir))
    best_epoch = id_payload.get("epoch", ood_payload.get("epoch", ""))

    return {
        "ablation": ckpt_dir.name,
        "status": status,
        "ckpt_dir": str(ckpt_dir),
        "mae_id": finite_or_blank(id_payload.get("mae")),
        "rmse_id": finite_or_blank(id_payload.get("rmse")),
        "mape_id": finite_or_blank(id_payload.get("mape")),
        "wape_id": finite_or_blank(id_payload.get("wape")),
        "mae_ood": finite_or_blank(ood_payload.get("mae")),
        "rmse_ood": finite_or_blank(ood_payload.get("rmse")),
        "mape_ood": finite_or_blank(ood_payload.get("mape")),
        "wape_ood": finite_or_blank(ood_payload.get("wape")),
        "diag_mask_density": finite_or_blank(diag_value("mask_density")),
        "diag_env_plus_future_nll": finite_or_blank(diag_value("env_plus_future_nll")),
        "diag_env_minus_future_nll": finite_or_blank(diag_value("env_minus_future_nll")),
        "diag_swap_delta_mae": finite_or_blank(diag_value("swap_delta_mae")),
        "best_epoch": finite_or_blank(best_epoch),
        "notes": notes,
    }


def discover_ckpt_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    children = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if children:
        return children
    return [root]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PEMS08-OOD GraphWaveNet/FPEM ablation metrics.")
    parser.add_argument("--root", default="checkpoints/pems08_ood_graphwavenet_ablation")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "ablation_summary.tsv"
    rows = [collect_one(path) for path in discover_ckpt_dirs(root)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[collect] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
