from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TABLE_SPECS = {
    "table_forecasting.csv": lambda r: r.get("category") == "forecasting"
    or (r.get("category") == "external" and r.get("setting") == "forecasting"),
    "table_ood.csv": lambda r: r.get("category") in {"st_ood", "external"} or r.get("setting") == "ood",
    "table_plugin_backbone.csv": lambda r: r.get("category") == "plugin_ours",
    "table_ablation.csv": lambda r: bool(r.get("ablation")),
    "table_separation.csv": lambda r: "separation" in (r.get("ablation") or r.get("method") or "").lower(),
    "table_persistence.csv": lambda r: "persistence" in (r.get("ablation") or r.get("method") or "").lower(),
}


def _to_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format(values: Iterable[float]) -> str:
    vals = list(values)
    if not vals:
        return ""
    mean = statistics.mean(vals)
    if len(vals) == 1:
        return f"{mean:.6f}"
    std = statistics.stdev(vals)
    return f"{mean:.6f} ± {std:.6f}"


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("setting", ""),
            row.get("method", ""),
            row.get("backbone", ""),
            row.get("ablation", ""),
        )
        grouped[key].append(row)

    out = []
    for (dataset, setting, method, backbone, ablation), group in sorted(grouped.items()):
        metric_values = {
            metric: [
                value
                for value in (_to_float(item.get(metric, "")) for item in group)
                if value is not None
            ]
            for metric in ["mae", "rmse", "mape"]
        }
        out.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "backbone": backbone,
                "ablation": ablation,
                "seeds": str(len({item.get("seed", "") for item in group})),
                "mae": _format(metric_values["mae"]),
                "rmse": _format(metric_values["rmse"]),
                "mape": _format(metric_values["mape"]),
                "status": group[0].get("status", ""),
                "notes": group[0].get("notes", ""),
            }
        )
    return out


def write_table(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "setting", "method", "backbone", "ablation", "seeds", "mae", "rmse", "mape", "status", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style CSV tables from all_results.csv.")
    parser.add_argument("--input", default="results/tables/all_results.csv")
    parser.add_argument("--out_dir", default="results/tables")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    for filename, predicate in TABLE_SPECS.items():
        table_rows = aggregate([row for row in rows if predicate(row)])
        write_table(table_rows, out_dir / filename)
        print(f"wrote {len(table_rows)} rows to {out_dir / filename}")


if __name__ == "__main__":
    main()
