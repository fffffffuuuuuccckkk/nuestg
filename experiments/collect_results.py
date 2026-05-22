from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


FIELDS = [
    "dataset",
    "setting",
    "method",
    "category",
    "backbone",
    "ablation",
    "seed",
    "mae",
    "rmse",
    "mape",
    "best_epoch",
    "ckpt_path",
    "config_path",
    "status",
    "notes",
]


def _row_from_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "dataset": payload.get("dataset", ""),
        "setting": payload.get("setting", payload.get("split", "val")),
        "method": payload.get("method", ""),
        "category": payload.get("category", ""),
        "backbone": payload.get("backbone", ""),
        "ablation": payload.get("ablation", ""),
        "seed": payload.get("seed", ""),
        "mae": payload.get("mae", ""),
        "rmse": payload.get("rmse", ""),
        "mape": payload.get("mape", ""),
        "best_epoch": payload.get("epoch", ""),
        "ckpt_path": payload.get("ckpt_path", str(path.parent)),
        "config_path": payload.get("config_path", ""),
        "status": payload.get("status", "runnable"),
        "notes": payload.get("notes", ""),
    }


def _rows_from_external_csv(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            if not item.get("baseline") and not item.get("method"):
                continue
            yield {
                "dataset": item.get("dataset", ""),
                "setting": item.get("setting", item.get("split", "external")),
                "method": item.get("baseline", item.get("method", "")),
                "category": item.get("category", "external"),
                "backbone": item.get("backbone", ""),
                "ablation": item.get("ablation", ""),
                "seed": item.get("seed", ""),
                "mae": item.get("mae", ""),
                "rmse": item.get("rmse", ""),
                "mape": item.get("mape", ""),
                "best_epoch": item.get("best_epoch", ""),
                "ckpt_path": item.get("ckpt_path", ""),
                "config_path": item.get("config_path", ""),
                "status": item.get("status", "external_required"),
                "notes": item.get("notes", f"imported from {path}"),
            }


def collect(results_dir: Path, checkpoints_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    search_roots = []
    if checkpoints_dir.exists():
        search_roots.append(checkpoints_dir)
    if results_dir.exists():
        search_roots.append(results_dir)
    seen_dirs = set()
    for root in search_roots:
        candidates = []
        candidates.extend(root.rglob("test_metrics.json"))
        candidates.extend(root.rglob("best_metrics.json"))
        candidates.extend(root.rglob("last_metrics.json"))
        for metrics_path in candidates:
            if metrics_path.parent in seen_dirs:
                continue
            seen_dirs.add(metrics_path.parent)
            rows.append(_row_from_metrics(metrics_path))

    raw_dir = results_dir / "raw"
    if raw_dir.exists():
        for csv_path in sorted(raw_dir.glob("*.csv")):
            rows.extend(_rows_from_external_csv(csv_path))
    return rows


def write_rows(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NUE-STG experiment metrics into one CSV.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--out", default="results/tables/all_results.csv")
    args = parser.parse_args()

    rows = collect(Path(args.results_dir), Path(args.checkpoints_dir))
    write_rows(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
