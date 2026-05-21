from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


class AverageMeterDict:
    """Small helper for aggregating scalar logs."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            if value is None:
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def mean(self) -> Dict[str, float]:
        return {key: self.sums[key] / max(self.counts.get(key, 0), 1) for key in self.sums}

    def average(self) -> Dict[str, float]:
        return self.mean()

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


def tensor_to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def stringify_logs(logs: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in logs.items():
        if value is None:
            continue
        try:
            out[key] = tensor_to_float(value)
        except Exception:
            continue
    return out


def format_logs(logs: Dict[str, object], keys: Optional[Iterable[str]] = None) -> str:
    scalar_logs = stringify_logs(logs)
    ordered_keys = list(keys) if keys is not None else sorted(scalar_logs)
    return " ".join(f"{key}={scalar_logs[key]:.6f}" for key in ordered_keys if key in scalar_logs)


def append_csv_log(path: str | Path, row: Dict[str, object], fieldnames: Optional[Iterable[str]] = None) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames) if fieldnames is not None else list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
