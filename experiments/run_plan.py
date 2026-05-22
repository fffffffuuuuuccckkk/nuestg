from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablation_registry import iter_ablations
from experiments.baseline_registry import iter_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Print runnable NUE-STG experiment commands.")
    parser.add_argument("--dataset", default="pems08")
    parser.add_argument("--kind", choices=["all", "baselines", "ours", "ablations", "external"], default="all")
    args = parser.parse_args()

    if args.kind in {"all", "baselines"}:
        for entry in iter_baselines(args.dataset, status="runnable", category="forecasting"):
            print(entry["command"])
    if args.kind in {"all", "ours"}:
        for entry in iter_baselines(args.dataset, status="runnable", category="plugin_ours"):
            print(entry["command"])
    if args.kind in {"all", "ablations"}:
        for entry in iter_ablations():
            if entry.get("status") == "not_implemented":
                continue
            print(entry["command"])
    if args.kind in {"all", "external"}:
        for entry in iter_baselines(args.dataset, status="external_required"):
            print(f"# {entry['name']}: {entry['command']}")


if __name__ == "__main__":
    main()
