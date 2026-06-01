from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baseline_registry import iter_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug runnable baselines and print reference status.")
    parser.add_argument("--dataset", default="pems08")
    parser.add_argument(
        "--category",
        default="all",
        choices=["forecasting", "st_ood", "plugin_ours", "all"],
        help=(
            "Which registry category to print. `all` means baseline categories "
            "(forecasting + st_ood); use plugin_ours explicitly for our method variants. "
            "Runnable entries are debugged; external entries are reported and skipped."
        ),
    )
    parser.add_argument("--batch_size", default="4")
    parser.add_argument("--num_workers", default="0")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and reference statuses without running.")
    args, extra = parser.parse_known_args()

    categories = ["forecasting", "st_ood"] if args.category == "all" else [args.category]
    entries = []
    for category in categories:
        entries.extend(iter_baselines(args.dataset, category=category))

    for entry in entries:
        ref_status = entry.get("reference_status", "unknown")
        impl_type = entry.get("implementation_type", ref_status)
        status = entry.get("status", "unknown")
        print(f"[{entry['name']}] status={status} reference_status={ref_status} implementation_type={impl_type}")
        print(f"  repo={entry.get('official_repo') or 'n/a'}")
        print(f"  files={', '.join(entry.get('referenced_files') or ['n/a'])}")
        if status != "runnable":
            print(f"  skip={entry.get('command', 'external result import required')}")
            continue
        cmd = [
            sys.executable,
            "train.py",
            "--config",
            entry["config"],
            "--debug_batch",
            "--set",
            f"TRAIN.batch_size={args.batch_size}",
            "--set",
            f"TRAIN.num_workers={args.num_workers}",
            *extra,
        ]
        print("  command=" + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
