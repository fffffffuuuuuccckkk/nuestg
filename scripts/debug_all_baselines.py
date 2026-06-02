from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config


DEFAULT_BASELINES_ROOT = Path("/data/OuXiaoyu/mystg/baselines")
DEFAULT_BASICTS_PYTHON = Path("/data/OuXiaoyu/miniconda3/envs/basicts/bin/python")


TARGET_BASELINES: List[Dict] = [
    {
        "name": "STGCN",
        "config": "configs/baselines/pems08/stgcn.py",
        "aliases": ["STGCN", "stgcn"],
    },
    {
        "name": "GraphWaveNet",
        "config": "configs/baselines/pems08/graphwavenet.py",
        "aliases": ["Graph-WaveNet", "GraphWaveNet", "GWNet", "gwnet"],
    },
    {
        "name": "AGCRN",
        "config": "configs/baselines/pems08/agcrn.py",
        "aliases": ["AGCRN"],
    },
    {
        "name": "ST-Norm",
        "config": "configs/baselines/pems08/stnorm.py",
        "aliases": ["ST-Norm", "STNorm", "stnorm"],
    },
    {
        "name": "D2STGNN",
        "config": "configs/baselines/pems08/d2stgnn.py",
        "aliases": ["D2STGNN"],
    },
    {
        "name": "STID",
        "config": "configs/baselines/pems08/stid.py",
        "aliases": ["STID"],
    },
    {
        "name": "CaST-adapter",
        "config": "configs/baselines/pems08/cast.py",
        "aliases": ["CaST", "CAST"],
    },
    {
        "name": "STONE-adapter",
        "config": "configs/baselines/pems08/stone.py",
        "aliases": ["STONE-KDD-2024", "STONE"],
    },
    {
        "name": "STOP-adapter",
        "config": "configs/baselines/pems08/stop.py",
        "aliases": ["STOP"],
    },
]


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_local_repo(baselines_root: Path, aliases: Iterable[str]) -> Path | None:
    if not baselines_root.exists():
        return None
    wanted = {_normalize_name(alias) for alias in aliases}
    for child in baselines_root.iterdir():
        if child.is_dir() and _normalize_name(child.name) in wanted:
            return child
    return None


def _resolve_reference_status(config_path: Path) -> str:
    cfg = load_config(str(config_path))
    run_status = cfg.get("RUN", {}).get("reference_status")
    model_status = cfg.get("MODEL", {}).get("reference_status")
    return str(model_status or run_status or "native_adapter")


def _tail(text: str, lines: int = 80) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug the nine local baseline entries.")
    parser.add_argument("--dataset", default="pems08", choices=["pems08"])
    parser.add_argument("--batch_size", default="4")
    parser.add_argument("--num_workers", default="0")
    parser.add_argument("--baselines_root", default=str(DEFAULT_BASELINES_ROOT))
    parser.add_argument("--python", default=os.environ.get("PYTHON"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args, extra = parser.parse_known_args()

    python_bin = args.python
    if not python_bin:
        python_bin = str(DEFAULT_BASICTS_PYTHON if DEFAULT_BASICTS_PYTHON.exists() else Path(sys.executable))
    baselines_root = Path(args.baselines_root)
    results = []

    print(f"python={python_bin}")
    print(f"baselines_root={baselines_root}")
    for entry in TARGET_BASELINES:
        config_path = PROJECT_ROOT / entry["config"]
        repo_path = find_local_repo(baselines_root, entry["aliases"])
        if repo_path is None:
            results.append((entry["name"], "SKIP", "skipped_local_repo_missing", "local reference repo not found"))
            print(f"[SKIP] {entry['name']} reference_status=skipped_local_repo_missing repo=missing")
            continue
        if not config_path.exists():
            results.append((entry["name"], "SKIP", "skipped_local_repo_missing", f"missing config {entry['config']}"))
            print(f"[SKIP] {entry['name']} reference_status=skipped_local_repo_missing repo={repo_path} config=missing")
            continue

        try:
            reference_status = _resolve_reference_status(config_path)
        except Exception as exc:
            results.append((entry["name"], "FAIL", "unknown", f"config load failed: {exc}"))
            print(f"[FAIL] {entry['name']} config_load_error={exc}")
            continue

        cmd = [
            python_bin,
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
        print(f"[RUN] {entry['name']} reference_status={reference_status} repo={repo_path}")
        print("  command=" + " ".join(cmd))
        if args.dry_run:
            results.append((entry["name"], "DRY_RUN", reference_status, str(repo_path)))
            continue

        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if args.verbose and proc.stdout:
            print(proc.stdout)
        if proc.returncode == 0:
            results.append((entry["name"], "PASS", reference_status, str(repo_path)))
            print(f"[PASS] {entry['name']} reference_status={reference_status}")
        else:
            detail = (_tail(proc.stdout) + "\n" + _tail(proc.stderr)).strip()
            results.append((entry["name"], "FAIL", reference_status, detail))
            print(f"[FAIL] {entry['name']} reference_status={reference_status}")
            if detail:
                print(detail)

    print("\nsummary:")
    for name, status, reference_status, detail in results:
        print(f"{status}\t{name}\t{reference_status}\t{detail}")
    if any(status == "FAIL" for _, status, _, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
