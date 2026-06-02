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

MAIN_TABLE_SAFE_STATUSES = {
    "reference_native",
    "graphwavenet_native_adapter",
    "faithful_native_adapter",
    "stnorm_wavenet_adapter",
    "official_local_wrapper",
}

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
        "aliases": ["D2STGNN", "GestaltCogTeam-D2STGNN"],
    },
    {
        "name": "STID",
        "config": "configs/baselines/pems08/stid.py",
        "aliases": ["STID"],
    },
    {
        "name": "CaST-faithful-pytorch-adapter",
        "config": "configs/baselines/pems08/cast.py",
        "aliases": ["CaST", "CAST"],
    },
    {
        "name": "STONE-faithful-pytorch-adapter",
        "config": "configs/baselines/pems08/stone.py",
        "aliases": ["STONE-KDD-2024", "STONE"],
    },
    {
        "name": "STOP-faithful-architecture-adapter",
        "config": "configs/baselines/pems08/stop.py",
        "aliases": ["STOP"],
    },
    {
        "name": "CaST-official",
        "config": "configs/baselines/pems08/cast_official.py",
        "aliases": ["CaST", "CAST"],
    },
    {
        "name": "STONE-official",
        "config": "configs/baselines/pems08/stone_official.py",
        "aliases": ["STONE-KDD-2024", "STONE"],
    },
    {
        "name": "STOP-official",
        "config": "configs/baselines/pems08/stop_official.py",
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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _resolve_metadata(config_path: Path) -> Dict:
    cfg = load_config(str(config_path))
    run_cfg = cfg.get("RUN", {})
    model_cfg = cfg.get("MODEL", {})
    reference_status = str(
        model_cfg.get("reference_status")
        or run_cfg.get("reference_status")
        or "native_adapter"
    )
    explicit_safe = _bool_or_none(run_cfg.get("main_table_safe", model_cfg.get("main_table_safe")))
    unsupported_reason = str(run_cfg.get("unsupported_reason") or model_cfg.get("unsupported_reason") or "")
    if explicit_safe is None:
        main_table_safe = reference_status in MAIN_TABLE_SAFE_STATUSES and not unsupported_reason
    else:
        main_table_safe = explicit_safe
    return {
        "reference_status": reference_status,
        "main_table_safe": main_table_safe,
        "unsupported_reason": unsupported_reason,
    }


def _tail(text: str, lines: int = 80) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _line_value(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(prefix.lower()):
            return line.split(":", 1)[1].strip() if ":" in line else line.strip()
    return ""


def _is_official_skip(text: str) -> bool:
    lowered = text.lower()
    return (
        "skipped official" in lowered
        or "reference_status: unsupported_current_dataset" in lowered
        or "reference_status: skipped_local_repo_missing" in lowered
    )


def _extract_skip_reason(text: str, fallback: str) -> str:
    unsupported = _line_value(text, "unsupported_reason:")
    if unsupported:
        return unsupported
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("skipped official"):
            return stripped
    return fallback


def _print_result(result: Dict) -> None:
    print(f"Baseline: {result['name']}")
    print(f"Status: {result['status']}")
    print(f"Reference: {result['reference_status']}")
    print(f"Main-table safe: {_yes_no(result['main_table_safe'])}")
    print(f"Reason: {result['reason']}")
    if result.get("detail"):
        print(result["detail"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug local baseline entries and official wrapper checks.")
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
    print()

    for entry in TARGET_BASELINES:
        config_path = PROJECT_ROOT / entry["config"]
        metadata = {
            "reference_status": "unknown",
            "main_table_safe": False,
            "unsupported_reason": "",
        }
        if config_path.exists():
            try:
                metadata = _resolve_metadata(config_path)
            except Exception as exc:
                result = {
                    "name": entry["name"],
                    "status": "FAIL",
                    "reference_status": "unknown",
                    "main_table_safe": False,
                    "reason": f"config load failed: {exc}",
                    "detail": "",
                }
                results.append(result)
                _print_result(result)
                continue

        repo_path = find_local_repo(baselines_root, entry["aliases"])
        if repo_path is None:
            result = {
                "name": entry["name"],
                "status": "SKIP",
                "reference_status": "skipped_local_repo_missing",
                "main_table_safe": False,
                "reason": "local reference repo not found",
                "detail": "",
            }
            results.append(result)
            _print_result(result)
            continue

        if not config_path.exists():
            result = {
                "name": entry["name"],
                "status": "SKIP",
                "reference_status": "skipped_local_repo_missing",
                "main_table_safe": False,
                "reason": f"missing config {entry['config']}",
                "detail": "",
            }
            results.append(result)
            _print_result(result)
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
        if args.dry_run:
            result = {
                "name": entry["name"],
                "status": "DRY_RUN",
                "reference_status": metadata["reference_status"],
                "main_table_safe": bool(metadata["main_table_safe"]),
                "reason": str(repo_path),
                "detail": "command=" + " ".join(cmd),
            }
            results.append(result)
            _print_result(result)
            continue

        print(f"[RUN] {entry['name']} repo={repo_path}")
        print("command=" + " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if args.verbose and proc.stdout:
            print(proc.stdout)

        output_reference = _line_value(combined, "reference_status:") or metadata["reference_status"]
        if _is_official_skip(combined):
            result = {
                "name": entry["name"],
                "status": "SKIP",
                "reference_status": output_reference,
                "main_table_safe": False,
                "reason": _extract_skip_reason(combined, metadata["unsupported_reason"] or "official wrapper skipped"),
                "detail": "",
            }
        elif proc.returncode == 0:
            result = {
                "name": entry["name"],
                "status": "PASS",
                "reference_status": output_reference,
                "main_table_safe": bool(metadata["main_table_safe"]),
                "reason": "debug_batch ok",
                "detail": "",
            }
        else:
            detail = (_tail(proc.stdout) + "\n" + _tail(proc.stderr)).strip()
            result = {
                "name": entry["name"],
                "status": "FAIL",
                "reference_status": output_reference,
                "main_table_safe": False,
                "reason": f"debug_batch failed with returncode={proc.returncode}",
                "detail": detail,
            }
        results.append(result)
        _print_result(result)

    print("summary:")
    print("Status\tBaseline\tReference\tMain-table safe\tReason")
    for result in results:
        print(
            f"{result['status']}\t{result['name']}\t{result['reference_status']}\t"
            f"{_yes_no(result['main_table_safe'])}\t{result['reason']}"
        )
    if any(result["status"] == "FAIL" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
