from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_BASELINES_ROOT = Path("/data/OuXiaoyu/mystg/baselines")


class OfficialBaselineSkip(RuntimeError):
    """Raised when a full official wrapper cannot run in the current setting."""

    def __init__(
        self,
        baseline_name: str,
        reason: str,
        reference_status: str = "unsupported_current_dataset",
    ) -> None:
        self.baseline_name = str(baseline_name)
        self.reason = str(reason)
        self.reference_status = str(reference_status)
        super().__init__(f"SKIPPED official {self.baseline_name}: {self.reason}")


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_local_repo(
    baseline_name: str,
    candidate_names: Iterable[str],
    baselines_root: str | Path = DEFAULT_BASELINES_ROOT,
) -> Path | None:
    root = Path(baselines_root)
    if not root.exists():
        return None
    wanted = {_normalize_name(item) for item in candidate_names}
    wanted.add(_normalize_name(baseline_name))
    for child in root.iterdir():
        if child.is_dir() and _normalize_name(child.name) in wanted:
            return child.resolve()
    return None


def ensure_repo_exists_or_skip(
    baseline_name: str,
    candidate_names: Iterable[str],
    baselines_root: str | Path = DEFAULT_BASELINES_ROOT,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path:
        repo_path = Path(explicit_path).resolve()
        if repo_path.exists():
            return repo_path
    else:
        repo_path = find_local_repo(baseline_name, candidate_names, baselines_root)
        if repo_path is not None:
            return repo_path
    raise OfficialBaselineSkip(
        baseline_name,
        "local repo not found",
        reference_status="skipped_local_repo_missing",
    )


def safe_import_from_repo(repo_path: str | Path, module_path: str, class_name: str):
    repo = Path(repo_path).resolve()
    path = repo / module_path
    if not path.exists():
        raise ImportError(f"Official file not found: {path}")
    module_name = f"_official_{_normalize_name(repo.name)}_{_normalize_name(module_path)}"
    old_path = list(sys.path)
    sys.path.insert(0, str(repo))
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {module_path} from {repo}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            raise ImportError(f"{module_path} does not define {class_name}")
        return getattr(module, class_name)
    finally:
        sys.path[:] = old_path


def run_official_subprocess(
    baseline_name: str,
    repo_path: str | Path,
    command: Sequence[str],
    cwd: str | Path | None = None,
    check: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess:
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise OfficialBaselineSkip(
            baseline_name,
            "local repo not found",
            reference_status="skipped_local_repo_missing",
        )
    return subprocess.run(command, cwd=Path(cwd or repo), check=check, **kwargs)
