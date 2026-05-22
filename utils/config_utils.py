from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_config(path: str) -> Dict[str, Any]:
    """Load a Python config file defining CONFIG or get_config()."""
    config_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_config"):
        cfg = module.get_config()
    elif hasattr(module, "CONFIG"):
        cfg = module.CONFIG
    else:
        raise AttributeError(f"{config_path} must define CONFIG or get_config()")
    if not isinstance(cfg, dict):
        raise TypeError(f"Config from {config_path} must be a dict, got {type(cfg).__name__}")
    return copy.deepcopy(cfg)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def cast_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        if lowered.startswith("0") and len(lowered) > 1 and not lowered.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_dotlist_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    """Parse KEY.SUBKEY=value command-line overrides into a nested dict."""
    parsed: Dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Override key is empty in {item!r}")
        cursor = parsed
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path conflicts at {key!r} in {item!r}")
        cursor[keys[-1]] = cast_value(raw_value)
    return parsed


def save_resolved_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save a resolved config as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)


def apply_ablation(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Apply a named ablation by mutating and returning cfg."""
    name = name.lower()
    if name == "no_env":
        deep_update(
            cfg,
            {
                "MODEL": {
                    "force_gate_value": 0.0,
                    "persistence": {"enabled": False},
                    "separation": {"enabled": False, "mode": "none"},
                    "use_separated_z_for_y_inv": True,
                },
                "LOSS": {
                    "use_gate": False,
                    "use_swap": False,
                    "use_kl": False,
                    "use_ind": False,
                    "use_sparse": False,
                    "use_entropy": False,
                    "use_residual_norm": False,
                    "use_env_consistency": False,
                    "use_persistence_mi": False,
                    "persistence_affects_gate": False,
                },
                "SWAP": {"enabled": False},
            },
        )
    elif name == "no_gate":
        deep_update(
            cfg,
            {
                "MODEL": {"force_gate_value": 1.0},
                "LOSS": {"use_gate": False, "persistence_affects_gate": False},
            },
        )
    elif name == "no_swap":
        deep_update(cfg, {"LOSS": {"use_swap": False}, "SWAP": {"enabled": False}})
    elif name == "no_kl":
        deep_update(cfg, {"LOSS": {"use_kl": False}})
    elif name == "no_ind":
        deep_update(cfg, {"LOSS": {"use_ind": False}})
    elif name == "no_sparse":
        deep_update(cfg, {"LOSS": {"use_sparse": False}})
    elif name == "no_separation":
        deep_update(
            cfg,
            {
                "MODEL": {
                    "separation": {"enabled": False, "mode": "none"},
                    "use_separated_z_for_y_inv": True,
                }
            },
        )
    elif name == "global_env":
        deep_update(cfg, {"MODEL": {"env_global_mode": True}})
    elif name == "shuffled_env":
        deep_update(
            cfg,
            {
                "MODEL": {"use_shuffled_env_train": True, "use_shuffled_env_eval": True},
                "LOSS": {"use_swap": False, "persistence_affects_gate": False},
                "SWAP": {"enabled": False},
            },
        )
    elif name == "no_persistence":
        deep_update(
            cfg,
            {
                "MODEL": {"persistence": {"enabled": False}},
                "LOSS": {"use_persistence_mi": False, "persistence_affects_gate": False},
            },
        )
    else:
        raise ValueError(
            f"Unknown ablation {name!r}; expected one of "
            "no_env,no_gate,no_swap,no_kl,no_ind,no_sparse,no_separation,"
            "global_env,shuffled_env,no_persistence"
        )
    return cfg


def apply_ablations(cfg: Dict[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    for name in names or []:
        apply_ablation(cfg, name)
    return cfg


def resolve_cli_config(config_path: str, ablations: List[str], dotlist: List[str]) -> Dict[str, Any]:
    cfg = load_config(config_path)
    apply_ablations(cfg, ablations)
    deep_update(cfg, parse_dotlist_overrides(dotlist))
    cfg.setdefault("RUN", {})["ablations"] = list(ablations or [])
    cfg.setdefault("RUN", {})["config_path"] = str(Path(config_path).resolve())
    return cfg
