from __future__ import annotations

import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class ZScoreDataScaler:
    """Graph WaveNet / BasicTS-style z-score scaler fitted on train data.

    The local runner keeps the model and loss in normalized space, then inverse
    transforms predictions and targets for reported metrics. By default the
    scaler uses one global mean/std over the training split, matching the common
    Graph WaveNet preprocessing convention. Per-channel stats are also
    supported for multi-channel data.
    """

    def __init__(
        self,
        mean: np.ndarray | float,
        std: np.ndarray | float,
        enabled: bool = True,
        eps: float = 1e-5,
    ) -> None:
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(self.eps)

    @classmethod
    def identity(cls) -> "ZScoreDataScaler":
        return cls(0.0, 1.0, enabled=False)

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        null_val: Optional[float] = None,
        norm_each_channel: bool = False,
        eps: float = 1e-5,
    ) -> "ZScoreDataScaler":
        array = np.asarray(data, dtype=np.float32)
        mask = np.isfinite(array)
        if null_val is not None:
            if isinstance(null_val, float) and np.isnan(null_val):
                mask &= ~np.isnan(array)
            else:
                mask &= array != null_val

        if norm_each_channel and array.ndim >= 1:
            channels = array.shape[-1] if array.ndim >= 3 else 1
            reshaped = array.reshape(-1, channels)
            mask_reshaped = mask.reshape(-1, channels)
            mean = np.zeros((channels,), dtype=np.float32)
            std = np.ones((channels,), dtype=np.float32)
            for idx in range(channels):
                values = reshaped[mask_reshaped[:, idx], idx]
                if values.size > 0:
                    mean[idx] = np.mean(values, dtype=np.float64)
                    std[idx] = np.std(values, dtype=np.float64)
            std = np.maximum(std, eps)
            view_shape = (1,) * (max(array.ndim, 3) - 1) + (channels,)
            return cls(mean.reshape(view_shape), std.reshape(view_shape), enabled=True, eps=eps)

        values = array[mask]
        if values.size == 0:
            return cls(0.0, 1.0, enabled=True, eps=eps)
        mean = float(np.mean(values, dtype=np.float64))
        std = max(float(np.std(values, dtype=np.float64)), eps)
        return cls(mean, std, enabled=True, eps=eps)

    def to(self, device: torch.device | str) -> "ZScoreDataScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return tensor
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        scaled = (tensor - mean) / std
        if mask is not None:
            scaled = torch.where(mask.to(tensor.device).bool(), scaled, tensor)
        return scaled

    def inverse_transform(self, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return tensor
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        restored = tensor * std + mean
        if mask is not None:
            restored = torch.where(mask.to(tensor.device).bool(), restored, tensor)
        return restored

    def state_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "eps": self.eps,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    def summary(self) -> Dict[str, float | bool]:
        mean = self.mean.detach().float().cpu()
        std = self.std.detach().float().cpu()
        return {
            "enabled": self.enabled,
            "mean": float(mean.mean()),
            "std": float(std.mean()),
            "mean_min": float(mean.min()),
            "mean_max": float(mean.max()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        }


def ensure_blnc(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    """Convert BasicTS forecasting tensors to [B, L, N, C]."""
    if x.dim() == 3:
        return x.unsqueeze(-1)
    if x.dim() == 4:
        return x
    raise ValueError(f"{name} must have shape [B, L, N] or [B, L, N, C], got {tuple(x.shape)}")


def align_target(targets: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Align y_true to prediction shape [B, H, N, C_out]."""
    targets = ensure_blnc(targets, "targets")
    if targets.shape[:3] != prediction.shape[:3]:
        raise ValueError(
            "targets and prediction must share [B, H, N], "
            f"got targets={tuple(targets.shape)} prediction={tuple(prediction.shape)}"
        )
    if targets.shape[-1] == prediction.shape[-1]:
        return targets
    if targets.shape[-1] > prediction.shape[-1]:
        return targets[..., : prediction.shape[-1]]
    raise ValueError(
        "targets has fewer channels than prediction, "
        f"got targets={tuple(targets.shape)} prediction={tuple(prediction.shape)}"
    )


def make_valid_mask(
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if null_val is not None:
        if isinstance(null_val, float) and np.isnan(null_val):
            mask = mask & ~torch.isnan(targets)
        else:
            mask = mask & (targets != null_val)
    if existing_mask is not None:
        if existing_mask.dim() == 3:
            existing_mask = existing_mask.unsqueeze(-1)
        if existing_mask.dim() != targets.dim():
            raise ValueError(
                "existing_mask must have the same rank as targets after optional channel unsqueeze, "
                f"got mask={tuple(existing_mask.shape)} targets={tuple(targets.shape)}"
            )
        for mask_size, target_size in zip(existing_mask.shape, targets.shape):
            if mask_size not in (1, target_size):
                raise ValueError(
                    "existing_mask is not broadcast-compatible with targets, "
                    f"got mask={tuple(existing_mask.shape)} targets={tuple(targets.shape)}"
                )
        mask = mask & existing_mask.to(device=targets.device).bool().expand_as(targets)
    return mask


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(dtype=values.dtype, device=values.device)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_abs_error(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = align_target(targets, prediction)
    mask = make_valid_mask(targets, null_val, existing_mask)
    abs_error = (prediction - torch.nan_to_num(targets, nan=0.0)).abs()
    return abs_error, mask


def masked_mae_value(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    abs_error, mask = masked_abs_error(prediction, targets, null_val, existing_mask)
    return masked_mean(abs_error, mask)


def masked_rmse_value(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.sqrt(masked_mse_value(prediction, targets, null_val, existing_mask).clamp_min(0.0))


def masked_mse_value(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = align_target(targets, prediction)
    mask = make_valid_mask(targets, null_val, existing_mask)
    mask = mask & torch.isfinite(prediction)
    squared_error = (torch.nan_to_num(prediction, nan=0.0) - torch.nan_to_num(targets, nan=0.0)).pow(2)
    return masked_mean(squared_error, mask)


def make_mape_valid_mask(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
    threshold: float = 1.0,
) -> torch.Tensor:
    targets = align_target(targets, prediction)
    valid = make_valid_mask(targets, null_val, existing_mask)
    valid = valid & torch.isfinite(prediction) & torch.isfinite(targets)
    valid = valid & (targets.abs() > float(threshold))
    return valid


def masked_mape_value(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    threshold: float = 1.0,
    as_percent: bool = True,
) -> torch.Tensor:
    targets = align_target(targets, prediction)
    valid = make_mape_valid_mask(prediction, targets, null_val, existing_mask, threshold=threshold)
    if valid.sum() == 0:
        return prediction.new_tensor(float("nan"))
    denom = targets[valid].abs().clamp_min(float(eps))
    ape = (prediction[valid] - targets[valid]).abs() / denom
    value = ape.mean()
    return value * 100.0 if as_percent else value


def masked_wape_value(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val: Optional[float] = None,
    existing_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    as_percent: bool = True,
) -> torch.Tensor:
    targets = align_target(targets, prediction)
    mask = make_valid_mask(targets, null_val, existing_mask)
    mask = mask & torch.isfinite(prediction) & torch.isfinite(targets)
    if mask.sum() == 0:
        return prediction.new_tensor(float("nan"))
    abs_error = (torch.nan_to_num(prediction, nan=0.0) - torch.nan_to_num(targets, nan=0.0)).abs()
    numerator = abs_error[mask].sum()
    denominator = targets[mask].abs().sum().clamp_min(float(eps))
    value = numerator / denominator
    return value * 100.0 if as_percent else value


def generate_tod_dow_timestamps(
    num_steps: int,
    frequency_minutes: int,
    start_time: str = "2000-01-03T00:00:00",
) -> np.ndarray:
    """Generate BasicTS-style [time_of_day, day_of_week] features.

    Values are normalized to [0, 1): time_of_day is minute-of-day / 1440 and
    day_of_week is Python weekday / 7. The default start date is a Monday.
    """
    start = datetime.fromisoformat(start_time)
    freq = int(frequency_minutes)
    timestamps = np.zeros((int(num_steps), 2), dtype=np.float32)
    for step in range(int(num_steps)):
        current = start + timedelta(minutes=freq * step)
        timestamps[step, 0] = (current.hour * 60 + current.minute) / 1440.0
        timestamps[step, 1] = current.weekday() / 7.0
    return timestamps


def maybe_generate_timestamp_file(
    data_file_path: str | Path,
    split: str,
    dataset_cfg: Dict,
) -> bool:
    """Create missing BasicTS timestamp files from frequency/start metadata.

    The function is intentionally local-runner scoped: it never edits BasicTS
    package code. If timestamps already exist, no work is done.
    """
    if not dataset_cfg.get("use_timestamps", False):
        return False
    data_dir = Path(data_file_path)
    timestamp_path = data_dir / f"{split}_timestamps.npy"
    if timestamp_path.exists():
        return False
    if not dataset_cfg.get("auto_generate_timestamps", True):
        return False
    data_path = data_dir / f"{split}_data.npy"
    if not data_path.exists():
        return False

    meta = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        try:
            import json

            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:  # pragma: no cover - metadata is optional
            warnings.warn(f"Failed to read {meta_path}: {exc}; falling back to config metadata.", RuntimeWarning)

    freq = (
        dataset_cfg.get("frequency_minutes")
        or dataset_cfg.get("freq_minutes")
        or meta.get("frequency_minutes")
        or meta.get("frequency (minutes)")
        or meta.get("freq_minutes")
        or 5
    )
    start_times = dataset_cfg.get("timestamp_start_times") or meta.get("timestamp_start_times") or {}
    split_meta = (meta.get("splits") or {}).get(split, {}) if isinstance(meta.get("splits"), dict) else {}
    start_time = (
        start_times.get(split)
        or split_meta.get("start_time")
        or dataset_cfg.get("start_time")
        or meta.get("start_time")
    )
    if start_time is None:
        start_time = "2000-01-03T00:00:00"
        warnings.warn(
            f"{timestamp_path} is missing and no start_time metadata was found; "
            f"generating relative timestamps from {start_time}.",
            RuntimeWarning,
        )

    data = np.load(data_path, mmap_mode="r")
    timestamps = generate_tod_dow_timestamps(data.shape[0], int(freq), str(start_time))
    np.save(timestamp_path, timestamps)
    return True


def normalize_adjacency(adj: np.ndarray, adj_norm: str) -> np.ndarray:
    adj_norm = (adj_norm or "none").lower()
    if adj_norm == "none":
        return adj.astype(np.float32)
    if adj_norm == "row":
        row_sum = adj.sum(axis=-1, keepdims=True)
        return (adj / np.maximum(row_sum, 1e-6)).astype(np.float32)
    if adj_norm == "sym":
        degree = adj.sum(axis=-1)
        degree_inv_sqrt = np.power(np.maximum(degree, 1e-6), -0.5)
        return (degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]).astype(np.float32)
    raise ValueError(f"Unsupported adj_norm={adj_norm!r}; expected one of none,row,sym")


def random_walk_matrix(adj: np.ndarray) -> np.ndarray:
    row_sum = adj.sum(axis=-1, keepdims=True)
    return (adj / np.maximum(row_sum, 1e-6)).astype(np.float32)


def load_graph_supports(
    adj_path: Optional[str],
    num_nodes: int,
    adjtype: str = "doubletransition",
    add_self_loop: bool = False,
) -> Optional[torch.Tensor]:
    """Load Graph WaveNet static supports.

    `doubletransition` returns forward and reverse random-walk matrices,
    matching the common Graph WaveNet preprocessing trick.
    """
    if not adj_path:
        return None
    path = Path(adj_path)
    if not path.exists():
        warnings.warn(f"Adjacency file not found at {path}; GraphWaveNet static supports disabled.", RuntimeWarning)
        return None
    try:
        if path.suffix == ".npy":
            adj = np.load(path)
        else:
            with path.open("rb") as f:
                adj = pickle.load(f)
            if isinstance(adj, (list, tuple)):
                candidates = [x for x in adj if hasattr(x, "shape") and len(x.shape) == 2]
                adj = candidates[-1] if candidates else adj[0]
        adj = np.asarray(adj, dtype=np.float32)
        if adj.shape != (num_nodes, num_nodes):
            warnings.warn(
                f"Adjacency shape from {path} is {adj.shape}, expected {(num_nodes, num_nodes)}; "
                "GraphWaveNet static supports disabled.",
                RuntimeWarning,
            )
            return None
        if add_self_loop:
            adj = adj + np.eye(num_nodes, dtype=np.float32)
        adjtype = str(adjtype or "doubletransition").lower()
        if adjtype in {"doubletransition", "dual_random_walk", "double_transition"}:
            supports = [random_walk_matrix(adj), random_walk_matrix(adj.T)]
        elif adjtype in {"transition", "random_walk", "row"}:
            supports = [random_walk_matrix(adj)]
        elif adjtype in {"sym", "symadj", "symmetric"}:
            supports = [normalize_adjacency(adj, "sym")]
        elif adjtype in {"identity", "none"}:
            supports = [np.eye(num_nodes, dtype=np.float32)]
        else:
            raise ValueError(
                f"Unsupported GraphWaveNet adjtype={adjtype!r}; "
                "expected doubletransition, transition, sym, identity, or none."
            )
        return torch.from_numpy(np.stack(supports, axis=0).astype(np.float32))
    except Exception as exc:
        warnings.warn(f"Failed to load GraphWaveNet supports from {path}: {exc}", RuntimeWarning)
        return None


def load_adjacency(
    adj_path: Optional[str],
    num_nodes: int,
    adj_norm: str = "sym",
    add_self_loop: bool = True,
) -> Optional[torch.Tensor]:
    if not adj_path:
        warnings.warn("No adj_path configured; environment encoder will use self-only mode.", RuntimeWarning)
        return None
    path = Path(adj_path)
    if not path.exists():
        warnings.warn(f"Adjacency file not found at {path}; using self-only environment encoder.", RuntimeWarning)
        return None

    try:
        if path.suffix == ".npy":
            adj = np.load(path)
        else:
            with path.open("rb") as f:
                adj = pickle.load(f)
            if isinstance(adj, (list, tuple)):
                candidates = [x for x in adj if hasattr(x, "shape") and len(x.shape) == 2]
                adj = candidates[-1] if candidates else adj[0]

        adj = np.asarray(adj, dtype=np.float32)
        if adj.shape != (num_nodes, num_nodes):
            warnings.warn(
                f"Adjacency shape from {path} is {adj.shape}, expected {(num_nodes, num_nodes)}; "
                "using self-only environment encoder.",
                RuntimeWarning,
            )
            return None
        if add_self_loop:
            adj = adj + np.eye(num_nodes, dtype=np.float32)
        return torch.from_numpy(normalize_adjacency(adj, adj_norm))
    except Exception as exc:
        warnings.warn(f"Failed to load adjacency from {path}: {exc}; using self-only mode.", RuntimeWarning)
        return None


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
