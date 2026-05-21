from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


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
        mask = mask & existing_mask.bool()
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
