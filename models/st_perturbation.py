from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class STPerturbation(nn.Module):
    """Lightweight spatio-temporal perturbations for invariant consistency."""

    def __init__(
        self,
        enabled: bool = False,
        prob: float = 1.0,
        value_jitter: bool = True,
        jitter_std: float = 0.01,
        value_scale: bool = True,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        time_node_mask: bool = True,
        time_node_mask_ratio: float = 0.1,
        mask_value: str = "zero",
        temporal_block: bool = True,
        temporal_block_ratio: float = 0.1,
        temporal_block_len: int = 2,
        edge_dropout: bool = False,
        edge_dropout_p: float = 0.1,
        edge_dropout_for_env_only: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.prob = float(prob)
        self.value_jitter = bool(value_jitter)
        self.jitter_std = float(jitter_std)
        self.value_scale = bool(value_scale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.time_node_mask = bool(time_node_mask)
        self.time_node_mask_ratio = float(time_node_mask_ratio)
        self.mask_value = str(mask_value or "zero").lower()
        self.temporal_block = bool(temporal_block)
        self.temporal_block_ratio = float(temporal_block_ratio)
        self.temporal_block_len = max(int(temporal_block_len), 1)
        self.edge_dropout = bool(edge_dropout)
        self.edge_dropout_p = float(edge_dropout_p)
        self.edge_dropout_for_env_only = bool(edge_dropout_for_env_only)
        if self.mask_value not in {"zero", "mean"}:
            raise ValueError("MODEL.perturb_mask_value must be 'zero' or 'mean'")

    def enabled_types(self) -> Tuple[str, ...]:
        types = []
        if self.value_jitter:
            types.append("value_jitter")
        if self.value_scale:
            types.append("value_scale")
        if self.time_node_mask:
            types.append("time_node_mask")
        if self.temporal_block:
            types.append("temporal_block")
        if self.edge_dropout:
            types.append("edge_dropout")
        return tuple(types)

    @staticmethod
    def _finite_mean(x: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(x)
        if bool(finite.any()):
            return x[finite].mean()
        return x.new_zeros(())

    def _mask_fill_value(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_value == "mean":
            return self._finite_mean(x)
        return x.new_zeros(())

    def _apply_token_mask(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        fill_value: torch.Tensor,
    ) -> torch.Tensor:
        mask = token_mask.expand_as(x) & torch.isfinite(x)
        return torch.where(mask, fill_value.to(dtype=x.dtype, device=x.device), x)

    def _drop_edges(self, adj: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if adj is None or not self.edge_dropout or self.edge_dropout_p <= 0:
            like = adj if isinstance(adj, torch.Tensor) else None
            if like is None:
                return adj, torch.tensor(0.0)
            return adj, like.new_zeros(())
        p = min(max(float(self.edge_dropout_p), 0.0), 1.0)
        finite = torch.isfinite(adj)
        keep = torch.rand(adj.shape, device=adj.device, dtype=torch.float32) >= p
        dropped = finite & (~keep)
        adj_aug = torch.where(finite, adj * keep.to(dtype=adj.dtype), adj)
        denom = finite.to(dtype=adj.dtype).sum().clamp_min(1.0)
        dropped_ratio = dropped.to(dtype=adj.dtype).sum() / denom
        return adj_aug, dropped_ratio.detach()

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, object]]:
        if x.dim() != 4:
            raise AssertionError(f"STPerturbation expects x as [B,L,N,C], got {tuple(x.shape)}")
        info: Dict[str, object] = {
            "enabled": self.enabled,
            "applied": False,
            "types": self.enabled_types(),
            "edge_dropout_for_env_only": self.edge_dropout_for_env_only,
        }
        if not self.training or not self.enabled:
            return x, adj, info
        if self.prob <= 0 or float(torch.rand((), device=x.device).detach().cpu()) > min(self.prob, 1.0):
            return x, adj, info

        x_aug = x.clone()
        finite = torch.isfinite(x_aug)

        if self.value_scale:
            low = min(self.scale_min, self.scale_max)
            high = max(self.scale_min, self.scale_max)
            scale = torch.empty((x.shape[0], 1, 1, x.shape[-1]), device=x.device, dtype=x.dtype).uniform_(low, high)
            x_aug = torch.where(finite, x_aug * scale, x_aug)
            info["scale_mean"] = scale.detach().mean()

        if self.value_jitter and self.jitter_std > 0:
            noise = torch.randn_like(x_aug) * float(self.jitter_std)
            x_aug = torch.where(torch.isfinite(x_aug), x_aug + noise, x_aug)
            info["jitter_std"] = x_aug.new_tensor(float(self.jitter_std))

        fill_value = self._mask_fill_value(x_aug)
        token_mask = torch.zeros(x_aug.shape[:3] + (1,), dtype=torch.bool, device=x_aug.device)
        if self.time_node_mask and self.time_node_mask_ratio > 0:
            ratio = min(max(float(self.time_node_mask_ratio), 0.0), 1.0)
            sampled = torch.rand(token_mask.shape, device=x_aug.device) < ratio
            token_mask |= sampled
            info["time_node_mask_ratio_actual"] = sampled.to(dtype=x_aug.dtype).mean().detach()

        if self.temporal_block and self.temporal_block_ratio > 0:
            batch_size, seq_len = x_aug.shape[0], x_aug.shape[1]
            block_len = min(self.temporal_block_len, seq_len)
            num_blocks = max(1, int(round(seq_len * min(max(self.temporal_block_ratio, 0.0), 1.0) / block_len)))
            block_mask = torch.zeros_like(token_mask)
            max_start = max(seq_len - block_len + 1, 1)
            for batch_idx in range(batch_size):
                for _ in range(num_blocks):
                    start = int(torch.randint(0, max_start, (), device=x_aug.device).detach().cpu())
                    block_mask[batch_idx, start : start + block_len, :, :] = True
            token_mask |= block_mask
            info["temporal_block_ratio_actual"] = block_mask.to(dtype=x_aug.dtype).mean().detach()

        if token_mask.any():
            x_aug = self._apply_token_mask(x_aug, token_mask, fill_value)
            info["mask_ratio_actual"] = token_mask.to(dtype=x_aug.dtype).mean().detach()
        else:
            info["mask_ratio_actual"] = x_aug.new_zeros(())

        adj_aug, edge_dropped_ratio = self._drop_edges(adj)
        info["edge_dropped_ratio"] = edge_dropped_ratio.to(device=x_aug.device) if isinstance(edge_dropped_ratio, torch.Tensor) else edge_dropped_ratio
        x_aug = torch.nan_to_num(x_aug, nan=0.0, posinf=1e6, neginf=-1e6)
        info["x_aug_stats"] = {
            "min": x_aug.detach().amin(),
            "max": x_aug.detach().amax(),
            "mean": x_aug.detach().mean(),
            "std": x_aug.detach().std(unbiased=False),
        }
        info["applied"] = True
        return x_aug, adj_aug, info
