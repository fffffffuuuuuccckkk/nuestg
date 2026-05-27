from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class FuturePredictiveEnvMask(nn.Module):
    """Select historical environment tokens that are predictive of future env."""

    def __init__(
        self,
        env_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        init_bias: float = -1.0,
        temperature: float = 1.0,
        force_mask_value: Optional[float] = None,
        pooling: str = "masked_mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.temperature = float(temperature)
        self.force_mask_value = force_mask_value
        self.pooling = str(pooling)
        self.eps = float(eps)
        if self.pooling not in {"masked_mean", "mean"}:
            raise ValueError("MODEL.mask_pooling must be 'masked_mean' or 'mean'")

        self.mask_net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        last = self.mask_net[-1]
        if isinstance(last, nn.Linear) and init_bias is not None:
            nn.init.constant_(last.bias, float(init_bias))

    def _pool(self, tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return (tokens * weight).mean(dim=1)
        weighted_sum = (tokens * weight).sum(dim=1)
        denom = weight.sum(dim=1).clamp_min(self.eps)
        return weighted_sum / denom

    def forward(self, env_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if env_tokens.dim() != 4:
            raise AssertionError(f"env_tokens must be [B, L, N, D_env], got {tuple(env_tokens.shape)}")
        batch_size, input_len, num_nodes, env_dim = env_tokens.shape
        if env_dim != self.env_dim:
            raise AssertionError(f"expected env_dim={self.env_dim}, got {env_dim}")

        if self.force_mask_value is None:
            logits = self.mask_net(env_tokens) / max(self.temperature, self.eps)
            mask = torch.sigmoid(logits)
        else:
            mask = torch.full(
                (batch_size, input_len, num_nodes, 1),
                float(self.force_mask_value),
                dtype=env_tokens.dtype,
                device=env_tokens.device,
            )
        mask = mask.clamp(0.0, 1.0)
        inv_mask = 1.0 - mask
        env_plus_tokens = mask * env_tokens
        env_minus_tokens = inv_mask * env_tokens
        env_plus = self._pool(env_tokens, mask)
        env_minus = self._pool(env_tokens, inv_mask)

        expected_mask = (batch_size, input_len, num_nodes, 1)
        expected_env = (batch_size, num_nodes, env_dim)
        if tuple(mask.shape) != expected_mask:
            raise AssertionError(f"mask must be {expected_mask}, got {tuple(mask.shape)}")
        for name, tensor in {"env_plus": env_plus, "env_minus": env_minus}.items():
            if tuple(tensor.shape) != expected_env:
                raise AssertionError(f"{name} must be {expected_env}, got {tuple(tensor.shape)}")

        return {
            "mask": mask,
            "env_plus": env_plus,
            "env_minus": env_minus,
            "env_plus_tokens": env_plus_tokens,
            "env_minus_tokens": env_minus_tokens,
        }
