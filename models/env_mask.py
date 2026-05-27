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
        time_emb_dim: int = 0,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.use_time = bool(use_time) and self.time_emb_dim > 0
        self.temperature = float(temperature)
        self.force_mask_value = force_mask_value
        self.pooling = str(pooling)
        self.eps = float(eps)
        if self.pooling not in {"masked_mean", "mean"}:
            raise ValueError("MODEL.mask_pooling must be 'masked_mean' or 'mean'")

        mask_in_dim = env_dim + (2 * self.time_emb_dim if self.use_time else 0)
        self.mask_net = nn.Sequential(
            nn.Linear(mask_in_dim, hidden_dim),
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

    def forward(
        self,
        env_tokens: torch.Tensor,
        seq_time_emb: Optional[torch.Tensor] = None,
        cur_time_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if env_tokens.dim() != 4:
            raise AssertionError(f"env_tokens must be [B, L, N, D_env], got {tuple(env_tokens.shape)}")
        batch_size, input_len, num_nodes, env_dim = env_tokens.shape
        if env_dim != self.env_dim:
            raise AssertionError(f"expected env_dim={self.env_dim}, got {env_dim}")

        if self.force_mask_value is None:
            mask_input = env_tokens
            if self.use_time:
                if seq_time_emb is None:
                    seq_time_emb = env_tokens.new_zeros(batch_size, input_len, self.time_emb_dim)
                if cur_time_emb is None:
                    cur_time_emb = env_tokens.new_zeros(batch_size, self.time_emb_dim)
                if tuple(seq_time_emb.shape) != (batch_size, input_len, self.time_emb_dim):
                    raise AssertionError(
                        f"seq_time_emb must be {(batch_size, input_len, self.time_emb_dim)}, "
                        f"got {tuple(seq_time_emb.shape)}"
                    )
                if tuple(cur_time_emb.shape) != (batch_size, self.time_emb_dim):
                    raise AssertionError(
                        f"cur_time_emb must be {(batch_size, self.time_emb_dim)}, got {tuple(cur_time_emb.shape)}"
                    )
                seq = seq_time_emb.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                cur = cur_time_emb.unsqueeze(1).unsqueeze(2).expand(-1, input_len, num_nodes, -1)
                mask_input = torch.cat([env_tokens, seq, cur], dim=-1)
            logits = self.mask_net(mask_input) / max(self.temperature, self.eps)
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
        env_hist = env_tokens.mean(dim=1)

        expected_mask = (batch_size, input_len, num_nodes, 1)
        expected_env = (batch_size, num_nodes, env_dim)
        if tuple(mask.shape) != expected_mask:
            raise AssertionError(f"mask must be {expected_mask}, got {tuple(mask.shape)}")
        for name, tensor in {"env_plus": env_plus, "env_minus": env_minus, "env_hist": env_hist}.items():
            if tuple(tensor.shape) != expected_env:
                raise AssertionError(f"{name} must be {expected_env}, got {tuple(tensor.shape)}")

        return {
            "mask": mask,
            "env_plus": env_plus,
            "env_minus": env_minus,
            "env_hist": env_hist,
            "env_plus_tokens": env_plus_tokens,
            "env_minus_tokens": env_minus_tokens,
        }
