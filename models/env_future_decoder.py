from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class FutureEnvDistributionDecoder(nn.Module):
    """Predict a future environment Gaussian from selected historical env."""

    def __init__(
        self,
        env_dim: int,
        time_emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_time: bool = True,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.use_time = bool(use_time) and self.time_emb_dim > 0
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        in_dim = self.env_dim + (2 * self.time_emb_dim if self.use_time else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * self.env_dim),
        )

    def forward(
        self,
        env_context: torch.Tensor,
        future_time_emb: Optional[torch.Tensor] = None,
        cur_time_emb: Optional[torch.Tensor] = None,
        future_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if env_context.dim() != 3:
            raise AssertionError(f"env_context must be [B,N,D_env], got {tuple(env_context.shape)}")
        batch_size, num_nodes, env_dim = env_context.shape
        if env_dim != self.env_dim:
            raise AssertionError(f"expected env_dim={self.env_dim}, got {env_dim}")
        if future_time_emb is not None:
            if future_time_emb.dim() != 3:
                raise AssertionError(f"future_time_emb must be [B,H,D_t], got {tuple(future_time_emb.shape)}")
            future_len = int(future_time_emb.shape[1])
        if future_len is None:
            raise ValueError("future_len is required when future_time_emb is None")

        context = env_context.unsqueeze(1).expand(-1, future_len, -1, -1)
        parts = [context]
        if self.use_time:
            if future_time_emb is None:
                future_time_emb = env_context.new_zeros(batch_size, future_len, self.time_emb_dim)
            if cur_time_emb is None:
                cur_time_emb = env_context.new_zeros(batch_size, self.time_emb_dim)
            if tuple(future_time_emb.shape) != (batch_size, future_len, self.time_emb_dim):
                raise AssertionError(
                    "future_time_emb must be "
                    f"{(batch_size, future_len, self.time_emb_dim)}, got {tuple(future_time_emb.shape)}"
                )
            if tuple(cur_time_emb.shape) != (batch_size, self.time_emb_dim):
                raise AssertionError(
                    f"cur_time_emb must be {(batch_size, self.time_emb_dim)}, got {tuple(cur_time_emb.shape)}"
                )
            fut = future_time_emb.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            cur = cur_time_emb.unsqueeze(1).unsqueeze(2).expand(-1, future_len, num_nodes, -1)
            parts.extend([fut, cur])

        decoder_in = torch.cat(parts, dim=-1)
        out = self.net(decoder_in)
        pred_mu, pred_logvar = out.chunk(2, dim=-1)
        pred_logvar = pred_logvar.clamp(self.logvar_min, self.logvar_max)
        expected = (batch_size, future_len, num_nodes, self.env_dim)
        for name, tensor in {"pred_fut_mu": pred_mu, "pred_fut_logvar": pred_logvar}.items():
            if tuple(tensor.shape) != expected:
                raise AssertionError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        return {"pred_fut_mu": pred_mu, "pred_fut_logvar": pred_logvar}
