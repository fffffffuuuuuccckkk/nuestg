from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from utils.tensor_ops import ensure_blnc


class NodeWiseEnvironmentEncoder(nn.Module):
    """Node-wise dynamic environment encoder for NUE-STG.

    The encoder explicitly preserves node identity and returns local environment
    variables with shape [B, N, D_env]. A local environment E_{v,t} is useful
    only if it adds conditional predictive information beyond Z_{v,t}; this
    branch therefore estimates node-level environment variables instead of a
    graph-level context vector.
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        env_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.env_dim = env_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
            nn.GELU(),
        )
        self.mu_self = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
        )
        self.logvar_self = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
        )
        self.mu_nei = nn.Sequential(
            nn.Linear(env_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
        )
        self.logvar_nei = nn.Sequential(
            nn.Linear(env_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = ensure_blnc(x, "x")
        batch_size, input_len, num_nodes, input_dim = x.shape
        if input_len != self.input_len:
            raise ValueError(f"expected input_len={self.input_len}, got {input_len}")
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")

        node_history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, input_len * input_dim)
        h_self = self.self_encoder(node_history)

        if adj_norm is not None:
            h_nei = torch.einsum("ij,bjd->bid", adj_norm.to(h_self.device, h_self.dtype), h_self)
            env_input = torch.cat([h_self, h_nei], dim=-1)
            env_mu = self.mu_nei(env_input)
            env_logvar = self.logvar_nei(env_input)
        else:
            env_mu = self.mu_self(h_self)
            env_logvar = self.logvar_self(h_self)

        env_logvar = env_logvar.clamp(-10.0, 10.0)
        if self.training:
            eps = torch.randn_like(env_mu)
            env = env_mu + torch.exp(0.5 * env_logvar) * eps
        else:
            env = env_mu

        if not (env.dim() == 3 and env.shape[:2] == (batch_size, num_nodes)):
            raise AssertionError(f"env must be [B, N, D_env], got {tuple(env.shape)}")
        return env_mu, env_logvar, env
