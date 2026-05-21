from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from utils.tensor_ops import ensure_blnc


class NodeWiseEnvironmentEncoder(nn.Module):
    """Node-wise dynamic environment encoder for NUE-STG.

    The default path explicitly preserves node identity and returns
    env_mu/env_logvar/env with shape [B, N, D_env]. When env_global_mode=True,
    it intentionally collapses the node dimension to [B, D_env] and broadcasts
    back to [B, N, D_env] for the global-environment ablation only.
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        env_dim: int,
        hidden_dim: int,
        dropout: float,
        use_neighbor: bool,
        global_mode: bool,
        logvar_min: float,
        logvar_max: float,
        reparameterize: bool,
        deterministic_eval: bool,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.env_dim = env_dim
        self.use_neighbor = use_neighbor
        self.global_mode = global_mode
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.reparameterize = reparameterize
        self.deterministic_eval = deterministic_eval

        self.self_encoder = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
            nn.GELU(),
        )
        self.mu_self = self._make_head(env_dim, hidden_dim, env_dim, dropout)
        self.logvar_self = self._make_head(env_dim, hidden_dim, env_dim, dropout)
        self.mu_nei = self._make_head(env_dim * 2, hidden_dim, env_dim, dropout)
        self.logvar_nei = self._make_head(env_dim * 2, hidden_dim, env_dim, dropout)
        self.mu_global = self._make_head(env_dim, hidden_dim, env_dim, dropout)
        self.logvar_global = self._make_head(env_dim, hidden_dim, env_dim, dropout)

    @staticmethod
    def _make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _sample_env(self, env_mu: torch.Tensor, env_logvar: torch.Tensor) -> torch.Tensor:
        if not self.reparameterize:
            return env_mu
        if not self.training and self.deterministic_eval:
            return env_mu
        eps = torch.randn_like(env_mu)
        return env_mu + torch.exp(0.5 * env_logvar) * eps

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

        if self.global_mode:
            h_global = h_self.mean(dim=1)
            env_mu = self.mu_global(h_global).unsqueeze(1).expand(-1, num_nodes, -1)
            env_logvar = self.logvar_global(h_global).unsqueeze(1).expand(-1, num_nodes, -1)
        elif self.use_neighbor and adj_norm is not None:
            h_nei = torch.einsum("ij,bjd->bid", adj_norm.to(h_self.device, h_self.dtype), h_self)
            env_input = torch.cat([h_self, h_nei], dim=-1)
            env_mu = self.mu_nei(env_input)
            env_logvar = self.logvar_nei(env_input)
        else:
            env_mu = self.mu_self(h_self)
            env_logvar = self.logvar_self(h_self)

        env_logvar = env_logvar.clamp(self.logvar_min, self.logvar_max)
        env = self._sample_env(env_mu, env_logvar)

        if not (env.dim() == 3 and env.shape[:2] == (batch_size, num_nodes)):
            raise AssertionError(f"env must be [B, N, D_env], got {tuple(env.shape)}")
        return env_mu, env_logvar, env
