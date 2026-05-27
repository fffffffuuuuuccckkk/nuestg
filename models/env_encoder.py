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


class TimeNodeEnvironmentEncoder(nn.Module):
    """Time-node environment token encoder for FPEM.

    This encoder keeps the historical time axis and returns token-level
    Gaussian environment features with shape [B, L, N, D_env]. The old
    NodeWiseEnvironmentEncoder remains unchanged for NUE-STG.
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
        time_emb_dim: int = 0,
        use_node_embedding: bool = False,
        num_nodes: int = 1,
        node_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.env_dim = int(env_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.use_node_embedding = bool(use_node_embedding) and int(node_emb_dim) > 0
        self.use_neighbor = bool(use_neighbor)
        self.global_mode = bool(global_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.reparameterize = bool(reparameterize)
        self.deterministic_eval = bool(deterministic_eval)
        if self.use_node_embedding:
            self.node_emb = nn.Parameter(torch.empty(int(num_nodes), int(node_emb_dim)))
            nn.init.xavier_uniform_(self.node_emb)
            node_dim = int(node_emb_dim)
        else:
            self.node_emb = None
            node_dim = 0

        token_in_dim = input_dim + 2 * self.time_emb_dim + node_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(token_in_dim, hidden_dim),
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
        seq_time_emb: Optional[torch.Tensor] = None,
        cur_time_emb: Optional[torch.Tensor] = None,
        adj_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = ensure_blnc(x, "x")
        batch_size, input_len, num_nodes, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")

        parts = [x]
        if self.time_emb_dim > 0:
            if seq_time_emb is None:
                seq_time_emb = x.new_zeros(batch_size, input_len, self.time_emb_dim)
            if cur_time_emb is None:
                cur_time_emb = x.new_zeros(batch_size, self.time_emb_dim)
            if tuple(seq_time_emb.shape) != (batch_size, input_len, self.time_emb_dim):
                raise AssertionError(
                    f"seq_time_emb must be {(batch_size, input_len, self.time_emb_dim)}, "
                    f"got {tuple(seq_time_emb.shape)}"
                )
            if tuple(cur_time_emb.shape) != (batch_size, self.time_emb_dim):
                raise AssertionError(
                    f"cur_time_emb must be {(batch_size, self.time_emb_dim)}, got {tuple(cur_time_emb.shape)}"
                )
            parts.append(seq_time_emb.unsqueeze(2).expand(-1, -1, num_nodes, -1))
            parts.append(cur_time_emb.unsqueeze(1).unsqueeze(2).expand(-1, input_len, num_nodes, -1))
        if self.node_emb is not None:
            if self.node_emb.shape[0] != num_nodes:
                raise AssertionError(f"node_emb has {self.node_emb.shape[0]} nodes, got input num_nodes={num_nodes}")
            parts.append(self.node_emb.to(x.device, x.dtype).view(1, 1, num_nodes, -1).expand(batch_size, input_len, -1, -1))

        token_input = torch.cat(parts, dim=-1)
        h_self = self.token_encoder(token_input)
        if self.global_mode:
            h_global = h_self.mean(dim=2, keepdim=True).expand(-1, -1, num_nodes, -1)
            env_mu = self.mu_global(h_global)
            env_logvar = self.logvar_global(h_global)
        elif self.use_neighbor and adj_norm is not None:
            h_nei = torch.einsum("ij,bljd->blid", adj_norm.to(h_self.device, h_self.dtype), h_self)
            env_input = torch.cat([h_self, h_nei], dim=-1)
            env_mu = self.mu_nei(env_input)
            env_logvar = self.logvar_nei(env_input)
        else:
            env_mu = self.mu_self(h_self)
            env_logvar = self.logvar_self(h_self)

        env_logvar = env_logvar.clamp(self.logvar_min, self.logvar_max)
        env = self._sample_env(env_mu, env_logvar)

        expected = (batch_size, input_len, num_nodes, self.env_dim)
        for name, tensor in {
            "env_mu_tokens": env_mu,
            "env_logvar_tokens": env_logvar,
            "env_tokens": env,
        }.items():
            if tuple(tensor.shape) != expected:
                raise AssertionError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        return env_mu, env_logvar, env
