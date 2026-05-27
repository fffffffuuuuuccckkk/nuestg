from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CLUBEstimator(nn.Module):
    """CLUB upper-bound estimator for I(E; Z).

    q_psi(Z | E) is modeled as a diagonal Gaussian. The upper bound compares
    positive pairs against shuffled negative Z samples over batch-node items.
    """

    def __init__(
        self,
        env_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.z_dim = int(z_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def _params(self, env_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(env_flat)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    @staticmethod
    def _log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((z - mu).pow(2) * torch.exp(-logvar) + logvar).sum(dim=-1)

    def forward(
        self,
        env_hist_bar: torch.Tensor,
        z_inv: torch.Tensor,
        detach_pair: bool = True,
        negative_mode: str = "shuffle",
    ) -> Dict[str, torch.Tensor]:
        if env_hist_bar.dim() != 3:
            raise AssertionError(f"env_hist_bar must be [B,N,D_env], got {tuple(env_hist_bar.shape)}")
        if z_inv.dim() != 3:
            raise AssertionError(f"z_inv must be [B,N,D_z], got {tuple(z_inv.shape)}")
        if env_hist_bar.shape[:2] != z_inv.shape[:2]:
            raise AssertionError(f"E/Z batch-node mismatch: {tuple(env_hist_bar.shape)} vs {tuple(z_inv.shape)}")
        env_flat = env_hist_bar.reshape(-1, env_hist_bar.shape[-1])
        z_flat = z_inv.reshape(-1, z_inv.shape[-1])
        if detach_pair:
            env_for_fit = env_flat.detach()
            z_for_fit = z_flat.detach()
        else:
            env_for_fit = env_flat
            z_for_fit = z_flat
        mu_fit, logvar_fit = self._params(env_for_fit)
        fit_nll = -self._log_prob(z_for_fit, mu_fit, logvar_fit).mean()

        mu, logvar = self._params(env_flat)
        if negative_mode != "shuffle":
            raise ValueError("CLUBEstimator currently supports club_negative_mode='shuffle'")
        if z_flat.shape[0] <= 1:
            club = z_flat.new_zeros(())
        else:
            perm = torch.randperm(z_flat.shape[0], device=z_flat.device)
            same = perm == torch.arange(z_flat.shape[0], device=z_flat.device)
            if same.any():
                perm[same] = (perm[same] + 1) % z_flat.shape[0]
            positive = self._log_prob(z_flat, mu, logvar)
            negative = self._log_prob(z_flat[perm], mu, logvar)
            club = (positive - negative).mean()
        return {"club_upper_bound": club, "club_fit_nll": fit_nll}
