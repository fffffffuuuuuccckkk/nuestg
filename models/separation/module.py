from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
from torch import nn

from models.separation.basis_projection import batch_env_basis, project_orthogonal_to_basis
from models.separation.lowrank_residual import hidden_lowrank_residual
from models.separation.orthogonal_projection import cosine_mean, orthogonal_project


class SeparationModule(nn.Module):
    """Computation-level Z/E separation for NUE-STG.

    The module changes the forward computation before prediction. It does not
    add a soft loss. The output z_inv/env are the tensors consumed by invariant
    prediction, environment residual, utility gate, and swap regularization.
    """

    EXTRA_KEYS = [
        "sep_projection_ratio",
        "sep_cos_z_env_before",
        "sep_cos_z_env_after",
        "sep_lowrank_energy_ratio",
        "sep_residual_norm",
        "sep_z_raw_norm",
        "sep_z_inv_norm",
        "sep_env_raw_norm",
        "sep_env_norm",
        "sep_proj_norm",
        "sep_basis_rank",
        "sep_svd_top_singular_mean",
        "sep_lowrank_rank",
        "sep_env_residual_norm",
    ]

    def __init__(
        self,
        cfg: Optional[Dict],
        num_nodes: int,
        z_dim: int,
        env_dim: int,
        input_len: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        del num_nodes, input_len, input_dim
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = str(self.cfg.get("mode", "none"))
        self.z_dim = int(z_dim)
        self.env_dim = int(env_dim)
        self.env_to_z = nn.Linear(env_dim, z_dim)
        self.res_to_env = nn.Linear(z_dim, env_dim)

        basis_cfg = self.cfg.get("basis", {}) or {}
        rank = int(basis_cfg.get("rank", 8))
        self.learnable_basis = nn.Parameter(torch.randn(rank, z_dim) * 0.02)

    def _zero_extra(self, z_raw: torch.Tensor, env_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = z_raw.new_zeros(())
        extra = {key: zero for key in self.EXTRA_KEYS}
        extra["sep_z_raw_norm"] = z_raw.detach().norm(dim=-1).mean()
        extra["sep_z_inv_norm"] = extra["sep_z_raw_norm"]
        extra["sep_env_raw_norm"] = env_raw.detach().norm(dim=-1).mean()
        extra["sep_env_norm"] = extra["sep_env_raw_norm"]
        return extra

    def _finish_extra(
        self,
        extra: Dict[str, torch.Tensor],
        z_raw: torch.Tensor,
        z_inv: torch.Tensor,
        env_raw: torch.Tensor,
        env: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        filled = self._zero_extra(z_raw, env_raw)
        filled.update(extra)
        filled["sep_z_raw_norm"] = z_raw.detach().norm(dim=-1).mean()
        filled["sep_z_inv_norm"] = z_inv.detach().norm(dim=-1).mean()
        filled["sep_env_raw_norm"] = env_raw.detach().norm(dim=-1).mean()
        filled["sep_env_norm"] = env.detach().norm(dim=-1).mean()
        return filled

    def _orthogonal(self, z_raw: torch.Tensor, env_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.get("orthogonal", {}) or {}
        eps = float(cfg.get("eps", 1e-6))
        e_z = self.env_to_z(env_raw)
        out = orthogonal_project(
            z_raw,
            e_z,
            alpha=float(cfg.get("alpha", 1.0)),
            renorm=bool(cfg.get("renorm", True)),
            eps=eps,
        )
        return {"z_inv": out["z_inv"], "env": env_raw, "extra": out["extra"]}

    def _basis(self, z_raw: torch.Tensor, env_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.get("basis", {}) or {}
        source = str(cfg.get("source", "batch_env"))
        rank = int(cfg.get("rank", 8))
        alpha = float(cfg.get("alpha", 1.0))
        eps = float(cfg.get("eps", 1e-5))
        e_z = self.env_to_z(env_raw)

        try:
            if source == "batch_env":
                basis_out = batch_env_basis(e_z, rank=rank, eps=eps)
                basis = basis_out["basis"]
                singular_mean = basis_out["singular_mean"]
            elif source == "learnable":
                q, _ = torch.linalg.qr(self.learnable_basis.transpose(0, 1), mode="reduced")
                safe_rank = min(rank, q.shape[1])
                basis = q.transpose(0, 1)[:safe_rank]
                singular_mean = z_raw.new_zeros(())
            else:
                raise NotImplementedError(
                    f"MODEL.separation.basis.source={source!r} is not implemented; "
                    "expected batch_env or learnable."
                )
            out = project_orthogonal_to_basis(
                z_raw,
                e_z,
                basis=basis,
                alpha=alpha,
                eps=eps,
                singular_mean=singular_mean,
            )
            extra = out["extra"]
            extra["sep_basis_source_is_learnable"] = z_raw.new_tensor(float(source == "learnable"))
            return {"z_inv": out["z_inv"], "env": env_raw, "extra": extra}
        except Exception as exc:
            fallback = str(cfg.get("fallback", "orthogonal_projection"))
            if fallback != "orthogonal_projection":
                raise
            warnings.warn(
                f"basis_projection failed ({exc}); falling back to orthogonal_projection.",
                RuntimeWarning,
            )
            return self._orthogonal(z_raw, env_raw)

    def _lowrank(self, z_raw: torch.Tensor, env_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.get("lowrank", {}) or {}
        target = str(cfg.get("target", "hidden"))
        if target != "hidden":
            raise NotImplementedError(
                "MODEL.separation.lowrank.target='input' is not implemented yet; "
                "use target='hidden' for the current computation-level low-rank split."
            )

        rank = int(cfg.get("rank", 8))
        eps = float(cfg.get("eps", 1e-6))
        try:
            out = hidden_lowrank_residual(z_raw, rank=rank, eps=eps)
            low = out["low"]
            residual = out["residual"]
        except Exception as exc:
            warnings.warn(f"lowrank_residual SVD failed ({exc}); using z_raw/env_raw fallback.", RuntimeWarning)
            extra = {
                "sep_lowrank_rank": z_raw.new_tensor(float(rank)),
                "sep_lowrank_energy_ratio": z_raw.new_zeros(()),
                "sep_residual_norm": z_raw.new_zeros(()),
            }
            return {"z_inv": z_raw, "env": env_raw, "extra": extra}

        residual_to_env = residual.detach() if bool(cfg.get("detach_residual_to_env", False)) else residual
        env_residual = self.res_to_env(residual_to_env)
        env = env_raw + float(cfg.get("env_residual_scale", 1.0)) * env_residual
        e_z = self.env_to_z(env)
        extra = dict(out["extra"])
        extra.update(
            {
                "sep_env_residual_norm": env_residual.norm(dim=-1).mean(),
                "sep_cos_z_env_before": cosine_mean(z_raw, self.env_to_z(env_raw), eps),
                "sep_cos_z_env_after": cosine_mean(low, e_z, eps),
            }
        )
        return {"z_inv": low, "env": env, "extra": extra}

    def forward(
        self,
        x: torch.Tensor,
        z_raw: torch.Tensor,
        env_raw: torch.Tensor,
        y_inv_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        del x, y_inv_raw
        mode = "none" if not self.enabled else self.mode
        if mode in {"none", "", "false"}:
            z_inv = z_raw
            env = env_raw
            extra = self._zero_extra(z_raw, env_raw)
        elif mode == "orthogonal_projection":
            out = self._orthogonal(z_raw, env_raw)
            z_inv, env, extra = out["z_inv"], out["env"], out["extra"]
        elif mode == "basis_projection":
            out = self._basis(z_raw, env_raw)
            z_inv, env, extra = out["z_inv"], out["env"], out["extra"]
        elif mode == "lowrank_residual":
            out = self._lowrank(z_raw, env_raw)
            z_inv, env, extra = out["z_inv"], out["env"], out["extra"]
        else:
            raise NotImplementedError(
                f"MODEL.separation.mode={mode!r} is not implemented; "
                "expected none, orthogonal_projection, basis_projection, or lowrank_residual."
            )

        if tuple(z_inv.shape) != tuple(z_raw.shape):
            raise AssertionError(f"z_inv must match z_raw shape, got {tuple(z_inv.shape)} vs {tuple(z_raw.shape)}")
        if tuple(env.shape) != tuple(env_raw.shape):
            raise AssertionError(f"env must match env_raw shape, got {tuple(env.shape)} vs {tuple(env_raw.shape)}")
        return {
            "z_inv": z_inv,
            "env": env,
            "extra": self._finish_extra(extra, z_raw, z_inv, env_raw, env),
            "mode": mode,
        }
