from __future__ import annotations

from typing import Dict

import torch


def cosine_mean(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    a_norm = a.norm(dim=-1).clamp_min(eps)
    b_norm = b.norm(dim=-1).clamp_min(eps)
    return ((a * b).sum(dim=-1) / (a_norm * b_norm)).mean()


def orthogonal_project(
    z_raw: torch.Tensor,
    e_z: torch.Tensor,
    alpha: float = 1.0,
    renorm: bool = True,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Remove the per-sample component of z_raw that lies along e_z."""
    e_norm = e_z.norm(dim=-1, keepdim=True).clamp_min(eps)
    e_unit = e_z / e_norm
    proj = (z_raw * e_unit).sum(dim=-1, keepdim=True) * e_unit
    z_inv = z_raw - float(alpha) * proj
    if renorm:
        raw_norm = z_raw.norm(dim=-1, keepdim=True)
        inv_norm = z_inv.norm(dim=-1, keepdim=True).clamp_min(eps)
        z_inv = z_inv * (raw_norm / inv_norm)

    proj_norm = proj.norm(dim=-1)
    raw_norm = z_raw.norm(dim=-1)
    inv_norm = z_inv.norm(dim=-1)
    extra = {
        "sep_proj_norm": proj_norm.mean(),
        "sep_projection_ratio": (proj_norm / raw_norm.clamp_min(eps)).mean(),
        "sep_cos_z_env_before": cosine_mean(z_raw, e_z, eps),
        "sep_cos_z_env_after": cosine_mean(z_inv, e_z, eps),
        "sep_z_raw_norm": raw_norm.mean(),
        "sep_z_inv_norm": inv_norm.mean(),
    }
    return {"z_inv": z_inv, "extra": extra}
