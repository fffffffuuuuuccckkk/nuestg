from __future__ import annotations

from typing import Dict, Optional

import torch

from models.separation.orthogonal_projection import cosine_mean


def project_orthogonal_to_basis(
    z_raw: torch.Tensor,
    e_z: torch.Tensor,
    basis: torch.Tensor,
    alpha: float,
    eps: float,
    singular_mean: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Project z_raw away from a shared environment subspace basis."""
    batch_size, num_nodes, z_dim = z_raw.shape
    z_flat = z_raw.reshape(batch_size * num_nodes, z_dim)
    coeff = z_flat.matmul(basis.transpose(0, 1))
    proj_flat = coeff.matmul(basis)
    z_inv = (z_flat - float(alpha) * proj_flat).reshape_as(z_raw)

    proj_norm = proj_flat.norm(dim=-1)
    raw_norm = z_flat.norm(dim=-1)
    extra = {
        "sep_basis_rank": z_raw.new_tensor(float(basis.shape[0])),
        "sep_projection_ratio": (proj_norm / raw_norm.clamp_min(eps)).mean(),
        "sep_cos_z_env_before": cosine_mean(z_raw, e_z, eps),
        "sep_cos_z_env_after": cosine_mean(z_inv, e_z, eps),
        "sep_z_raw_norm": raw_norm.mean(),
        "sep_z_inv_norm": z_inv.norm(dim=-1).mean(),
        "sep_svd_top_singular_mean": (
            singular_mean if singular_mean is not None else z_raw.new_zeros(())
        ),
    }
    return {"z_inv": z_inv, "extra": extra}


def batch_env_basis(e_z: torch.Tensor, rank: int, eps: float) -> Dict[str, torch.Tensor]:
    """Build an environment subspace from centered batch-node env directions."""
    flat = e_z.reshape(-1, e_z.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    safe_rank = min(int(rank), vh.shape[0], vh.shape[1])
    if safe_rank <= 0:
        raise RuntimeError("batch_env_basis rank became zero")
    basis = vh[:safe_rank]
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(eps)
    singular_mean = singular_values[:safe_rank].mean() if singular_values.numel() else flat.new_zeros(())
    return {"basis": basis, "singular_mean": singular_mean}
