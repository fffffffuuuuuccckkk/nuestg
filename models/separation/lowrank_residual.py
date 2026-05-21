from __future__ import annotations

from typing import Dict

import torch


def hidden_lowrank_residual(
    z_raw: torch.Tensor,
    rank: int,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Per-batch low-rank decomposition of hidden node-feature matrices."""
    lows = []
    residuals = []
    energy_ratios = []
    for z_b in z_raw:
        u, s, vh = torch.linalg.svd(z_b, full_matrices=False)
        safe_rank = min(int(rank), s.shape[0])
        if safe_rank <= 0:
            low = torch.zeros_like(z_b)
            energy_ratio = z_b.new_zeros(())
        else:
            low = (u[:, :safe_rank] * s[:safe_rank]).matmul(vh[:safe_rank])
            energy_ratio = s[:safe_rank].pow(2).sum() / s.pow(2).sum().clamp_min(eps)
        res = z_b - low
        lows.append(low)
        residuals.append(res)
        energy_ratios.append(energy_ratio)

    low = torch.stack(lows, dim=0)
    residual = torch.stack(residuals, dim=0)
    energy_ratio = torch.stack(energy_ratios).mean()
    extra = {
        "sep_lowrank_rank": z_raw.new_tensor(float(rank)),
        "sep_lowrank_energy_ratio": energy_ratio,
        "sep_residual_norm": residual.norm(dim=-1).mean(),
        "sep_z_inv_norm": low.norm(dim=-1).mean(),
    }
    return {"low": low, "residual": residual, "extra": extra}
