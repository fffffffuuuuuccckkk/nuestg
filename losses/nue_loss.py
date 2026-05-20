from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from basicts.metrics import masked_mae as basicts_masked_mae
except Exception:  # pragma: no cover - fallback for unusual installs
    basicts_masked_mae = None

from utils.tensor_ops import align_target, make_valid_mask, masked_abs_error, masked_mean


@dataclass
class NUESTGLossConfig:
    lambda_inv: float = 0.2
    lambda_gate: float = 0.1
    lambda_swap: float = 0.1
    lambda_swap_same: float = 0.05
    lambda_kl: float = 1e-4
    lambda_ind: float = 1e-3
    lambda_sparse: float = 1e-3
    gate_eta: float = 0.0
    gate_tau: float = 0.1
    swap_margin: float = 0.01
    null_val: Optional[float] = 0.0


class NUESTGLoss(nn.Module):
    """NUE-STG loss with node-wise utility-aware environment regularization.

    The gate target approximates conditional utility with prediction gain:

        Delta = loss(f_inv(Z), Y) - loss(f_inv(Z) + rho * R_env(Z,E), Y)

    A node-step gate should open when Delta exceeds a cost eta. KL constrains
    I(E;X), the Z/E covariance penalty discourages redundant information, and
    the sparse penalty prevents rho from staying open everywhere.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = NUESTGLossConfig(**kwargs)
        self.latest_log_dict: Dict[str, float] = {}

    def _mae_loss(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        abs_error, mask = masked_abs_error(prediction, targets, self.cfg.null_val, targets_mask)
        return masked_mean(abs_error, mask)

    def _channel_mean_error(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = align_target(targets, prediction)
        mask = make_valid_mask(targets, self.cfg.null_val, targets_mask)
        abs_error = (prediction - torch.nan_to_num(targets, nan=0.0)).abs()
        valid_counts = mask.to(abs_error.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        elem = (abs_error * mask.to(abs_error.dtype)).sum(dim=-1, keepdim=True) / valid_counts
        elem_mask = mask.any(dim=-1, keepdim=True)
        return elem, elem_mask

    @staticmethod
    def _independence_loss(z_inv: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        z = z_inv.reshape(-1, z_inv.shape[-1])
        e = env.reshape(-1, env.shape[-1])
        if z.shape[0] <= 1:
            return z.new_zeros(())
        z = z - z.mean(dim=0, keepdim=True)
        e = e - e.mean(dim=0, keepdim=True)
        cross_cov = z.transpose(0, 1).matmul(e) / (z.shape[0] - 1)
        return cross_cov.pow(2).mean()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        y_inv = output["y_inv"]
        rho = output["rho"]

        pred_loss = self._mae_loss(prediction, y_true, targets_mask)
        inv_loss = self._mae_loss(y_inv, y_true, targets_mask)

        inv_elem, elem_mask = self._channel_mean_error(y_inv, y_true, targets_mask)
        full_elem, _ = self._channel_mean_error(prediction, y_true, targets_mask)
        delta_gain = inv_elem - full_elem
        s_gain = torch.sigmoid((delta_gain - self.cfg.gate_eta) / max(self.cfg.gate_tau, 1e-6)).detach()
        gate_loss_elem = F.binary_cross_entropy(rho.clamp(1e-6, 1.0 - 1e-6), s_gain, reduction="none")
        gate_loss = masked_mean(gate_loss_elem, elem_mask)

        env_mu = output["env_mu"]
        env_logvar = output["env_logvar"]
        kl_loss = (-0.5 * (1 + env_logvar - env_mu.pow(2) - env_logvar.exp())).mean()
        ind_loss = self._independence_loss(output["z_inv"], output["env"])
        sparse_loss = rho.mean()

        swap_loss = prediction.new_zeros(())
        if "prediction_swap" in output:
            y_swap = output["prediction_swap"]
            swap_elem, _ = self._channel_mean_error(y_swap, y_true, targets_mask)
            swap_diff = F.relu(self.cfg.swap_margin + full_elem.detach() - swap_elem) * s_gain
            swap_diff = masked_mean(swap_diff, elem_mask)
            swap_same = (1.0 - s_gain) * (y_swap - prediction.detach()).abs().mean(dim=-1, keepdim=True)
            swap_same = masked_mean(swap_same, elem_mask)
            swap_loss = swap_diff + self.cfg.lambda_swap_same * swap_same

        total_loss = (
            pred_loss
            + self.cfg.lambda_inv * inv_loss
            + self.cfg.lambda_gate * gate_loss
            + self.cfg.lambda_swap * swap_loss
            + self.cfg.lambda_kl * kl_loss
            + self.cfg.lambda_ind * ind_loss
            + self.cfg.lambda_sparse * sparse_loss
        )

        log_dict = {
            "pred_loss": pred_loss.detach(),
            "inv_loss": inv_loss.detach(),
            "gate_loss": gate_loss.detach(),
            "swap_loss": swap_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "ind_loss": ind_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "rho_mean": rho.detach().mean(),
            "rho_std": rho.detach().std(unbiased=False),
            "delta_gain_mean": masked_mean(delta_gain.detach(), elem_mask),
            "total_loss": total_loss.detach(),
        }
        self.latest_log_dict = {k: float(v.cpu()) for k, v in log_dict.items()}
        return total_loss, log_dict


def nue_mae_metric(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
    targets = align_target(targets, prediction)
    if basicts_masked_mae is not None and targets.shape == prediction.shape:
        return basicts_masked_mae(prediction, targets, targets_mask)
    abs_error, mask = masked_abs_error(prediction, targets, None, targets_mask)
    return masked_mean(abs_error, mask)


def make_basicts_loss(loss_cfg: Dict[str, float]):
    loss_module = NUESTGLoss(**loss_cfg)

    def basicts_nue_loss(
        prediction: torch.Tensor,
        targets: torch.Tensor,
        y_inv: torch.Tensor,
        r_env: torch.Tensor,
        rho: torch.Tensor,
        z_inv: torch.Tensor,
        env_mu: torch.Tensor,
        env_logvar: torch.Tensor,
        env: torch.Tensor,
        prediction_swap: torch.Tensor = None,
        rho_swap: torch.Tensor = None,
        targets_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        output = {
            "prediction": prediction,
            "y_inv": y_inv,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "env_mu": env_mu,
            "env_logvar": env_logvar,
            "env": env,
        }
        if prediction_swap is not None:
            output["prediction_swap"] = prediction_swap
        if rho_swap is not None:
            output["rho_swap"] = rho_swap
        loss, _ = loss_module(output, targets, targets_mask)
        return loss

    basicts_nue_loss.__name__ = "nue_stg_loss"
    basicts_nue_loss.loss_module = loss_module
    return basicts_nue_loss
