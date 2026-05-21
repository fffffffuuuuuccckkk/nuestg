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
    loss_type: str = "mae"
    use_masked_mae: bool = True
    null_val: Optional[float] = None
    lambda_pred: float = 1.0
    use_inv: bool = True
    lambda_inv: float = 0.2
    use_gate: bool = True
    lambda_gate: float = 0.1
    gate_label_mode: str = "potential_gain"
    gate_eta: float = 0.0
    gate_tau: float = 0.1
    gate_bce_pos_weight: Optional[float] = None
    use_swap: bool = True
    lambda_swap: float = 0.1
    lambda_swap_diff: float = 1.0
    lambda_swap_same: float = 0.05
    swap_margin: float = 0.01
    swap_weight_mode: str = "sgain"
    swap_detach_inv: bool = True
    swap_detach_full: bool = True
    use_kl: bool = True
    lambda_kl: float = 1e-4
    kl_warmup_epochs: int = 5
    kl_free_bits: float = 0.0
    use_ind: bool = True
    lambda_ind: float = 1e-3
    ind_type: str = "cross_cov"
    use_sparse: bool = True
    lambda_sparse: float = 1e-3
    sparse_target: Optional[float] = None
    use_entropy: bool = False
    lambda_entropy: float = 0.0
    entropy_mode: str = "minimize"
    use_residual_norm: bool = False
    lambda_residual_norm: float = 0.0
    use_env_consistency: bool = False
    lambda_env_consistency: float = 0.0


class NUESTGLoss(nn.Module):
    """NUE-STG loss with node-wise utility-aware environment regularization.

    NUE-STG learns node-wise conditional environment utility. For each node v
    and time window t, the local environment E_{v,t} is considered useful only
    if it provides additional predictive information beyond invariant
    representation Z_{v,t}:

        I(Y_{v,t+h}; E_{v,t} | Z_{v,t}) > eta.

    Since this mutual information is hard to estimate, we approximate utility
    by potential prediction gain:

        Delta = loss(y_inv, Y) - loss(y_inv + r_env, Y).

    The gate rho_{v,t,h} is trained to open when Delta exceeds a usage cost eta.
    KL bottleneck limits I(E;X), covariance penalty reduces redundancy I(E;Z),
    sparse penalty prevents always using E, and counterfactual environment
    swapping regularizes whether replacing E changes prediction.
    """

    LOG_KEYS = [
        "total_loss",
        "pred_loss",
        "inv_loss",
        "gate_loss",
        "swap_loss",
        "swap_diff_loss",
        "swap_same_loss",
        "kl_loss",
        "effective_lambda_kl",
        "ind_loss",
        "sparse_loss",
        "entropy_loss",
        "residual_norm_loss",
        "env_consistency_loss",
        "rho_mean",
        "rho_std",
        "rho_min",
        "rho_max",
        "rho_entropy",
        "delta_gain_mean",
        "delta_gain_pos_ratio",
        "potential_gain_mean",
        "swap_delta_mean",
        "env_mu_abs_mean",
        "env_std_mean",
        "r_env_abs_mean",
        "y_inv_mae",
        "y_potential_mae",
        "y_hat_mae",
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = NUESTGLossConfig(**kwargs)
        self.epoch = 0
        self.latest_log_dict: Dict[str, float] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _zero(self, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros(())

    def _mae_loss(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.loss_type != "mae":
            raise ValueError(f"Only loss_type='mae' is implemented, got {self.cfg.loss_type!r}")
        abs_error, mask = masked_abs_error(prediction, targets, self.cfg.null_val, targets_mask)
        return masked_mean(abs_error, mask if self.cfg.use_masked_mae else None)

    def _channel_mean_error(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = align_target(targets, prediction)
        mask = make_valid_mask(targets, self.cfg.null_val, targets_mask)
        abs_error = (prediction - torch.nan_to_num(targets, nan=0.0)).abs()
        if not self.cfg.use_masked_mae:
            return abs_error.mean(dim=-1, keepdim=True), torch.ones_like(abs_error[..., :1], dtype=torch.bool)
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
        cross_cov = z.transpose(0, 1).matmul(e) / max(z.shape[0] - 1, 1)
        return cross_cov.pow(2).mean()

    def _bce_gate_loss(self, rho: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rho = rho.clamp(1e-6, 1.0 - 1e-6)
        if self.cfg.gate_bce_pos_weight is None:
            elem = F.binary_cross_entropy(rho, target, reduction="none")
        else:
            pos_weight = float(self.cfg.gate_bce_pos_weight)
            elem = -(pos_weight * target * torch.log(rho) + (1.0 - target) * torch.log(1.0 - rho))
        return masked_mean(elem, mask)

    def _kl_loss(self, env_mu: torch.Tensor, env_logvar: torch.Tensor) -> torch.Tensor:
        kl_elem = -0.5 * (1 + env_logvar - env_mu.pow(2) - env_logvar.exp())
        if self.cfg.kl_free_bits and self.cfg.kl_free_bits > 0:
            kl_elem = torch.clamp(kl_elem, min=float(self.cfg.kl_free_bits))
        return kl_elem.mean()

    def _effective_lambda_kl(self) -> float:
        if not self.cfg.use_kl or self.cfg.lambda_kl == 0:
            return 0.0
        warmup = int(self.cfg.kl_warmup_epochs or 0)
        if warmup <= 0:
            return float(self.cfg.lambda_kl)
        return float(self.cfg.lambda_kl) * min(1.0, max(self.epoch, 0) / warmup)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        y_inv = output["y_inv"]
        y_potential = output["y_potential"]
        r_env = output["r_env"]
        rho = output["rho"]
        zero = self._zero(prediction)

        pred_loss_raw = self._mae_loss(prediction, y_true, targets_mask)
        inv_loss_raw = self._mae_loss(y_inv, y_true, targets_mask)
        potential_loss_raw = self._mae_loss(y_potential, y_true, targets_mask)

        inv_elem, elem_mask = self._channel_mean_error(y_inv, y_true, targets_mask)
        potential_elem, _ = self._channel_mean_error(y_potential, y_true, targets_mask)
        full_elem, _ = self._channel_mean_error(prediction, y_true, targets_mask)

        delta_gain = inv_elem - potential_elem
        s_gain = torch.sigmoid((delta_gain - self.cfg.gate_eta) / max(self.cfg.gate_tau, 1e-6)).detach()

        gate_loss_raw = self._bce_gate_loss(rho, s_gain, elem_mask)
        env_mu = output["env_mu"]
        env_logvar = output["env_logvar"]
        env = output["env"]
        kl_loss_raw = self._kl_loss(env_mu, env_logvar)
        ind_loss_raw = self._independence_loss(output["z_inv"], env)
        sparse_loss_raw = rho.mean()
        if self.cfg.sparse_target is not None:
            sparse_loss_raw = (sparse_loss_raw - float(self.cfg.sparse_target)).abs()

        rho_clamped = rho.clamp(1e-6, 1.0 - 1e-6)
        rho_entropy = -(rho_clamped * rho_clamped.log() + (1.0 - rho_clamped) * (1.0 - rho_clamped).log())
        entropy_loss_raw = rho_entropy.mean()
        if self.cfg.entropy_mode == "maximize":
            entropy_loss_raw = -entropy_loss_raw
        residual_norm_loss_raw = r_env.abs().mean()

        swap_diff_loss_raw = zero
        swap_same_loss_raw = zero
        swap_loss_raw = zero
        swap_delta_mean = zero
        prediction_swap = output.get("prediction_swap")
        if prediction_swap is not None:
            swap_elem, _ = self._channel_mean_error(prediction_swap, y_true, targets_mask)
            loss_full_for_swap = full_elem.detach() if self.cfg.swap_detach_full else full_elem
            if self.cfg.swap_weight_mode == "sgain":
                swap_weight = s_gain
            elif self.cfg.swap_weight_mode == "uniform":
                swap_weight = torch.ones_like(s_gain)
            else:
                raise ValueError(f"Unsupported swap_weight_mode={self.cfg.swap_weight_mode!r}")
            swap_diff_loss_raw = masked_mean(
                F.relu(self.cfg.swap_margin + loss_full_for_swap - swap_elem) * swap_weight,
                elem_mask,
            )
            same_target = prediction.detach() if self.cfg.swap_detach_full else prediction
            swap_same_loss_raw = masked_mean(
                (1.0 - s_gain) * (prediction_swap - same_target).abs().mean(dim=-1, keepdim=True),
                elem_mask,
            )
            swap_loss_raw = (
                self.cfg.lambda_swap_diff * swap_diff_loss_raw
                + self.cfg.lambda_swap_same * swap_same_loss_raw
            )
            swap_delta_mean = masked_mean((swap_elem - full_elem).detach(), elem_mask)

        env_consistency_loss_raw = zero
        if output.get("env_perm") is not None:
            env_consistency_loss_raw = (output["env_perm"].detach() - env).abs().mean()

        pred_loss = pred_loss_raw
        inv_loss = inv_loss_raw if self.cfg.use_inv and self.cfg.lambda_inv != 0 else zero
        gate_loss = gate_loss_raw if self.cfg.use_gate and self.cfg.lambda_gate != 0 else zero
        swap_loss = (
            swap_loss_raw
            if self.cfg.use_swap and self.cfg.lambda_swap != 0 and prediction_swap is not None
            else zero
        )
        swap_diff_loss = swap_diff_loss_raw if swap_loss is not zero else zero
        swap_same_loss = swap_same_loss_raw if swap_loss is not zero else zero
        effective_lambda_kl = self._effective_lambda_kl()
        kl_loss = kl_loss_raw if effective_lambda_kl != 0 else zero
        ind_loss = ind_loss_raw if self.cfg.use_ind and self.cfg.lambda_ind != 0 else zero
        sparse_loss = sparse_loss_raw if self.cfg.use_sparse and self.cfg.lambda_sparse != 0 else zero
        entropy_loss = entropy_loss_raw if self.cfg.use_entropy and self.cfg.lambda_entropy != 0 else zero
        residual_norm_loss = (
            residual_norm_loss_raw
            if self.cfg.use_residual_norm and self.cfg.lambda_residual_norm != 0
            else zero
        )
        env_consistency_loss = (
            env_consistency_loss_raw
            if self.cfg.use_env_consistency and self.cfg.lambda_env_consistency != 0
            else zero
        )

        total_loss = (
            self.cfg.lambda_pred * pred_loss
            + self.cfg.lambda_inv * inv_loss
            + self.cfg.lambda_gate * gate_loss
            + self.cfg.lambda_swap * swap_loss
            + effective_lambda_kl * kl_loss
            + self.cfg.lambda_ind * ind_loss
            + self.cfg.lambda_sparse * sparse_loss
            + self.cfg.lambda_entropy * entropy_loss
            + self.cfg.lambda_residual_norm * residual_norm_loss
            + self.cfg.lambda_env_consistency * env_consistency_loss
        )

        logs = {
            "total_loss": total_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "inv_loss": inv_loss.detach(),
            "gate_loss": gate_loss.detach(),
            "swap_loss": swap_loss.detach(),
            "swap_diff_loss": swap_diff_loss.detach(),
            "swap_same_loss": swap_same_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "effective_lambda_kl": prediction.new_tensor(effective_lambda_kl),
            "ind_loss": ind_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "residual_norm_loss": residual_norm_loss.detach(),
            "env_consistency_loss": env_consistency_loss.detach(),
            "rho_mean": rho.detach().mean(),
            "rho_std": rho.detach().std(unbiased=False),
            "rho_min": rho.detach().min(),
            "rho_max": rho.detach().max(),
            "rho_entropy": rho_entropy.detach().mean(),
            "delta_gain_mean": masked_mean(delta_gain.detach(), elem_mask),
            "delta_gain_pos_ratio": masked_mean((delta_gain.detach() > 0).float(), elem_mask),
            "potential_gain_mean": masked_mean(delta_gain.detach(), elem_mask),
            "swap_delta_mean": swap_delta_mean.detach(),
            "env_mu_abs_mean": env_mu.detach().abs().mean(),
            "env_std_mean": torch.exp(0.5 * env_logvar.detach()).mean(),
            "r_env_abs_mean": r_env.detach().abs().mean(),
            "y_inv_mae": inv_loss_raw.detach(),
            "y_potential_mae": potential_loss_raw.detach(),
            "y_hat_mae": pred_loss_raw.detach(),
        }
        self.latest_log_dict = {key: float(value.cpu()) for key, value in logs.items()}
        return total_loss, logs


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
        y_potential: torch.Tensor,
        r_env: torch.Tensor,
        rho: torch.Tensor,
        z_inv: torch.Tensor,
        env_mu: torch.Tensor,
        env_logvar: torch.Tensor,
        env: torch.Tensor,
        prediction_swap: torch.Tensor = None,
        rho_swap: torch.Tensor = None,
        env_perm: torch.Tensor = None,
        targets_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        output = {
            "prediction": prediction,
            "y_inv": y_inv,
            "y_potential": y_potential,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "env_mu": env_mu,
            "env_logvar": env_logvar,
            "env": env,
            "prediction_swap": prediction_swap,
            "rho_swap": rho_swap,
            "env_perm": env_perm,
        }
        loss, _ = loss_module(output, targets, targets_mask)
        return loss

    basicts_nue_loss.__name__ = "nue_stg_loss"
    basicts_nue_loss.loss_module = loss_module
    return basicts_nue_loss
