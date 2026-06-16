from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from basicts.metrics import masked_mae as basicts_masked_mae
except Exception:  # pragma: no cover - fallback for unusual installs
    basicts_masked_mae = None

from utils.tensor_ops import align_target, make_valid_mask, masked_abs_error, masked_mean
from models.mi_estimators import CLUBEstimator


@dataclass
class NUESTGLossConfig:
    loss_type: str = "mae"
    use_masked_mae: bool = True
    null_val: Optional[float] = None
    mask_value_mode: str = "null_val"
    grad_consensus: Dict = field(default_factory=dict)
    grad_surgery: Dict = field(default_factory=dict)
    z_inv_bottleneck: Dict = field(default_factory=dict)
    use_pseudo_env_heads: bool = False
    pseudo_env_k: int = 3
    pseudo_env_tau: float = 1.0
    pseudo_env_lambda_head: float = 0.0
    pseudo_env_lambda_var: float = 0.0
    pseudo_env_lambda_balance: float = 0.0
    pseudo_env_lambda_entropy: float = 0.0
    pseudo_env_lambda_diverse: float = 0.0
    pseudo_env_warmup_epochs: int = 0
    pseudo_env_update_interval: int = 1
    pseudo_env_detach_assignment: bool = True
    pseudo_env_use_global_cache: bool = True
    pseudo_env_use_temporal_smoothing: bool = True
    pseudo_env_smooth_radius: int = 2
    pseudo_env_assignment_mode: str = "cached_soft"
    pseudo_env_level: str = "window"
    use_env_routed_inv_heads: bool = False
    env_route_k: int = 3
    env_route_tau: float = 1.0
    env_route_oracle_tau: float = 0.3
    env_route_mode: str = "confidence_mix"
    env_route_replace_final: bool = False
    env_route_lambda_final: float = 1.0
    env_route_lambda_global: float = 0.2
    env_route_lambda_route_soft: float = 0.5
    env_route_lambda_expert: float = 0.2
    env_route_lambda_router_oracle: float = 0.5
    env_route_lambda_inv_rex: float = 0.05
    env_route_inv_rex_use_oracle: bool = True
    env_route_use_z_env_adv: bool = False
    env_route_lambda_z_env_adv: float = 0.01
    env_route_adv_grl_lambda: float = 1.0
    env_route_adv_warmup_epochs: int = 10
    env_route_lambda_balance: float = 0.01
    env_route_lambda_diverse: float = 0.001
    env_route_lambda_entropy: float = 0.0
    env_route_warmup_epochs: int = 5
    env_route_detach_q_for_expert: bool = True
    env_route_use_oracle_weight_for_expert: bool = True
    env_route_alpha_detach: bool = False
    train_loss_scale: str = "normalized"
    warmup_epochs: int = 0
    aux_ramp_epochs: int = 0
    peak_weight_enabled: bool = False
    peak_quantile: float = 0.75
    peak_weight: float = 0.2
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
    swap_warmup_epochs: int = 0
    lambda_swap_diff: float = 1.0
    lambda_swap_same: float = 0.05
    swap_margin: float = 0.01
    swap_weight_mode: str = "sgain"
    swap_detach_inv: bool = True
    swap_detach_full: bool = True
    swap_detach_env: bool = False
    use_kl: bool = True
    lambda_kl: float = 1e-4
    kl_warmup_epochs: int = 5
    kl_free_bits: float = 0.0
    use_ind: bool = True
    lambda_ind: float = 1e-3
    sep_warmup_epochs: int = 0
    ind_type: str = "cross_cov"
    sep_mi_type: str = "cross_cov"
    sep_use_full_env: bool = True
    sep_proj_dim: int = 32
    z_dim: int = 0
    env_dim: int = 0
    lambda_sep: Optional[float] = None
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
    lambda_z_cons: float = 0.0
    lambda_y_cons: float = 0.0
    consistency_detach_target: bool = True
    consistency_loss: str = "mse"
    use_persistence_mi: bool = True
    lambda_persistence_mi: float = 0.05
    persistence_tau: float = 0.2
    persistence_margin: float = 0.0
    persistence_affects_gate: bool = True
    persistence_warmup_epochs: int = 5
    detach_future_env: bool = True
    use_envpred: bool = False
    lambda_envpred: float = 0.0
    envpred_loss_type: str = "mse"
    use_future_mi: bool = False
    lambda_future_mi: float = 0.0
    future_mi_warmup_epochs: int = 0
    future_mi_type: str = "ba_nll"
    future_mi_detach_target: bool = True
    future_mi_infonce_tau: float = 0.2
    future_mi_tau: float = 0.2
    infonce_granularity: str = "token"
    use_rank: bool = False
    lambda_rank: float = 0.0
    rank_margin: float = 0.1
    use_mask_sparse: bool = False
    lambda_mask_sparse: Optional[float] = None
    mask_sparse_warmup_epochs: int = 0
    use_backbone_aux: bool = True
    lambda_backbone_aux: float = 1.0
    use_club: bool = False
    lambda_club: float = 1e-3
    lambda_club_fit: float = 1.0
    club_separate_update: bool = False
    club_detach_pair: bool = True
    club_negative_mode: str = "shuffle"
    club_hidden_dim: int = 64
    hsic_kernel: str = "rbf"
    hsic_sample_size: int = 1024


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
        "effective_lambda_envpred",
        "effective_lambda_future_mi",
        "effective_lambda_swap",
        "effective_lambda_sep",
        "effective_lambda_mask_sparse",
        "aux_schedule_factor",
        "ind_loss",
        "sparse_loss",
        "entropy_loss",
        "residual_norm_loss",
        "env_consistency_loss",
        "z_cons_loss",
        "y_cons_loss",
        "effective_lambda_z_cons",
        "effective_lambda_y_cons",
        "persistence_mi_loss",
        "envpred_loss",
        "future_mi_loss",
        "future_mi_type",
        "env_fut_nll",
        "env_fut_nll_plus",
        "env_fut_nll_minus",
        "env_fut_kl",
        "pred_fut_logvar_mean",
        "pred_fut_mu_norm",
        "future_mi_valid",
        "sep_loss",
        "sep_mi_type",
        "club_upper_bound",
        "club_fit_nll",
        "cross_cov_loss",
        "hsic_loss",
        "rank_loss",
        "mask_sparse_loss",
        "effective_lambda_persistence_mi",
        "rho_mean",
        "rho_std",
        "rho_min",
        "rho_max",
        "rho_entropy",
        "delta_gain_mean",
        "delta_gain_std",
        "delta_gain_pos_ratio",
        "s_gain_mean",
        "persist_score_mean",
        "persist_score_std",
        "s_persist_mean",
        "s_gate_mean",
        "persistence_valid",
        "potential_gain_mean",
        "swap_delta_mean",
        "swap_weight_mean",
        "env_mu_abs_mean",
        "env_std_mean",
        "r_env_abs_mean",
        "y_inv_mae",
        "y_potential_mae",
        "y_hat_mae",
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
        "mask_mean",
        "mask_std",
        "mask_min",
        "mask_max",
        "mask_entropy",
        "mask_active_ratio",
        "env_plus_norm",
        "env_minus_norm",
        "env_hist_norm",
        "env_fut_norm",
        "env_fut_pred_norm",
        "fusion_gamma_abs_mean",
        "fusion_beta_abs_mean",
        "timestamp_valid",
        "cur_time_emb_norm",
        "seq_time_emb_norm",
        "future_time_emb_norm",
        "backbone_aux_loss",
        "cast_vq_loss",
        "cast_commit_loss",
        "cast_mi_loss",
        "stone_graph_perturb_loss",
        "stone_spatial_graph_entropy",
        "stone_temporal_graph_entropy",
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = NUESTGLossConfig(**kwargs)
        if self.cfg.gate_label_mode != "potential_gain":
            raise NotImplementedError(
                "NUE-STG currently supports only LOSS.gate_label_mode='potential_gain'. "
                "Gate targets must be computed from y_potential = y_inv + r_env, not gated prediction."
        )
        self.epoch = 0
        self.latest_log_dict: Dict[str, float] = {}
        self._pseudo_env_warned_collapse = False
        self.sep_z_proj = (
            nn.Linear(self.cfg.z_dim, self.cfg.sep_proj_dim, bias=False)
            if self.cfg.z_dim > 0 and self.cfg.sep_proj_dim > 0
            else None
        )
        self.sep_e_proj = (
            nn.Linear(self.cfg.env_dim, self.cfg.sep_proj_dim, bias=False)
            if self.cfg.env_dim > 0 and self.cfg.sep_proj_dim > 0
            else None
        )
        self.club_estimator = (
            CLUBEstimator(self.cfg.env_dim, self.cfg.z_dim, hidden_dim=self.cfg.club_hidden_dim)
            if self.cfg.env_dim > 0 and self.cfg.z_dim > 0
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _zero(self, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros(())

    def _forecast_pair(
        self,
        prediction: torch.Tensor,
        y_true: torch.Tensor,
        raw_y_true: Optional[torch.Tensor] = None,
        data_scaler=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = str(self.cfg.train_loss_scale or "normalized").lower()
        if scale == "normalized":
            return prediction, y_true
        if scale != "original":
            raise ValueError("LOSS.train_loss_scale must be 'normalized' or 'original'")
        if raw_y_true is None or data_scaler is None:
            raise ValueError(
                "LOSS.train_loss_scale='original' requires raw_y_true and data_scaler "
                "to be passed from the training loop."
            )
        return data_scaler.inverse_transform(prediction), raw_y_true

    def _peak_weight_mask(self, targets: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.cfg.peak_weight_enabled or float(self.cfg.peak_weight) == 0.0:
            return None
        valid_values = targets.detach()[mask]
        if valid_values.numel() == 0:
            return None
        quantile = float(self.cfg.peak_quantile)
        quantile = min(max(quantile, 0.0), 1.0)
        threshold = torch.quantile(valid_values.float(), quantile).to(device=targets.device, dtype=targets.dtype)
        return 1.0 + float(self.cfg.peak_weight) * (targets >= threshold).to(dtype=targets.dtype)

    def _mae_loss(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.loss_type != "mae":
            raise ValueError(f"Only loss_type='mae' is implemented, got {self.cfg.loss_type!r}")
        null_val = None if targets_mask is not None else self.cfg.null_val
        abs_error, mask = masked_abs_error(prediction, targets, null_val, targets_mask)
        peak_weight = self._peak_weight_mask(align_target(targets, prediction), mask)
        if peak_weight is not None:
            abs_error = abs_error * peak_weight
        return masked_mean(abs_error, mask if self.cfg.use_masked_mae else None)

    def _channel_mean_error(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = align_target(targets, prediction)
        null_val = None if targets_mask is not None else self.cfg.null_val
        mask = make_valid_mask(targets, null_val, targets_mask)
        abs_error = (prediction - torch.nan_to_num(targets, nan=0.0)).abs()
        peak_weight = self._peak_weight_mask(targets, mask)
        if peak_weight is not None:
            abs_error = abs_error * peak_weight
        if not self.cfg.use_masked_mae:
            return abs_error.mean(dim=-1, keepdim=True), torch.ones_like(abs_error[..., :1], dtype=torch.bool)
        valid_counts = mask.to(abs_error.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        elem = (abs_error * mask.to(abs_error.dtype)).sum(dim=-1, keepdim=True) / valid_counts
        elem_mask = mask.any(dim=-1, keepdim=True)
        return elem, elem_mask

    @staticmethod
    def _ensure_bhnc(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(-1)
        if tensor.dim() == 4:
            return tensor
        raise AssertionError(f"{name} must be [B,H,N] or [B,H,N,C], got {tuple(tensor.shape)}")

    def _pseudo_env_head_losses(
        self,
        head_pred: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if head_pred.dim() != 5:
            raise AssertionError(f"pseudo_env_head_pred must be [B,K,H,N,C], got {tuple(head_pred.shape)}")
        targets = self._ensure_bhnc(targets, "pseudo_env_targets")
        horizon = head_pred.shape[2]
        if targets.shape[1] != horizon:
            targets = targets[:, :horizon]
        if tuple(targets.shape) != (
            head_pred.shape[0],
            horizon,
            head_pred.shape[3],
            head_pred.shape[4],
        ):
            raise AssertionError(
                f"pseudo-env target shape must align to {(head_pred.shape[0], horizon, head_pred.shape[3], head_pred.shape[4])}, "
                f"got {tuple(targets.shape)}"
            )
        if targets_mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)
            if mask.shape[1] != horizon:
                mask = mask[:, :horizon]
            mask = mask.to(device=head_pred.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(targets.shape):
                mask = mask.expand_as(targets)
        abs_error = (head_pred - targets.unsqueeze(1)).abs()
        mask_5d = mask.unsqueeze(1).expand_as(abs_error)
        if str(self.cfg.pseudo_env_level).lower() == "node":
            counts = mask_5d.to(abs_error.dtype).sum(dim=(2, 4)).clamp_min(1.0)
            loss_node = (abs_error * mask_5d.to(abs_error.dtype)).sum(dim=(2, 4)) / counts
            valid_node = mask_5d.any(dim=4).any(dim=2)
            return loss_node.permute(0, 2, 1).contiguous(), valid_node.permute(0, 2, 1).contiguous(), mask
        counts = mask_5d.to(abs_error.dtype).sum(dim=(2, 3, 4)).clamp_min(1.0)
        loss_window = (abs_error * mask_5d.to(abs_error.dtype)).sum(dim=(2, 3, 4)) / counts
        valid_window = mask_5d.any(dim=4).any(dim=3).any(dim=2)
        return loss_window, valid_window, mask

    def _pseudo_env_inv_losses(
        self,
        y_inv_prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        targets = self._ensure_bhnc(targets, "pseudo_env_inv_targets")
        if targets.shape[1] != y_inv_prediction.shape[1]:
            targets = targets[:, : y_inv_prediction.shape[1]]
        if targets_mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)
            if mask.shape[1] != y_inv_prediction.shape[1]:
                mask = mask[:, : y_inv_prediction.shape[1]]
            mask = mask.to(device=y_inv_prediction.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(targets.shape):
                mask = mask.expand_as(targets)
        abs_error = (y_inv_prediction - targets).abs()
        if str(self.cfg.pseudo_env_level).lower() == "node":
            counts = mask.to(abs_error.dtype).sum(dim=(1, 3)).clamp_min(1.0)
            return (abs_error * mask.to(abs_error.dtype)).sum(dim=(1, 3)) / counts
        counts = mask.to(abs_error.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
        return (abs_error * mask.to(abs_error.dtype)).sum(dim=(1, 2, 3)) / counts

    def _pseudo_env_terms(
        self,
        output: Dict[str, torch.Tensor],
        y_loss_view: torch.Tensor,
        y_inv_loss_view: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
        data_scaler=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        zero = self._zero(prediction)
        if not bool(self.cfg.use_pseudo_env_heads):
            return zero, {}
        logs: Dict[str, torch.Tensor] = {
            "pseudo_env/enabled": prediction.new_tensor(1.0),
            "pseudo_env/cache_updated": prediction.new_tensor(float(bool(output.get("pseudo_env_cache_updated", False)))),
            "pseudo_env/smoothing_enabled": prediction.new_tensor(float(bool(output.get("pseudo_env_smoothing_enabled", False)))),
        }
        k = int(self.cfg.pseudo_env_k)
        if self.epoch < int(self.cfg.pseudo_env_warmup_epochs):
            logs.update({
                "pseudo_env/head_loss": zero,
                "pseudo_env/var_loss": zero,
                "pseudo_env/balance_loss": zero,
                "pseudo_env/entropy": zero,
                "pseudo_env/q_entropy": zero,
                "pseudo_env/diverse_loss": zero,
                "pseudo_env/q_max_mean": zero,
                "pseudo_env/r_var": zero,
                "pseudo_env/collapse_warning": zero,
            })
            for idx in range(k):
                logs[f"pseudo_env/count_head_{idx}"] = zero
                logs[f"pseudo_env/per_head_mae_{idx}"] = zero
                logs[f"pseudo_env/risk_head_{idx}"] = zero
            return zero, logs

        head_pred = output.get("pseudo_env_head_pred")
        if head_pred is None:
            raise RuntimeError("LOSS.use_pseudo_env_heads=True requires model output 'pseudo_env_head_pred'.")
        head_pred_loss_view = head_pred
        if str(self.cfg.train_loss_scale or "normalized").lower() == "original":
            if data_scaler is None:
                raise ValueError("Pseudo-env original-scale loss requires data_scaler.")
            head_pred_loss_view = data_scaler.inverse_transform(head_pred)

        level = str(self.cfg.pseudo_env_level or "window").lower()
        if level not in {"window", "node"}:
            raise ValueError("LOSS.pseudo_env_level must be 'window' or 'node'")
        loss_head, valid_assign, _ = self._pseudo_env_head_losses(head_pred_loss_view, y_loss_view, targets_mask)
        q_env = torch.softmax(-loss_head / max(float(self.cfg.pseudo_env_tau), 1e-6), dim=-1)
        q_env = torch.where(valid_assign, q_env, torch.full_like(q_env, 1.0 / max(k, 1)))
        hard_env = q_env.argmax(dim=-1)
        q_weight = output.get("pseudo_env_q_weight")
        if not isinstance(q_weight, torch.Tensor):
            mode = str(self.cfg.pseudo_env_assignment_mode or "cached_soft").lower()
            if mode.endswith("hard") or mode == "hard":
                q_weight = F.one_hot(hard_env, num_classes=k).to(dtype=q_env.dtype, device=q_env.device)
            else:
                q_weight = q_env
        else:
            q_weight = q_weight.to(device=q_env.device, dtype=q_env.dtype)
            if tuple(q_weight.shape) != tuple(q_env.shape):
                raise AssertionError(f"pseudo_env_q_weight shape {tuple(q_weight.shape)} must match q_env {tuple(q_env.shape)}")
        if bool(self.cfg.pseudo_env_detach_assignment):
            q_weight = q_weight.detach()

        weighted_head = (q_weight * loss_head).sum(dim=-1)
        head_loss = weighted_head.mean()
        reduce_dims = (0, 1) if level == "node" else (0,)
        q_mean = q_env.mean(dim=reduce_dims)
        balance_loss = (q_mean - (1.0 / k)).pow(2).mean()
        entropy = -(q_env * (q_env + 1e-8).log()).sum(dim=-1).mean()
        q_max_mean = q_env.max(dim=-1).values.mean()

        flat_pred = head_pred_loss_view.permute(1, 0, 2, 3, 4).reshape(k, -1)
        flat_pred = F.normalize(flat_pred.float(), dim=-1, eps=1e-8)
        if k > 1:
            sim = flat_pred @ flat_pred.t()
            diverse_loss = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)].mean().to(prediction.dtype)
        else:
            diverse_loss = zero

        inv_loss = self._pseudo_env_inv_losses(y_inv_loss_view, y_loss_view, targets_mask)
        if level == "node":
            inv_view = inv_loss.unsqueeze(-1)
            denom = q_weight.sum(dim=(0, 1)).clamp_min(1e-8)
            risk = (q_weight * inv_view).sum(dim=(0, 1)) / denom
            hard_flat = hard_env.reshape(-1)
        else:
            inv_view = inv_loss.unsqueeze(-1)
            denom = q_weight.sum(dim=0).clamp_min(1e-8)
            risk = (q_weight * inv_view).sum(dim=0) / denom
            hard_flat = hard_env.reshape(-1)
        var_loss = risk.var(unbiased=False)
        counts = torch.bincount(hard_flat, minlength=k).to(device=prediction.device, dtype=prediction.dtype)
        per_head_mae = (q_weight * loss_head).sum(dim=reduce_dims) / q_weight.sum(dim=reduce_dims).clamp_min(1e-8)
        collapse_warning = (q_mean.max() > 0.98).to(dtype=prediction.dtype)
        if (
            bool(collapse_warning.detach().cpu().item())
            and float(self.cfg.pseudo_env_lambda_balance) > 0
            and not self._pseudo_env_warned_collapse
        ):
            warnings.warn(
                "Pseudo-env assignment is nearly collapsed to one head; balance loss is active.",
                RuntimeWarning,
            )
            self._pseudo_env_warned_collapse = True

        total = (
            float(self.cfg.pseudo_env_lambda_head) * head_loss
            + float(self.cfg.pseudo_env_lambda_var) * var_loss
            + float(self.cfg.pseudo_env_lambda_balance) * balance_loss
            + float(self.cfg.pseudo_env_lambda_entropy) * entropy
            + float(self.cfg.pseudo_env_lambda_diverse) * diverse_loss
        )
        logs.update({
            "pseudo_env/head_loss": head_loss.detach(),
            "pseudo_env/var_loss": var_loss.detach(),
            "pseudo_env/balance_loss": balance_loss.detach(),
            "pseudo_env/entropy": entropy.detach(),
            "pseudo_env/q_entropy": entropy.detach(),
            "pseudo_env/diverse_loss": diverse_loss.detach(),
            "pseudo_env/q_max_mean": q_max_mean.detach(),
            "pseudo_env/r_var": var_loss.detach(),
            "pseudo_env/collapse_warning": collapse_warning.detach(),
        })
        for idx in range(k):
            logs[f"pseudo_env/count_head_{idx}"] = counts[idx].detach()
            logs[f"pseudo_env/per_head_mae_{idx}"] = per_head_mae[idx].detach()
            logs[f"pseudo_env/risk_head_{idx}"] = risk[idx].detach()
        return total, logs

    def _env_route_head_losses(
        self,
        head_pred: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if head_pred.dim() != 5:
            raise AssertionError(f"y_route_heads must be [B,K,H,N,C], got {tuple(head_pred.shape)}")
        targets = self._ensure_bhnc(targets, "env_route_targets")
        horizon = head_pred.shape[2]
        if targets.shape[1] != horizon:
            targets = targets[:, :horizon]
        expected = (head_pred.shape[0], horizon, head_pred.shape[3], head_pred.shape[4])
        if tuple(targets.shape) != expected:
            raise AssertionError(f"env route target shape must align to {expected}, got {tuple(targets.shape)}")
        if targets_mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)
            if mask.shape[1] != horizon:
                mask = mask[:, :horizon]
            mask = mask.to(device=head_pred.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(targets.shape):
                mask = mask.expand_as(targets)
        abs_error = (head_pred - targets.unsqueeze(1)).abs()
        mask_5d = mask.unsqueeze(1).expand_as(abs_error)
        counts = mask_5d.to(abs_error.dtype).sum(dim=(2, 3, 4)).clamp_min(1.0)
        return (abs_error * mask_5d.to(abs_error.dtype)).sum(dim=(2, 3, 4)) / counts

    def _env_route_sample_losses(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        targets = self._ensure_bhnc(targets, "env_route_sample_targets")
        if targets.shape[1] != prediction.shape[1]:
            targets = targets[:, : prediction.shape[1]]
        if targets_mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = targets_mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)
            if mask.shape[1] != prediction.shape[1]:
                mask = mask[:, : prediction.shape[1]]
            mask = mask.to(device=prediction.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(targets.shape):
                mask = mask.expand_as(targets)
        abs_error = (prediction - targets).abs()
        counts = mask.to(abs_error.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
        return (abs_error * mask.to(abs_error.dtype)).sum(dim=(1, 2, 3)) / counts

    def _env_route_terms(
        self,
        output: Dict[str, torch.Tensor],
        y_loss_view: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
        data_scaler=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        zero = self._zero(prediction)
        if not bool(self.cfg.use_env_routed_inv_heads):
            return zero, {}
        logs: Dict[str, torch.Tensor] = {"env_route/enabled": prediction.new_tensor(1.0)}
        k = int(self.cfg.env_route_k)
        if self.epoch < int(self.cfg.env_route_warmup_epochs):
            logs.update({
                "env_route/final_loss": zero,
                "env_route/global_loss": zero,
                "env_route/route_soft_loss": zero,
                "env_route/expert_loss": zero,
                "env_route/router_oracle_loss": zero,
                "env_route/inv_rex_loss": zero,
                "env_route/z_env_adv_loss": zero,
                "env_route/balance_loss": zero,
                "env_route/diverse_loss": zero,
                "env_route/entropy": zero,
                "env_route/L_final": zero,
                "env_route/L_route_final": zero,
                "env_route/L_global": zero,
                "env_route/L_route_soft": zero,
                "env_route/L_expert": zero,
                "env_route/L_router_oracle": zero,
                "env_route/L_inv_rex": zero,
                "env_route/L_z_env_adv": zero,
                "env_route/L_balance": zero,
                "env_route/L_diverse": zero,
                "env_route/L_entropy": zero,
                "env_route/oracle_tau": zero,
                "env_route/q_entropy": zero,
                "env_route/q_oracle_entropy": zero,
                "env_route/q_oracle_max_mean": zero,
                "env_route/alpha_mean": zero,
                "env_route/alpha_std": zero,
                "env_route/q_max_mean": zero,
                "env_route/z_env_adv_acc": zero,
                "env_route/z_env_adv_entropy": zero,
                "env_route/router_oracle_acc": zero,
                "env_route/y_inv_mae": zero,
                "env_route/y_global_mae": zero,
                "env_route/y_route_mae": zero,
                "env_route/y_route_soft_mae": zero,
                "env_route/y_route_final_mae": zero,
                "env_route/oracle_route_mae": zero,
            })
            for idx in range(k):
                logs[f"env_route/count_head_{idx}"] = zero
                logs[f"env_route/counts_per_head_{idx}"] = zero
                logs[f"env_route/oracle_count_head_{idx}"] = zero
                logs[f"env_route/oracle_counts_per_head_{idx}"] = zero
                logs[f"env_route/per_head_mae_{idx}"] = zero
                logs[f"env_route/inv_risk_head_{idx}"] = zero
            return zero, logs

        required = [
            "y_route_heads",
            "env_route_logits",
            "env_route_q",
            "env_route_entropy",
            "env_route_alpha",
            "y_route_soft",
            "y_route",
            "y_global",
            "y_route_final",
            "y_inv",
        ]
        missing = [key for key in required if not isinstance(output.get(key), torch.Tensor)]
        if missing:
            raise RuntimeError(f"LOSS.use_env_routed_inv_heads=True requires model outputs: {missing}")

        y_heads = output["y_route_heads"]
        y_route_soft = output["y_route_soft"]
        y_route = output["y_route"]
        y_global = output["y_global"]
        y_route_final = output["y_route_final"]
        y_inv = output["y_inv"]
        logits = output["env_route_logits"]
        q = output["env_route_q"]
        route_entropy = output["env_route_entropy"]
        route_alpha = output["env_route_alpha"]
        if tuple(q.shape) != (y_heads.shape[0], k):
            raise AssertionError(f"env_route_q must be [B,K]={y_heads.shape[0], k}, got {tuple(q.shape)}")
        q_row_sum_error = (q.sum(dim=-1) - 1.0).abs().max()
        if torch.isfinite(q_row_sum_error) and float(q_row_sum_error.detach().cpu()) > 1e-4:
            raise AssertionError(f"env_route_q rows must sum to 1, max error={float(q_row_sum_error.detach().cpu()):.6e}")
        if tuple(route_alpha.shape) != (y_heads.shape[0],):
            raise AssertionError(f"env_route_alpha must be [B]={y_heads.shape[0]}, got {tuple(route_alpha.shape)}")
        if tuple(route_entropy.shape) != (y_heads.shape[0],):
            raise AssertionError(f"env_route_entropy must be [B]={y_heads.shape[0]}, got {tuple(route_entropy.shape)}")
        if (route_alpha.detach() < 0).any() or (route_alpha.detach() > 1).any():
            raise AssertionError("env_route_alpha must stay in [0, 1]")

        if str(self.cfg.train_loss_scale or "normalized").lower() == "original":
            if data_scaler is None:
                raise ValueError("Env route original-scale loss requires data_scaler.")
            y_heads_loss_view = data_scaler.inverse_transform(y_heads)
            y_global_loss_view = data_scaler.inverse_transform(y_global)
            y_route_loss_view = data_scaler.inverse_transform(y_route)
            y_route_soft_loss_view = data_scaler.inverse_transform(y_route_soft)
            y_route_final_loss_view = data_scaler.inverse_transform(y_route_final)
            y_inv_loss_view = data_scaler.inverse_transform(y_inv)
        else:
            y_heads_loss_view = y_heads
            y_global_loss_view = y_global
            y_route_loss_view = y_route
            y_route_soft_loss_view = y_route_soft
            y_route_final_loss_view = y_route_final
            y_inv_loss_view = y_inv

        loss_head = self._env_route_head_losses(y_heads_loss_view, y_loss_view, targets_mask)
        global_loss = self._mae_loss(y_global_loss_view, y_loss_view, targets_mask)
        route_final_loss = self._mae_loss(y_route_final_loss_view, y_loss_view, targets_mask)
        route_soft_loss = self._mae_loss(y_route_soft_loss_view, y_loss_view, targets_mask)
        oracle_tau = max(float(self.cfg.env_route_oracle_tau), 1e-6)
        q_oracle = torch.softmax(-loss_head.detach() / oracle_tau, dim=1)
        if bool(self.cfg.env_route_use_oracle_weight_for_expert):
            # Gradient-level assignment is still oracle-free at inference: the
            # detached soft oracle only prevents near-uniform q from training
            # all invariant heads toward the same average target.
            expert_weight = q_oracle
        else:
            expert_weight = q.detach() if bool(self.cfg.env_route_detach_q_for_expert) else q
        expert_loss = (expert_weight * loss_head).sum(dim=-1).mean()
        router_oracle_loss = (
            q_oracle * ((q_oracle + 1e-8).log() - (q + 1e-8).log())
        ).sum(dim=1).mean()
        inv_sample_loss = self._env_route_sample_losses(y_inv_loss_view, y_loss_view, targets_mask)
        rex_weight = q_oracle.detach() if bool(self.cfg.env_route_inv_rex_use_oracle) else q.detach()
        inv_risk = (rex_weight * inv_sample_loss.unsqueeze(1)).sum(dim=0) / rex_weight.sum(dim=0).clamp_min(1e-8)
        inv_rex_loss = inv_risk.var(unbiased=False)
        z_env_adv_loss = zero
        z_env_adv_acc = zero
        z_env_adv_entropy = zero
        z_env_adv_logits = output.get("z_env_adv_logits", None)
        z_env_adv_active = (
            bool(self.cfg.env_route_use_z_env_adv)
            and self.epoch >= int(self.cfg.env_route_adv_warmup_epochs)
        )
        if z_env_adv_active:
            if not isinstance(z_env_adv_logits, torch.Tensor):
                raise RuntimeError(
                    "LOSS.env_route_use_z_env_adv=True requires output['z_env_adv_logits']; "
                    "check MODEL.env_routed_inv_heads.use_z_env_adv."
                )
            if tuple(z_env_adv_logits.shape) != tuple(q_oracle.shape):
                raise AssertionError(
                    f"z_env_adv_logits must match q_oracle shape {tuple(q_oracle.shape)}, "
                    f"got {tuple(z_env_adv_logits.shape)}"
                )
            log_p_adv = F.log_softmax(z_env_adv_logits, dim=1)
            z_env_adv_loss = F.kl_div(log_p_adv, q_oracle.detach(), reduction="batchmean")
            p_adv = log_p_adv.exp()
            z_env_adv_entropy = -(p_adv * log_p_adv).sum(dim=1).mean()
            z_env_adv_acc = (
                z_env_adv_logits.argmax(dim=1) == q_oracle.detach().argmax(dim=1)
            ).to(dtype=prediction.dtype).mean()
        oracle = loss_head.detach().argmin(dim=1)
        q_mean = q.mean(dim=0)
        balance_loss = (q_mean - (1.0 / k)).pow(2).mean()
        entropy = -(q * (q + 1e-8).log()).sum(dim=-1).mean()
        q_max_mean = q.max(dim=-1).values.mean()
        q_oracle_entropy = -(q_oracle * (q_oracle + 1e-8).log()).sum(dim=-1).mean()
        q_oracle_max_mean = q_oracle.max(dim=-1).values.mean()

        flat_pred = y_heads_loss_view.permute(1, 0, 2, 3, 4).reshape(k, -1)
        flat_pred = F.normalize(flat_pred.float(), dim=-1, eps=1e-8)
        if k > 1:
            sim = flat_pred @ flat_pred.t()
            diverse_loss = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)].mean().to(prediction.dtype)
        else:
            diverse_loss = zero

        hard = q.argmax(dim=-1)
        counts = torch.bincount(hard, minlength=k).to(device=prediction.device, dtype=prediction.dtype)
        oracle_counts = torch.bincount(oracle, minlength=k).to(device=prediction.device, dtype=prediction.dtype)
        per_head_mae = loss_head.mean(dim=0)
        router_oracle_acc = (hard == oracle).to(dtype=prediction.dtype).mean()
        oracle_route_mae = loss_head.gather(1, oracle.view(-1, 1)).mean()
        y_inv_mae = inv_sample_loss.mean()
        y_global_mae = self._env_route_sample_losses(y_global_loss_view, y_loss_view, targets_mask).mean()
        y_route_mae = self._env_route_sample_losses(y_route_loss_view, y_loss_view, targets_mask).mean()
        y_route_soft_mae = self._env_route_sample_losses(y_route_soft_loss_view, y_loss_view, targets_mask).mean()
        y_route_final_mae = self._env_route_sample_losses(y_route_final_loss_view, y_loss_view, targets_mask).mean()

        total = (
            float(self.cfg.env_route_lambda_final) * route_final_loss
            + float(self.cfg.env_route_lambda_global) * global_loss
            + float(self.cfg.env_route_lambda_route_soft) * route_soft_loss
            + float(self.cfg.env_route_lambda_expert) * expert_loss
            + float(self.cfg.env_route_lambda_router_oracle) * router_oracle_loss
            + float(self.cfg.env_route_lambda_inv_rex) * inv_rex_loss
            + float(self.cfg.env_route_lambda_z_env_adv) * z_env_adv_loss
            + float(self.cfg.env_route_lambda_balance) * balance_loss
            + float(self.cfg.env_route_lambda_diverse) * diverse_loss
            + float(self.cfg.env_route_lambda_entropy) * entropy
        )
        logs.update({
            "env_route/final_loss": route_final_loss.detach(),
            "env_route/global_loss": global_loss.detach(),
            "env_route/route_soft_loss": route_soft_loss.detach(),
            "env_route/expert_loss": expert_loss.detach(),
            "env_route/router_oracle_loss": router_oracle_loss.detach(),
            "env_route/inv_rex_loss": inv_rex_loss.detach(),
            "env_route/z_env_adv_loss": z_env_adv_loss.detach(),
            "env_route/balance_loss": balance_loss.detach(),
            "env_route/diverse_loss": diverse_loss.detach(),
            "env_route/entropy": entropy.detach(),
            "env_route/L_final": route_final_loss.detach(),
            "env_route/L_route_final": route_final_loss.detach(),
            "env_route/L_global": global_loss.detach(),
            "env_route/L_route_soft": route_soft_loss.detach(),
            "env_route/L_expert": expert_loss.detach(),
            "env_route/L_router_oracle": router_oracle_loss.detach(),
            "env_route/L_inv_rex": inv_rex_loss.detach(),
            "env_route/L_z_env_adv": z_env_adv_loss.detach(),
            "env_route/L_balance": balance_loss.detach(),
            "env_route/L_diverse": diverse_loss.detach(),
            "env_route/L_entropy": entropy.detach(),
            "env_route/oracle_tau": prediction.new_tensor(oracle_tau).detach(),
            "env_route/q_entropy": entropy.detach(),
            "env_route/q_oracle_entropy": q_oracle_entropy.detach(),
            "env_route/q_oracle_max_mean": q_oracle_max_mean.detach(),
            "env_route/alpha_mean": route_alpha.detach().mean(),
            "env_route/alpha_std": route_alpha.detach().std(unbiased=False),
            "env_route/q_max_mean": q_max_mean.detach(),
            "env_route/z_env_adv_acc": z_env_adv_acc.detach(),
            "env_route/z_env_adv_entropy": z_env_adv_entropy.detach(),
            "env_route/router_oracle_acc": router_oracle_acc.detach(),
            "env_route/y_inv_mae": y_inv_mae.detach(),
            "env_route/y_global_mae": y_global_mae.detach(),
            "env_route/y_route_mae": y_route_mae.detach(),
            "env_route/y_route_soft_mae": y_route_soft_mae.detach(),
            "env_route/y_route_final_mae": y_route_final_mae.detach(),
            "env_route/oracle_route_mae": oracle_route_mae.detach(),
        })
        for idx in range(k):
            logs[f"env_route/count_head_{idx}"] = counts[idx].detach()
            logs[f"env_route/counts_per_head_{idx}"] = counts[idx].detach()
            logs[f"env_route/oracle_count_head_{idx}"] = oracle_counts[idx].detach()
            logs[f"env_route/oracle_counts_per_head_{idx}"] = oracle_counts[idx].detach()
            logs[f"env_route/per_head_mae_{idx}"] = per_head_mae[idx].detach()
            logs[f"env_route/inv_risk_head_{idx}"] = inv_risk[idx].detach()
        return total, logs

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
            return float(self.cfg.lambda_kl) * self._aux_schedule_factor()
        return float(self.cfg.lambda_kl) * min(1.0, max(self.epoch, 0) / warmup) * self._aux_schedule_factor()

    def _effective_lambda_persistence_mi(self) -> float:
        if not self.cfg.use_persistence_mi or self.cfg.lambda_persistence_mi == 0:
            return 0.0
        warmup = int(self.cfg.persistence_warmup_epochs or 0)
        if warmup <= 0:
            return float(self.cfg.lambda_persistence_mi)
        return float(self.cfg.lambda_persistence_mi) * min(1.0, max(self.epoch, 0) / warmup)

    def _warmup_factor(self, warmup_epochs: int) -> float:
        warmup = int(warmup_epochs or 0)
        if warmup <= 0:
            return 1.0
        return min(1.0, max(self.epoch - 1, 0) / warmup)

    def _aux_schedule_factor(self) -> float:
        warmup = int(self.cfg.warmup_epochs or 0)
        ramp = int(self.cfg.aux_ramp_epochs or 0)
        if warmup > 0 and self.epoch <= warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, max(self.epoch - warmup, 0) / ramp)

    def _effective_aux_lambda(self, enabled: bool, value: float, warmup_epochs: int) -> float:
        if not enabled or value == 0:
            return 0.0
        return float(value) * self._warmup_factor(warmup_epochs) * self._aux_schedule_factor()

    def _backbone_aux_terms(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = self._zero(like)
        logs = {
            "backbone_aux_loss": zero,
            "cast_vq_loss": zero,
            "cast_commit_loss": zero,
            "cast_mi_loss": zero,
            "stone_graph_perturb_loss": zero,
            "stone_spatial_graph_entropy": zero,
            "stone_temporal_graph_entropy": zero,
        }
        aux_losses = output.get("backbone_aux_losses") or {}
        aux_weights = output.get("backbone_aux_weights") or {}
        total = zero
        for name, value in aux_losses.items():
            if not isinstance(value, torch.Tensor):
                value = like.new_tensor(float(value))
            weight = float(aux_weights.get(name, 1.0))
            total = total + weight * value
            logs[name] = value
        logs["backbone_aux_loss"] = total
        return total, logs

    def _consistency_term(
        self,
        source: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        like: torch.Tensor,
        default: str,
    ) -> torch.Tensor:
        if source is None or target is None:
            return self._zero(like)
        if tuple(source.shape) != tuple(target.shape):
            raise AssertionError(
                f"consistency tensors must share shape, got {tuple(source.shape)} and {tuple(target.shape)}"
            )
        target_view = target.detach() if self.cfg.consistency_detach_target else target
        source_view = torch.nan_to_num(source)
        target_view = torch.nan_to_num(target_view)
        mode = str(self.cfg.consistency_loss or "mse").lower()
        if mode == "cosine":
            source_flat = source_view.reshape(-1, source_view.shape[-1])
            target_flat = target_view.reshape(-1, target_view.shape[-1])
            return (1.0 - F.cosine_similarity(source_flat, target_flat, dim=-1, eps=1e-8)).mean()
        if mode != "mse":
            raise ValueError("LOSS.consistency_loss must be 'mse' or 'cosine'")
        if default == "mae":
            return (source_view - target_view).abs().mean()
        return (source_view - target_view).pow(2).mean()

    def _z_inv_bottleneck_terms(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg.z_inv_bottleneck or {}
        if not bool(cfg.get("enabled", False)):
            return self._zero(like), {}
        kind = str(cfg.get("type", "vib") or "vib").lower()
        beta = float(cfg.get("beta", 1.0e-4))
        free_bits = max(float(cfg.get("kl_free_bits", 0.0)), 0.0)
        zero = self._zero(like)
        type_id = output.get("z_inv_ib_type_id", like.new_tensor({"vib": 1.0, "gaussian_noise": 2.0, "l2_norm": 3.0}.get(kind, 0.0)))
        z_sample_std = output.get("z_inv_ib_z_sample_std", zero)
        z_mu_abs_mean = output.get("z_inv_ib_z_mu_abs_mean", zero)
        z_logvar_mean = output.get("z_inv_ib_z_logvar_mean", zero)

        if kind == "vib":
            kl_raw = output.get("z_inv_ib_kl")
            if kl_raw is None:
                raise RuntimeError("LOSS.z_inv_bottleneck.type='vib' requires model output 'z_inv_ib_kl'.")
            kl_value = F.relu(kl_raw - free_bits)
            loss = beta * kl_value
        elif kind == "gaussian_noise":
            kl_value = zero
            loss = zero
        elif kind == "l2_norm":
            l2_raw = output.get("z_inv_ib_l2")
            if l2_raw is None:
                raise RuntimeError("LOSS.z_inv_bottleneck.type='l2_norm' requires model output 'z_inv_ib_l2'.")
            kl_value = l2_raw
            loss = beta * l2_raw
        else:
            raise ValueError("LOSS.z_inv_bottleneck.type must be one of: vib, gaussian_noise, l2_norm.")

        logs = {
            "loss_z_inv_ib": loss.detach(),
            "z_inv_ib/enabled": like.new_tensor(1.0),
            "z_inv_ib/type": type_id.detach() if isinstance(type_id, torch.Tensor) else like.new_tensor(float(type_id)),
            "z_inv_ib/kl": kl_value.detach(),
            "z_inv_ib/beta": like.new_tensor(beta),
            "z_inv_ib/z_mu_abs_mean": z_mu_abs_mean.detach() if isinstance(z_mu_abs_mean, torch.Tensor) else zero,
            "z_inv_ib/z_logvar_mean": z_logvar_mean.detach() if isinstance(z_logvar_mean, torch.Tensor) else zero,
            "z_inv_ib/z_sample_std": z_sample_std.detach() if isinstance(z_sample_std, torch.Tensor) else zero,
        }
        return loss, logs

    def _persistence_terms(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        q = output.get("persist_q")
        k = output.get("persist_k")
        zero = self._zero(like)
        logs = {
            "persist_score_mean": zero,
            "persist_score_std": zero,
            "s_persist_mean": zero,
            "persistence_valid": zero,
        }
        if q is None or k is None or q.numel() == 0 or k.numel() == 0:
            return zero, None, logs

        q_flat = q.reshape(-1, q.shape[-1])
        k_for_loss = k.detach() if self.cfg.detach_future_env else k
        k_flat = k_for_loss.reshape(-1, k_for_loss.shape[-1])
        if q_flat.shape[0] <= 1 or k_flat.shape[0] <= 1:
            return zero, None, logs
        q_norm = F.normalize(q_flat, dim=-1)
        k_norm = F.normalize(k_flat, dim=-1)
        tau = max(float(self.cfg.persistence_tau), 1e-6)
        logits = q_norm.matmul(k_norm.transpose(0, 1)) / tau
        labels = torch.arange(q_flat.shape[0], device=q_flat.device)
        mi_loss = F.cross_entropy(logits, labels)

        persist_score = (q_norm * k_norm).sum(dim=-1).reshape(q.shape[0], q.shape[1], 1)
        s_persist = torch.sigmoid(
            (persist_score - float(self.cfg.persistence_margin)) / tau
        )
        logs = {
            "persist_score_mean": persist_score.detach().mean(),
            "persist_score_std": persist_score.detach().std(unbiased=False),
            "s_persist_mean": s_persist.detach().mean(),
            "persistence_valid": like.new_tensor(1.0),
        }
        return mi_loss, s_persist, logs

    def _separation_logs(self, output: Dict[str, torch.Tensor], like: torch.Tensor) -> Dict[str, torch.Tensor]:
        extra = output.get("separation_extra") or {}
        keys = [
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
        logs: Dict[str, torch.Tensor] = {}
        for key in keys:
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                logs[key] = value.detach()
            elif value is None:
                logs[key] = like.new_zeros(())
            else:
                logs[key] = like.new_tensor(float(value))
        return logs

    @staticmethod
    def _diag_gaussian_nll(target: torch.Tensor, pred_mu: torch.Tensor, pred_logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((target - pred_mu).pow(2) * torch.exp(-pred_logvar) + pred_logvar)

    @staticmethod
    def _diag_gaussian_kl(
        true_mu: torch.Tensor,
        true_logvar: torch.Tensor,
        pred_mu: torch.Tensor,
        pred_logvar: torch.Tensor,
    ) -> torch.Tensor:
        true_var = true_logvar.exp()
        pred_var = pred_logvar.exp().clamp_min(1e-8)
        return 0.5 * (
            pred_logvar
            - true_logvar
            + (true_var + (true_mu - pred_mu).pow(2)) / pred_var
            - 1.0
        )

    def _future_infonce(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        zero = self._zero(like)
        granularity = str(self.cfg.infonce_granularity)
        if granularity == "token":
            q = output.get("env_plus_tokens")
            k = output.get("env_fut_tokens")
            if q is not None and k is not None and q.shape[:3] == k.shape[:3]:
                q_flat = q.reshape(-1, q.shape[-1])
                k_flat = k.detach().reshape(-1, k.shape[-1])
            else:
                q = output.get("env_plus")
                k = output.get("env_fut")
                if q is None or k is None:
                    return zero
                q_flat = q.reshape(-1, q.shape[-1])
                k_flat = k.detach().reshape(-1, k.shape[-1])
        else:
            q = output.get("env_plus")
            k = output.get("env_fut")
            if q is None or k is None:
                return zero
            q_flat = q.reshape(-1, q.shape[-1])
            k_flat = k.detach().reshape(-1, k.shape[-1])
        if q_flat.shape[0] <= 1:
            return zero
        q_norm = F.normalize(q_flat, dim=-1)
        k_norm = F.normalize(k_flat, dim=-1)
        tau = max(float(self.cfg.future_mi_infonce_tau or self.cfg.future_mi_tau), 1e-6)
        logits = q_norm.matmul(k_norm.transpose(0, 1)) / tau
        labels = torch.arange(q_flat.shape[0], device=q_flat.device)
        return F.cross_entropy(logits, labels)

    def _future_mi_terms(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = self._zero(like)
        type_map = {"ba_nll": 1.0, "ba_kl": 2.0, "mse": 3.0, "infonce": 4.0}
        mi_type = str(self.cfg.future_mi_type or "ba_nll")
        pred_mu = output.get("pred_fut_mu")
        pred_logvar = output.get("pred_fut_logvar")
        env_fut_tokens = output.get("env_fut_tokens")
        env_fut_mu = output.get("env_fut_mu_tokens")
        env_fut_logvar = output.get("env_fut_logvar_tokens")
        logs = {
            "future_mi_type": like.new_tensor(type_map.get(mi_type, 0.0)),
            "env_fut_nll": zero,
            "env_fut_nll_plus": zero,
            "env_fut_nll_minus": zero,
            "env_fut_kl": zero,
            "pred_fut_logvar_mean": zero,
            "pred_fut_mu_norm": zero,
            "future_mi_valid": zero,
        }
        valid_dist = pred_mu is not None and pred_logvar is not None and env_fut_tokens is not None
        if pred_mu is not None:
            logs["pred_fut_mu_norm"] = pred_mu.detach().norm(dim=-1).mean()
        if pred_logvar is not None:
            logs["pred_fut_logvar_mean"] = pred_logvar.detach().mean()

        if mi_type == "infonce":
            loss = self._future_infonce(output, like)
            logs["future_mi_valid"] = like.new_tensor(float(loss.detach().abs().item() > 0.0))
            return loss, logs
        if not valid_dist:
            return zero, logs

        target_tokens = env_fut_tokens.detach() if self.cfg.future_mi_detach_target else env_fut_tokens
        nll = self._diag_gaussian_nll(target_tokens, pred_mu, pred_logvar).mean()
        logs["env_fut_nll"] = nll.detach()
        logs["env_fut_nll_plus"] = nll.detach()
        logs["future_mi_valid"] = like.new_tensor(1.0)
        if mi_type == "ba_nll":
            return nll, logs
        if mi_type == "ba_kl":
            if env_fut_mu is None or env_fut_logvar is None:
                return nll, logs
            true_mu = env_fut_mu.detach() if self.cfg.future_mi_detach_target else env_fut_mu
            true_logvar = env_fut_logvar.detach() if self.cfg.future_mi_detach_target else env_fut_logvar
            kl = self._diag_gaussian_kl(true_mu, true_logvar, pred_mu, pred_logvar).mean()
            logs["env_fut_kl"] = kl.detach()
            return kl, logs
        if mi_type == "mse":
            target = env_fut_mu.detach() if (env_fut_mu is not None and self.cfg.future_mi_detach_target) else (
                env_fut_mu if env_fut_mu is not None else target_tokens
            )
            return F.mse_loss(pred_mu, target), logs
        raise ValueError(f"Unsupported LOSS.future_mi_type={self.cfg.future_mi_type!r}")

    def _project_for_sep(self, z_inv: torch.Tensor, env_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z_inv.reshape(-1, z_inv.shape[-1])
        e = env_hist.reshape(-1, env_hist.shape[-1])
        if self.sep_z_proj is not None:
            z = self.sep_z_proj(z)
        if self.sep_e_proj is not None:
            e = self.sep_e_proj(e)
        return z, e

    def _cross_cov_projected(self, z_inv: torch.Tensor, env_hist: torch.Tensor) -> torch.Tensor:
        z, e = self._project_for_sep(z_inv, env_hist)
        if z.shape[0] <= 1:
            return z.new_zeros(())
        z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        e = (e - e.mean(dim=0, keepdim=True)) / e.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        cross_cov = e.transpose(0, 1).matmul(z) / max(z.shape[0] - 1, 1)
        return cross_cov.pow(2).mean()

    def _hsic_loss(self, z_inv: torch.Tensor, env_hist: torch.Tensor) -> torch.Tensor:
        z, e = self._project_for_sep(z_inv, env_hist)
        if z.shape[0] <= 1:
            return z.new_zeros(())
        sample_size = int(self.cfg.hsic_sample_size or 0)
        if sample_size > 0 and z.shape[0] > sample_size:
            idx = torch.randperm(z.shape[0], device=z.device)[:sample_size]
            z = z[idx]
            e = e[idx]
        z = z - z.mean(dim=0, keepdim=True)
        e = e - e.mean(dim=0, keepdim=True)
        n = z.shape[0]
        if str(self.cfg.hsic_kernel) == "linear":
            kz = z.matmul(z.transpose(0, 1))
            ke = e.matmul(e.transpose(0, 1))
        else:
            dz = torch.cdist(z, z).pow(2)
            de = torch.cdist(e, e).pow(2)
            sig_z = dz.detach().median().clamp_min(1e-6)
            sig_e = de.detach().median().clamp_min(1e-6)
            kz = torch.exp(-dz / (2.0 * sig_z))
            ke = torch.exp(-de / (2.0 * sig_e))
        h = torch.eye(n, device=z.device, dtype=z.dtype) - (1.0 / n)
        return (h.matmul(kz).matmul(h) * h.matmul(ke).matmul(h)).sum() / max((n - 1) ** 2, 1)

    def _sep_terms(
        self,
        output: Dict[str, torch.Tensor],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = self._zero(like)
        type_map = {"cross_cov": 1.0, "club": 2.0, "hsic": 3.0}
        sep_type = str(self.cfg.sep_mi_type or self.cfg.ind_type or "cross_cov")
        env_hist = output.get("env_hist_bar")
        if env_hist is None:
            env_hist = output.get("env_hist")
        if env_hist is None:
            env_hist = output.get("env")
        z_inv = output.get("z_inv")
        logs = {
            "sep_mi_type": like.new_tensor(type_map.get(sep_type, 0.0)),
            "club_upper_bound": zero,
            "club_fit_nll": zero,
            "cross_cov_loss": zero,
            "hsic_loss": zero,
        }
        if env_hist is None or z_inv is None:
            return zero, logs
        cross_cov = self._cross_cov_projected(z_inv, env_hist)
        logs["cross_cov_loss"] = cross_cov.detach()
        if sep_type == "cross_cov":
            return cross_cov, logs
        if sep_type == "club":
            if self.club_estimator is None:
                return cross_cov, logs
            club = self.club_estimator(
                env_hist,
                z_inv,
                detach_pair=bool(self.cfg.club_detach_pair),
                negative_mode=str(self.cfg.club_negative_mode),
            )
            logs["club_upper_bound"] = club["club_upper_bound"].detach()
            logs["club_fit_nll"] = club["club_fit_nll"].detach()
            return club["club_upper_bound"] + float(self.cfg.lambda_club_fit) * club["club_fit_nll"], logs
        if sep_type == "hsic":
            hsic = self._hsic_loss(z_inv, env_hist)
            logs["hsic_loss"] = hsic.detach()
            return hsic, logs
        raise ValueError(f"Unsupported LOSS.sep_mi_type={self.cfg.sep_mi_type!r}")

    def _fpem_swap_weight(
        self,
        output: Dict[str, torch.Tensor],
        full_elem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = str(self.cfg.swap_weight_mode)
        if mode == "uniform":
            weight = torch.ones_like(full_elem)
            return weight, weight.mean()

        if mode == "future_env_diff":
            source = output.get("env_fut")
            if source is None:
                source = output.get("env_plus")
        elif mode == "env_plus_diff":
            source = output.get("env_plus")
        else:
            raise ValueError(f"Unsupported swap_weight_mode={self.cfg.swap_weight_mode!r}")
        if source is None or output.get("env_perm_index") is None:
            weight = torch.ones_like(full_elem)
            return weight, weight.mean()
        batch_size, num_nodes, dim = source.shape
        flat = source.reshape(batch_size * num_nodes, dim)
        perm = output["env_perm_index"].reshape(-1).long()
        paired = flat[perm]
        cosine = F.cosine_similarity(flat.detach(), paired.detach(), dim=-1)
        alpha = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0).reshape(batch_size, 1, num_nodes, 1)
        return alpha.expand_as(full_elem), alpha.mean()

    def _forward_fpem(
        self,
        output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
        raw_y_true: Optional[torch.Tensor] = None,
        data_scaler=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        y_inv = output["y_inv"]
        zero = self._zero(prediction)
        prediction_loss_view, y_loss_view = self._forecast_pair(prediction, y_true, raw_y_true, data_scaler)
        y_inv_loss_view, _ = self._forecast_pair(y_inv, y_true, raw_y_true, data_scaler)

        pred_loss_raw = self._mae_loss(prediction_loss_view, y_loss_view, targets_mask)
        inv_loss_raw = self._mae_loss(y_inv_loss_view, y_loss_view, targets_mask)
        full_elem, elem_mask = self._channel_mean_error(prediction_loss_view, y_loss_view, targets_mask)

        env_fut = output.get("env_fut")
        pred_fut_mu = output.get("pred_fut_mu")
        pred_fut_logvar = output.get("pred_fut_logvar")
        future_mi_loss_raw, future_mi_logs = self._future_mi_terms(output, prediction)
        envpred_loss_raw = future_mi_logs["env_fut_nll"]
        sep_loss_raw, sep_logs = self._sep_terms(output, prediction)
        env_hist = output.get("env_hist_bar")
        if env_hist is None:
            env_hist = output.get("env_hist", output.get("env"))
        env_mu = output["env_mu"]
        env_logvar = output["env_logvar"]
        kl_loss_raw = self._kl_loss(env_mu, env_logvar)

        mask = output.get("mask")
        if mask is None:
            sparse_loss_raw = zero
            mask_mean = zero
            mask_std = zero
            mask_min = zero
            mask_max = zero
            mask_entropy_mean = zero
            mask_active_ratio = zero
        else:
            mask_mean = mask.mean()
            mask_std = mask.detach().std(unbiased=False)
            mask_min = mask.detach().min()
            mask_max = mask.detach().max()
            mask_clamped = mask.clamp(1e-6, 1.0 - 1e-6)
            mask_entropy = -(
                mask_clamped * mask_clamped.log()
                + (1.0 - mask_clamped) * (1.0 - mask_clamped).log()
            )
            mask_entropy_mean = mask_entropy.mean()
            mask_active_ratio = (mask.detach() > 0.5).float().mean()
            sparse_loss_raw = mask_mean
            if self.cfg.sparse_target is not None:
                sparse_loss_raw = (sparse_loss_raw - float(self.cfg.sparse_target)).abs()

        swap_loss_raw = zero
        swap_diff_loss_raw = zero
        swap_same_loss_raw = zero
        swap_delta_mean = zero
        swap_weight_mean = zero
        prediction_swap = output.get("prediction_swap")
        has_swap = self.cfg.use_swap and self.cfg.lambda_swap != 0 and prediction_swap is not None
        if has_swap:
            prediction_swap_loss_view, _ = self._forecast_pair(prediction_swap, y_true, raw_y_true, data_scaler)
            swap_elem, _ = self._channel_mean_error(prediction_swap_loss_view, y_loss_view, targets_mask)
            swap_weight, swap_weight_mean = self._fpem_swap_weight(output, full_elem)
            swap_diff_loss_raw = masked_mean(
                F.relu(float(self.cfg.swap_margin) + full_elem.detach() - swap_elem) * swap_weight,
                elem_mask,
            )
            swap_loss_raw = swap_diff_loss_raw
            swap_delta_mean = masked_mean((swap_elem - full_elem).detach(), elem_mask)

        rank_loss_raw = zero
        pred_fut_mu_minus = output.get("pred_fut_mu_minus")
        pred_fut_logvar_minus = output.get("pred_fut_logvar_minus")
        env_fut_tokens = output.get("env_fut_tokens")
        if (
            env_fut_tokens is not None
            and pred_fut_mu is not None
            and pred_fut_logvar is not None
            and pred_fut_mu_minus is not None
            and pred_fut_logvar_minus is not None
        ):
            target = env_fut_tokens.detach()
            nll_plus = self._diag_gaussian_nll(target, pred_fut_mu, pred_fut_logvar).mean(dim=-1)
            nll_minus = self._diag_gaussian_nll(target, pred_fut_mu_minus, pred_fut_logvar_minus).mean(dim=-1)
            future_mi_logs["env_fut_nll_plus"] = nll_plus.mean().detach()
            future_mi_logs["env_fut_nll_minus"] = nll_minus.mean().detach()
            rank_loss_raw = F.relu(float(self.cfg.rank_margin) + nll_plus - nll_minus).mean()

        pred_loss = pred_loss_raw
        inv_loss = inv_loss_raw if self.cfg.use_inv and self.cfg.lambda_inv != 0 else zero
        effective_lambda_envpred = self._effective_aux_lambda(
            self.cfg.use_envpred,
            self.cfg.lambda_envpred,
            self.cfg.future_mi_warmup_epochs,
        )
        effective_lambda_future_mi = self._effective_aux_lambda(
            self.cfg.use_future_mi,
            self.cfg.lambda_future_mi,
            self.cfg.future_mi_warmup_epochs,
        )
        effective_lambda_swap = self._effective_aux_lambda(
            has_swap,
            self.cfg.lambda_swap,
            self.cfg.swap_warmup_epochs,
        )
        sep_lambda = self.cfg.lambda_ind if self.cfg.lambda_sep is None else self.cfg.lambda_sep
        effective_lambda_sep = self._effective_aux_lambda(
            self.cfg.use_ind,
            sep_lambda,
            self.cfg.sep_warmup_epochs,
        )
        sparse_enabled = self.cfg.use_mask_sparse or self.cfg.use_sparse
        lambda_mask_sparse = self.cfg.lambda_sparse if self.cfg.lambda_mask_sparse is None else self.cfg.lambda_mask_sparse
        effective_lambda_mask_sparse = self._effective_aux_lambda(
            sparse_enabled,
            lambda_mask_sparse,
            self.cfg.mask_sparse_warmup_epochs,
        )
        envpred_loss = envpred_loss_raw if effective_lambda_envpred != 0 else zero
        future_mi_loss = future_mi_loss_raw if effective_lambda_future_mi != 0 else zero
        rank_loss = rank_loss_raw if self.cfg.use_rank and self.cfg.lambda_rank != 0 else zero
        sparse_loss = sparse_loss_raw if effective_lambda_mask_sparse != 0 else zero
        effective_lambda_kl = self._effective_lambda_kl()
        kl_loss = kl_loss_raw if effective_lambda_kl != 0 else zero
        ind_loss = sep_loss_raw if effective_lambda_sep != 0 else zero
        swap_loss = swap_loss_raw if effective_lambda_swap != 0 else zero
        swap_diff_loss = swap_diff_loss_raw if effective_lambda_swap != 0 else zero
        backbone_aux_raw, backbone_aux_logs = self._backbone_aux_terms(output, prediction)
        backbone_aux_loss = (
            backbone_aux_raw
            if self.cfg.use_backbone_aux and self.cfg.lambda_backbone_aux != 0
            else zero
        )
        z_cons_loss_raw = self._consistency_term(output.get("z_inv_aug"), output.get("z_inv"), prediction, "mse")
        y_cons_loss_raw = self._consistency_term(output.get("y_inv_aug"), y_inv, prediction, "mae")
        effective_lambda_z_cons = (
            float(self.cfg.lambda_z_cons)
            if output.get("z_inv_aug") is not None and self.cfg.lambda_z_cons != 0
            else 0.0
        )
        effective_lambda_y_cons = (
            float(self.cfg.lambda_y_cons)
            if output.get("y_inv_aug") is not None and self.cfg.lambda_y_cons != 0
            else 0.0
        )
        z_cons_loss = z_cons_loss_raw if effective_lambda_z_cons != 0 else zero
        y_cons_loss = y_cons_loss_raw if effective_lambda_y_cons != 0 else zero
        loss_z_inv_ib, z_inv_ib_logs = self._z_inv_bottleneck_terms(output, prediction)
        pseudo_env_loss, pseudo_env_logs = self._pseudo_env_terms(
            output,
            y_loss_view,
            y_inv_loss_view,
            targets_mask,
            data_scaler=data_scaler,
        )
        env_route_loss, env_route_logs = self._env_route_terms(
            output,
            y_loss_view,
            targets_mask,
            data_scaler=data_scaler,
        )

        total_loss = (
            self.cfg.lambda_pred * pred_loss
            + self.cfg.lambda_inv * inv_loss
            + effective_lambda_envpred * envpred_loss
            + effective_lambda_future_mi * future_mi_loss
            + effective_lambda_sep * ind_loss
            + effective_lambda_mask_sparse * sparse_loss
            + effective_lambda_swap * swap_loss
            + effective_lambda_kl * kl_loss
            + self.cfg.lambda_rank * rank_loss
            + self.cfg.lambda_backbone_aux * backbone_aux_loss
            + effective_lambda_z_cons * z_cons_loss
            + effective_lambda_y_cons * y_cons_loss
            + loss_z_inv_ib
            + pseudo_env_loss
            + env_route_loss
        )

        rho = output.get("rho")
        env_plus = output.get("env_plus")
        env_minus = output.get("env_minus")
        fusion_gamma = output.get("fusion_gamma")
        fusion_beta = output.get("fusion_beta")
        logs = {
            "total_loss": total_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "inv_loss": inv_loss.detach(),
            "gate_loss": zero,
            "swap_loss": swap_loss.detach(),
            "swap_diff_loss": swap_diff_loss.detach(),
            "swap_same_loss": swap_same_loss_raw.detach(),
            "kl_loss": kl_loss.detach(),
            "effective_lambda_kl": prediction.new_tensor(effective_lambda_kl),
            "effective_lambda_envpred": prediction.new_tensor(effective_lambda_envpred),
            "effective_lambda_future_mi": prediction.new_tensor(effective_lambda_future_mi),
            "effective_lambda_swap": prediction.new_tensor(effective_lambda_swap),
            "effective_lambda_sep": prediction.new_tensor(effective_lambda_sep),
            "effective_lambda_mask_sparse": prediction.new_tensor(effective_lambda_mask_sparse),
            "aux_schedule_factor": prediction.new_tensor(self._aux_schedule_factor()),
            "ind_loss": ind_loss.detach(),
            "sep_loss": ind_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "mask_sparse_loss": sparse_loss.detach(),
            "entropy_loss": zero,
            "residual_norm_loss": zero,
            "env_consistency_loss": zero,
            "z_cons_loss": z_cons_loss.detach(),
            "y_cons_loss": y_cons_loss.detach(),
            "effective_lambda_z_cons": prediction.new_tensor(effective_lambda_z_cons),
            "effective_lambda_y_cons": prediction.new_tensor(effective_lambda_y_cons),
            "persistence_mi_loss": zero,
            "effective_lambda_persistence_mi": zero,
            "envpred_loss": envpred_loss.detach(),
            "future_mi_loss": future_mi_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "rho_mean": rho.detach().mean() if rho is not None else zero,
            "rho_std": rho.detach().std(unbiased=False) if rho is not None else zero,
            "rho_min": rho.detach().min() if rho is not None else zero,
            "rho_max": rho.detach().max() if rho is not None else zero,
            "rho_entropy": mask_entropy_mean.detach(),
            "delta_gain_mean": zero,
            "delta_gain_std": zero,
            "delta_gain_pos_ratio": zero,
            "s_gain_mean": zero,
            "persist_score_mean": zero,
            "persist_score_std": zero,
            "s_persist_mean": zero,
            "s_gate_mean": zero,
            "persistence_valid": prediction.new_tensor(float(env_fut is not None)),
            "potential_gain_mean": zero,
            "swap_delta_mean": swap_delta_mean.detach(),
            "swap_weight_mean": swap_weight_mean.detach(),
            "env_mu_abs_mean": env_mu.detach().abs().mean(),
            "env_std_mean": torch.exp(0.5 * env_logvar.detach()).mean(),
            "r_env_abs_mean": zero,
            "y_inv_mae": inv_loss_raw.detach(),
            "y_potential_mae": pred_loss_raw.detach(),
            "y_hat_mae": pred_loss_raw.detach(),
            "mask_mean": mask_mean.detach(),
            "mask_std": mask_std.detach(),
            "mask_min": mask_min.detach(),
            "mask_max": mask_max.detach(),
            "mask_entropy": mask_entropy_mean.detach(),
            "mask_active_ratio": mask_active_ratio.detach(),
            "env_plus_norm": env_plus.detach().norm(dim=-1).mean() if env_plus is not None else zero,
            "env_minus_norm": env_minus.detach().norm(dim=-1).mean() if env_minus is not None else zero,
            "env_hist_norm": env_hist.detach().norm(dim=-1).mean() if env_hist is not None else zero,
            "env_fut_norm": env_fut.detach().norm(dim=-1).mean() if env_fut is not None else zero,
            "env_fut_pred_norm": pred_fut_mu.detach().norm(dim=-1).mean() if pred_fut_mu is not None else zero,
            "pred_fut_mu_norm": pred_fut_mu.detach().norm(dim=-1).mean() if pred_fut_mu is not None else zero,
            "fusion_gamma_abs_mean": fusion_gamma.detach().abs().mean() if fusion_gamma is not None else zero,
            "fusion_beta_abs_mean": fusion_beta.detach().abs().mean() if fusion_beta is not None else zero,
            "timestamp_valid": prediction.new_tensor(float(bool(output.get("timestamp_valid", False)))),
            "cur_time_emb_norm": (
                output["cur_time_emb"].detach().norm(dim=-1).mean()
                if output.get("cur_time_emb") is not None else zero
            ),
            "seq_time_emb_norm": (
                output["seq_time_emb"].detach().norm(dim=-1).mean()
                if output.get("seq_time_emb") is not None else zero
            ),
            "future_time_emb_norm": (
                output["future_time_emb"].detach().norm(dim=-1).mean()
                if output.get("future_time_emb") is not None else zero
            ),
        }
        logs.update({key: value.detach() for key, value in future_mi_logs.items()})
        logs.update({key: value.detach() for key, value in sep_logs.items()})
        logs.update({key: value.detach() for key, value in backbone_aux_logs.items()})
        logs.update({key: value.detach() for key, value in z_inv_ib_logs.items()})
        logs.update({key: value.detach() for key, value in pseudo_env_logs.items()})
        logs.update({key: value.detach() for key, value in env_route_logs.items()})
        logs.update(self._separation_logs(output, prediction))
        logs["__loss_terms__"] = {
            "pred": self.cfg.lambda_pred * pred_loss,
            "inv": self.cfg.lambda_inv * inv_loss,
            "envpred": effective_lambda_envpred * envpred_loss,
            "future_mi": effective_lambda_future_mi * future_mi_loss,
            "swap": effective_lambda_swap * swap_loss,
            "sep": effective_lambda_sep * ind_loss,
            "sparse": effective_lambda_mask_sparse * sparse_loss,
            "z_inv_ib": loss_z_inv_ib,
            "pseudo_env": pseudo_env_loss,
            "env_route": env_route_loss,
        }
        self.latest_log_dict = {
            key: float(value.cpu())
            for key, value in logs.items()
            if key != "__loss_terms__"
        }
        return total_loss, logs

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
        raw_y_true: Optional[torch.Tensor] = None,
        data_scaler=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if output.get("method_variant", "nue") == "fpem":
            return self._forward_fpem(output, y_true, targets_mask, raw_y_true=raw_y_true, data_scaler=data_scaler)
        prediction = output["prediction"]
        y_inv = output["y_inv"]
        y_potential = output["y_potential"]
        r_env = output["r_env"]
        rho = output["rho"]
        zero = self._zero(prediction)
        prediction_loss_view, y_loss_view = self._forecast_pair(prediction, y_true, raw_y_true, data_scaler)
        y_inv_loss_view, _ = self._forecast_pair(y_inv, y_true, raw_y_true, data_scaler)
        y_potential_loss_view, _ = self._forecast_pair(y_potential, y_true, raw_y_true, data_scaler)

        pred_loss_raw = self._mae_loss(prediction_loss_view, y_loss_view, targets_mask)
        inv_loss_raw = self._mae_loss(y_inv_loss_view, y_loss_view, targets_mask)
        potential_loss_raw = self._mae_loss(y_potential_loss_view, y_loss_view, targets_mask)

        inv_elem, elem_mask = self._channel_mean_error(y_inv_loss_view, y_loss_view, targets_mask)
        potential_elem, _ = self._channel_mean_error(y_potential_loss_view, y_loss_view, targets_mask)
        full_elem, _ = self._channel_mean_error(prediction_loss_view, y_loss_view, targets_mask)

        delta_gain = inv_elem - potential_elem
        s_gain = torch.sigmoid((delta_gain - self.cfg.gate_eta) / max(self.cfg.gate_tau, 1e-6)).detach()
        persistence_mi_loss_raw, s_persist, persistence_logs = self._persistence_terms(output, prediction)
        if self.cfg.use_persistence_mi and self.cfg.persistence_affects_gate and s_persist is not None:
            s_gate = s_gain * s_persist.detach().unsqueeze(1)
        else:
            s_gate = s_gain
        delta_gain_detached = delta_gain.detach()
        delta_gain_mean = masked_mean(delta_gain_detached, elem_mask)
        delta_gain_std = torch.sqrt(
            masked_mean((delta_gain_detached - delta_gain_mean).pow(2), elem_mask).clamp_min(0.0)
        )
        s_gain_mean = masked_mean(s_gain, elem_mask)

        gate_loss_raw = self._bce_gate_loss(rho, s_gate.detach(), elem_mask)
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
        has_swap = self.cfg.use_swap and self.cfg.lambda_swap != 0 and prediction_swap is not None
        if has_swap:
            prediction_swap_loss_view, _ = self._forecast_pair(prediction_swap, y_true, raw_y_true, data_scaler)
            swap_elem, _ = self._channel_mean_error(prediction_swap_loss_view, y_loss_view, targets_mask)
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
            same_target = prediction_loss_view.detach() if self.cfg.swap_detach_full else prediction_loss_view
            swap_same_loss_raw = masked_mean(
                (1.0 - s_gain) * (prediction_swap_loss_view - same_target).abs().mean(dim=-1, keepdim=True),
                elem_mask,
            )
            swap_loss_raw = (
                self.cfg.lambda_swap_diff * swap_diff_loss_raw
                + self.cfg.lambda_swap_same * swap_same_loss_raw
            )
            swap_delta_mean = masked_mean((swap_elem - full_elem).detach(), elem_mask)

        env_consistency_loss_raw = zero
        has_env_consistency = self.cfg.use_env_consistency and self.cfg.lambda_env_consistency != 0
        if has_env_consistency and output.get("env_perm") is not None:
            env_consistency_loss_raw = (output["env_perm"].detach() - env).abs().mean()

        pred_loss = pred_loss_raw
        inv_loss = inv_loss_raw if self.cfg.use_inv and self.cfg.lambda_inv != 0 else zero
        gate_loss = gate_loss_raw if self.cfg.use_gate and self.cfg.lambda_gate != 0 else zero
        effective_lambda_swap = self._effective_aux_lambda(
            has_swap,
            self.cfg.lambda_swap,
            self.cfg.swap_warmup_epochs,
        )
        effective_lambda_sep = self._effective_aux_lambda(
            self.cfg.use_ind,
            self.cfg.lambda_ind,
            self.cfg.sep_warmup_epochs,
        )
        swap_loss = swap_loss_raw if effective_lambda_swap != 0 else zero
        swap_diff_loss = swap_diff_loss_raw if effective_lambda_swap != 0 else zero
        swap_same_loss = swap_same_loss_raw if effective_lambda_swap != 0 else zero
        effective_lambda_kl = self._effective_lambda_kl()
        effective_lambda_persistence_mi = self._effective_lambda_persistence_mi()
        kl_loss = kl_loss_raw if effective_lambda_kl != 0 else zero
        persistence_mi_loss = (
            persistence_mi_loss_raw
            if effective_lambda_persistence_mi != 0 and persistence_logs["persistence_valid"].item() == 1.0
            else zero
        )
        ind_loss = ind_loss_raw if effective_lambda_sep != 0 else zero
        sparse_loss = sparse_loss_raw if self.cfg.use_sparse and self.cfg.lambda_sparse != 0 else zero
        entropy_loss = entropy_loss_raw if self.cfg.use_entropy and self.cfg.lambda_entropy != 0 else zero
        residual_norm_loss = (
            residual_norm_loss_raw
            if self.cfg.use_residual_norm and self.cfg.lambda_residual_norm != 0
            else zero
        )
        env_consistency_loss = env_consistency_loss_raw if has_env_consistency else zero
        backbone_aux_raw, backbone_aux_logs = self._backbone_aux_terms(output, prediction)
        backbone_aux_loss = (
            backbone_aux_raw
            if self.cfg.use_backbone_aux and self.cfg.lambda_backbone_aux != 0
            else zero
        )

        total_loss = (
            self.cfg.lambda_pred * pred_loss
            + self.cfg.lambda_inv * inv_loss
            + self.cfg.lambda_gate * gate_loss
            + effective_lambda_swap * swap_loss
            + effective_lambda_kl * kl_loss
            + effective_lambda_persistence_mi * persistence_mi_loss
            + effective_lambda_sep * ind_loss
            + self.cfg.lambda_sparse * sparse_loss
            + self.cfg.lambda_entropy * entropy_loss
            + self.cfg.lambda_residual_norm * residual_norm_loss
            + self.cfg.lambda_env_consistency * env_consistency_loss
            + self.cfg.lambda_backbone_aux * backbone_aux_loss
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
            "effective_lambda_envpred": zero,
            "effective_lambda_future_mi": zero,
            "effective_lambda_swap": prediction.new_tensor(effective_lambda_swap),
            "effective_lambda_sep": prediction.new_tensor(effective_lambda_sep),
            "effective_lambda_mask_sparse": zero,
            "aux_schedule_factor": prediction.new_tensor(self._aux_schedule_factor()),
            "ind_loss": ind_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "residual_norm_loss": residual_norm_loss.detach(),
            "env_consistency_loss": env_consistency_loss.detach(),
            "z_cons_loss": zero,
            "y_cons_loss": zero,
            "effective_lambda_z_cons": zero,
            "effective_lambda_y_cons": zero,
            "persistence_mi_loss": persistence_mi_loss.detach(),
            "effective_lambda_persistence_mi": prediction.new_tensor(effective_lambda_persistence_mi),
            "rho_mean": rho.detach().mean(),
            "rho_std": rho.detach().std(unbiased=False),
            "rho_min": rho.detach().min(),
            "rho_max": rho.detach().max(),
            "rho_entropy": rho_entropy.detach().mean(),
            "delta_gain_mean": delta_gain_mean,
            "delta_gain_std": delta_gain_std,
            "delta_gain_pos_ratio": masked_mean((delta_gain_detached > 0).float(), elem_mask),
            "s_gain_mean": s_gain_mean,
            "s_gate_mean": masked_mean(s_gate.detach(), elem_mask),
            "potential_gain_mean": delta_gain_mean,
            "swap_delta_mean": swap_delta_mean.detach(),
            "env_mu_abs_mean": env_mu.detach().abs().mean(),
            "env_std_mean": torch.exp(0.5 * env_logvar.detach()).mean(),
            "r_env_abs_mean": r_env.detach().abs().mean(),
            "y_inv_mae": inv_loss_raw.detach(),
            "y_potential_mae": potential_loss_raw.detach(),
            "y_hat_mae": pred_loss_raw.detach(),
        }
        logs.update({key: value.detach() for key, value in persistence_logs.items()})
        logs.update({key: value.detach() for key, value in backbone_aux_logs.items()})
        logs.update(self._separation_logs(output, prediction))
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
