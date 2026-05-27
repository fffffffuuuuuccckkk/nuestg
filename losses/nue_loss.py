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
    future_mi_tau: float = 0.2
    use_rank: bool = False
    lambda_rank: float = 0.0
    rank_margin: float = 0.1
    use_mask_sparse: bool = False


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
        "persistence_mi_loss",
        "envpred_loss",
        "future_mi_loss",
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

    def _effective_lambda_persistence_mi(self) -> float:
        if not self.cfg.use_persistence_mi or self.cfg.lambda_persistence_mi == 0:
            return 0.0
        warmup = int(self.cfg.persistence_warmup_epochs or 0)
        if warmup <= 0:
            return float(self.cfg.lambda_persistence_mi)
        return float(self.cfg.lambda_persistence_mi) * min(1.0, max(self.epoch, 0) / warmup)

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

    def _future_mi_loss(
        self,
        env_plus: Optional[torch.Tensor],
        env_fut: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        zero = self._zero(like)
        if env_plus is None or env_fut is None:
            return zero
        if env_plus.numel() == 0 or env_fut.numel() == 0:
            return zero
        q_flat = env_plus.reshape(-1, env_plus.shape[-1])
        k_flat = env_fut.detach().reshape(-1, env_fut.shape[-1])
        if q_flat.shape[0] <= 1 or k_flat.shape[0] <= 1:
            return zero
        q_norm = F.normalize(q_flat, dim=-1)
        k_norm = F.normalize(k_flat, dim=-1)
        logits = q_norm.matmul(k_norm.transpose(0, 1)) / max(float(self.cfg.future_mi_tau), 1e-6)
        labels = torch.arange(q_flat.shape[0], device=q_flat.device)
        return F.cross_entropy(logits, labels)

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
        else:
            source = output.get("env_plus")
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = output["prediction"]
        y_inv = output["y_inv"]
        zero = self._zero(prediction)

        pred_loss_raw = self._mae_loss(prediction, y_true, targets_mask)
        inv_loss_raw = self._mae_loss(y_inv, y_true, targets_mask)
        full_elem, elem_mask = self._channel_mean_error(prediction, y_true, targets_mask)

        env_fut = output.get("env_fut")
        env_fut_pred = output.get("env_fut_pred")
        envpred_loss_raw = zero
        if env_fut is not None and env_fut_pred is not None:
            env_fut_target = env_fut.detach()
            if self.cfg.envpred_loss_type == "mse":
                envpred_loss_raw = F.mse_loss(env_fut_pred, env_fut_target)
            elif self.cfg.envpred_loss_type == "cosine":
                envpred_loss_raw = 1.0 - F.cosine_similarity(env_fut_pred, env_fut_target, dim=-1).mean()
            else:
                raise ValueError(f"Unsupported envpred_loss_type={self.cfg.envpred_loss_type!r}")

        future_mi_loss_raw = self._future_mi_loss(output.get("env_plus"), env_fut, prediction)
        env_hist = output.get("env_hist", output.get("env"))
        ind_loss_raw = self._independence_loss(output["z_inv"], env_hist) if env_hist is not None else zero
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
            swap_elem, _ = self._channel_mean_error(prediction_swap, y_true, targets_mask)
            swap_weight, swap_weight_mean = self._fpem_swap_weight(output, full_elem)
            swap_diff_loss_raw = masked_mean(
                F.relu(float(self.cfg.swap_margin) + full_elem.detach() - swap_elem) * swap_weight,
                elem_mask,
            )
            swap_loss_raw = swap_diff_loss_raw
            swap_delta_mean = masked_mean((swap_elem - full_elem).detach(), elem_mask)

        rank_loss_raw = zero
        env_fut_pred_minus = output.get("env_fut_pred_minus")
        if env_fut is not None and env_fut_pred is not None and env_fut_pred_minus is not None:
            env_fut_target = env_fut.detach()
            s_plus = F.cosine_similarity(env_fut_pred, env_fut_target, dim=-1)
            s_minus = F.cosine_similarity(env_fut_pred_minus, env_fut_target, dim=-1)
            rank_loss_raw = F.relu(float(self.cfg.rank_margin) - s_plus + s_minus).mean()

        pred_loss = pred_loss_raw
        inv_loss = inv_loss_raw if self.cfg.use_inv and self.cfg.lambda_inv != 0 else zero
        envpred_loss = envpred_loss_raw if self.cfg.use_envpred and self.cfg.lambda_envpred != 0 else zero
        future_mi_loss = future_mi_loss_raw if self.cfg.use_future_mi and self.cfg.lambda_future_mi != 0 else zero
        rank_loss = rank_loss_raw if self.cfg.use_rank and self.cfg.lambda_rank != 0 else zero
        sparse_enabled = self.cfg.use_mask_sparse or self.cfg.use_sparse
        sparse_loss = sparse_loss_raw if sparse_enabled and self.cfg.lambda_sparse != 0 else zero
        effective_lambda_kl = self._effective_lambda_kl()
        kl_loss = kl_loss_raw if effective_lambda_kl != 0 else zero
        ind_loss = ind_loss_raw if self.cfg.use_ind and self.cfg.lambda_ind != 0 else zero
        swap_loss = swap_loss_raw if has_swap else zero
        swap_diff_loss = swap_diff_loss_raw if has_swap else zero

        total_loss = (
            self.cfg.lambda_pred * pred_loss
            + self.cfg.lambda_inv * inv_loss
            + self.cfg.lambda_envpred * envpred_loss
            + self.cfg.lambda_future_mi * future_mi_loss
            + self.cfg.lambda_ind * ind_loss
            + self.cfg.lambda_sparse * sparse_loss
            + self.cfg.lambda_swap * swap_loss
            + effective_lambda_kl * kl_loss
            + self.cfg.lambda_rank * rank_loss
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
            "ind_loss": ind_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "mask_sparse_loss": sparse_loss.detach(),
            "entropy_loss": zero,
            "residual_norm_loss": zero,
            "env_consistency_loss": zero,
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
            "env_fut_pred_norm": env_fut_pred.detach().norm(dim=-1).mean() if env_fut_pred is not None else zero,
            "fusion_gamma_abs_mean": fusion_gamma.detach().abs().mean() if fusion_gamma is not None else zero,
            "fusion_beta_abs_mean": fusion_beta.detach().abs().mean() if fusion_beta is not None else zero,
        }
        logs.update(self._separation_logs(output, prediction))
        self.latest_log_dict = {key: float(value.cpu()) for key, value in logs.items()}
        return total_loss, logs

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        targets_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if output.get("method_variant", "nue") == "fpem":
            return self._forward_fpem(output, y_true, targets_mask)
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
        has_env_consistency = self.cfg.use_env_consistency and self.cfg.lambda_env_consistency != 0
        if has_env_consistency and output.get("env_perm") is not None:
            env_consistency_loss_raw = (output["env_perm"].detach() - env).abs().mean()

        pred_loss = pred_loss_raw
        inv_loss = inv_loss_raw if self.cfg.use_inv and self.cfg.lambda_inv != 0 else zero
        gate_loss = gate_loss_raw if self.cfg.use_gate and self.cfg.lambda_gate != 0 else zero
        swap_loss = swap_loss_raw if has_swap else zero
        swap_diff_loss = swap_diff_loss_raw if has_swap else zero
        swap_same_loss = swap_same_loss_raw if has_swap else zero
        effective_lambda_kl = self._effective_lambda_kl()
        effective_lambda_persistence_mi = self._effective_lambda_persistence_mi()
        kl_loss = kl_loss_raw if effective_lambda_kl != 0 else zero
        persistence_mi_loss = (
            persistence_mi_loss_raw
            if effective_lambda_persistence_mi != 0 and persistence_logs["persistence_valid"].item() == 1.0
            else zero
        )
        ind_loss = ind_loss_raw if self.cfg.use_ind and self.cfg.lambda_ind != 0 else zero
        sparse_loss = sparse_loss_raw if self.cfg.use_sparse and self.cfg.lambda_sparse != 0 else zero
        entropy_loss = entropy_loss_raw if self.cfg.use_entropy and self.cfg.lambda_entropy != 0 else zero
        residual_norm_loss = (
            residual_norm_loss_raw
            if self.cfg.use_residual_norm and self.cfg.lambda_residual_norm != 0
            else zero
        )
        env_consistency_loss = env_consistency_loss_raw if has_env_consistency else zero

        total_loss = (
            self.cfg.lambda_pred * pred_loss
            + self.cfg.lambda_inv * inv_loss
            + self.cfg.lambda_gate * gate_loss
            + self.cfg.lambda_swap * swap_loss
            + effective_lambda_kl * kl_loss
            + effective_lambda_persistence_mi * persistence_mi_loss
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
