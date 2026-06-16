from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from basicts.data import BasicTSForecastingDataset

from losses import NUESTGLoss, make_basicts_loss, nue_mae_metric
from models import NUESTG, NUESTGConfig
from models.backbones import build_backbone
from models.backbones.official_utils import OfficialBaselineSkip
from utils import (
    AverageMeterDict,
    align_target,
    append_csv_log,
    assert_finite,
    ensure_blnc,
    format_logs,
    masked_mae_value,
    masked_mape_value,
    masked_rmse_value,
    masked_mse_value,
    masked_wape_value,
    make_mape_valid_mask,
    make_valid_mask,
    maybe_generate_timestamp_file,
    resolve_cli_config,
    save_resolved_config,
    ZScoreDataScaler,
)


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
    "grad_consensus/agree_mean",
    "grad_consensus/agree_std",
    "grad_consensus/agree_min",
    "grad_consensus/agree_max",
    "grad_consensus/rho_mean",
    "grad_consensus/rho_max",
    "grad_consensus/using_fallback_z_inv",
    "grad_consensus/ema_agree_mean",
    "grad_surgery/enabled",
    "grad_surgery/conflict_cos",
    "grad_surgery/conflict_dot",
    "grad_surgery/projection_norm",
    "grad_surgery/aux_grad_norm",
    "grad_surgery/primary_grad_norm",
    "curriculum_horizon",
    "cast_vq_loss",
    "cast_commit_loss",
    "cast_mi_loss",
    "stone_graph_perturb_loss",
    "stone_spatial_graph_entropy",
    "stone_temporal_graph_entropy",
]
Z_INV_IB_LOG_KEYS = [
    "loss_z_inv_ib",
    "z_inv_ib/enabled",
    "z_inv_ib/type",
    "z_inv_ib/kl",
    "z_inv_ib/beta",
    "z_inv_ib/z_mu_abs_mean",
    "z_inv_ib/z_logvar_mean",
    "z_inv_ib/z_sample_std",
]
PSEUDO_ENV_BASE_LOG_KEYS = [
    "pseudo_env/enabled",
    "pseudo_env/head_loss",
    "pseudo_env/var_loss",
    "pseudo_env/balance_loss",
    "pseudo_env/entropy",
    "pseudo_env/q_entropy",
    "pseudo_env/diverse_loss",
    "pseudo_env/q_max_mean",
    "pseudo_env/r_var",
    "pseudo_env/cache_updated",
    "pseudo_env/smoothing_enabled",
    "pseudo_env/collapse_warning",
]
ENV_ROUTE_BASE_LOG_KEYS = [
    "env_route/enabled",
    "env_route/final_loss",
    "env_route/global_loss",
    "env_route/route_soft_loss",
    "env_route/expert_loss",
    "env_route/router_oracle_loss",
    "env_route/balance_loss",
    "env_route/diverse_loss",
    "env_route/entropy",
    "env_route/L_final",
    "env_route/L_route_final",
    "env_route/L_global",
    "env_route/L_route_soft",
    "env_route/L_expert",
    "env_route/L_router_oracle",
    "env_route/L_balance",
    "env_route/L_diverse",
    "env_route/L_entropy",
    "env_route/oracle_tau",
    "env_route/q_entropy",
    "env_route/q_oracle_entropy",
    "env_route/q_oracle_max_mean",
    "env_route/alpha_mean",
    "env_route/alpha_std",
    "env_route/q_max_mean",
    "env_route/router_oracle_acc",
    "env_route/y_inv_mae",
    "env_route/y_global_mae",
    "env_route/y_route_mae",
    "env_route/y_route_soft_mae",
    "env_route/y_route_final_mae",
    "env_route/oracle_route_mae",
]
HORIZON_EVAL_STEPS = (3, 6, 12)
METRIC_FIELDS = [
    "val_mae",
    "val_mse",
    "val_rmse",
    "val_mape",
    "val_wape",
    "val_mae_h3",
    "val_rmse_h3",
    "val_mape_h3",
    "val_wape_h3",
    "val_mae_h6",
    "val_rmse_h6",
    "val_mape_h6",
    "val_wape_h6",
    "val_mae_h12",
    "val_rmse_h12",
    "val_mape_h12",
    "val_wape_h12",
    "val_mae_avg12",
    "val_rmse_avg12",
    "val_mape_avg12",
    "val_wape_avg12",
    "lr",
]
CSV_FIELDS = ["epoch", "step", "split", *LOG_KEYS, *METRIC_FIELDS]


def z_inv_bottleneck_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("LOSS", {}).get("z_inv_bottleneck", {}).get("enabled", False))


def pseudo_env_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("LOSS", {}).get("use_pseudo_env_heads", False))


def pseudo_env_log_keys(cfg: Dict) -> list[str]:
    if not pseudo_env_enabled(cfg):
        return []
    k = int(cfg.get("LOSS", {}).get("pseudo_env_k", 3))
    keys = list(PSEUDO_ENV_BASE_LOG_KEYS)
    for idx in range(k):
        keys.extend([
            f"pseudo_env/count_head_{idx}",
            f"pseudo_env/per_head_mae_{idx}",
            f"pseudo_env/risk_head_{idx}",
        ])
    return keys


def env_route_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("LOSS", {}).get("use_env_routed_inv_heads", False))


def env_route_log_keys(cfg: Dict) -> list[str]:
    if not env_route_enabled(cfg):
        return []
    k = int(cfg.get("LOSS", {}).get("env_route_k", 3))
    keys = list(ENV_ROUTE_BASE_LOG_KEYS)
    for idx in range(k):
        keys.extend([
            f"env_route/count_head_{idx}",
            f"env_route/counts_per_head_{idx}",
            f"env_route/oracle_count_head_{idx}",
            f"env_route/oracle_counts_per_head_{idx}",
            f"env_route/per_head_mae_{idx}",
        ])
    return keys


def log_keys_for_config(cfg: Dict) -> list[str]:
    keys = list(LOG_KEYS)
    if z_inv_bottleneck_enabled(cfg):
        keys.extend(Z_INV_IB_LOG_KEYS)
    keys.extend(pseudo_env_log_keys(cfg))
    keys.extend(env_route_log_keys(cfg))
    return keys


def csv_fields_for_config(cfg: Dict) -> list[str]:
    return ["epoch", "step", "split", *log_keys_for_config(cfg), *METRIC_FIELDS]


BACKBONE_DESCRIPTIONS = {
    "stid": "faithful_native_adapter STID backbone with spatial identity and TOD/DOW embeddings",
    "official_stid": "faithful_native_adapter STID backbone with spatial identity and TOD/DOW embeddings",
    "stid_mlp": "lightweight STID-like temporal MLP + node embedding",
    "mlp": "lightweight STID-like temporal MLP + node embedding",
    "stid_like": "lightweight STID-like temporal MLP + node embedding",
    "graphwavenet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "graph_wavenet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "gwnet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "graphwavenet_full": "official_graphwavenet_full prediction path with a thin FPEM representation adapter",
    "graph_wavenet_full": "official_graphwavenet_full prediction path with a thin FPEM representation adapter",
    "gwnet_full": "official_graphwavenet_full prediction path with a thin FPEM representation adapter",
    "graphwavenet-full": "official_graphwavenet_full prediction path with a thin FPEM representation adapter",
    "agcrn": "faithful_native_adapter AGCRN backbone adapted to the shared interface",
    "stgcn": "reference_native STGCN backbone adapted from hazdzz/STGCN",
    "stnorm": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "st_norm": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "stnorm_wavenet": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "d2stgnn": "official_local_wrapper D2STGNN backbone adapted to the shared interface",
    "cast": "cast_method_level_pytorch_reproduction_with_official_aux_loss; fixed-node PyTorch adapter, not full official PyG/ST-OOD pipeline",
    "cast_adapter": "cast_method_level_pytorch_reproduction_with_official_aux_loss; fixed-node PyTorch adapter, not full official PyG/ST-OOD pipeline",
    "cast_official": "official_local_wrapper gate for full CaST; skips when official PyG/data/loss protocol is unavailable",
    "stone": "stone_method_level_pytorch_adapter_with_adjacency_frechet_side_info_and_graph_perturbation",
    "stone_adapter": "stone_method_level_pytorch_adapter_with_adjacency_frechet_side_info_and_graph_perturbation",
    "stone_official": "official_local_wrapper gate for full STONE; skips when spatial side information is unavailable",
    "stop": "stop_released_code_method_reproduction_without_sood_protocol adapted from released STOP architecture",
    "stop_adapter": "stop_released_code_method_reproduction_without_sood_protocol adapted from released STOP architecture",
    "stop_official": "official_local_wrapper gate for full STOP; skips when official SOOD protocol is unavailable",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finalize_config(cfg: Dict) -> Dict:
    """Keep duplicated model/dataset shape fields synchronized."""
    ds_cfg = cfg["DATASET"]
    model_cfg = cfg["MODEL"]
    loss_cfg = cfg["LOSS"]
    train_cfg = cfg.setdefault("TRAIN", {})
    run_cfg = cfg.setdefault("RUN", {})
    swap_cfg = cfg.get("SWAP", {})
    cfg.setdefault("EVAL", {}).setdefault("horizon_metrics", True)
    cfg.setdefault("EVAL", {}).setdefault("save_test_diagnostics", False)
    cfg.setdefault("EVAL", {}).setdefault("metric_aggregation", "batch_mean")
    metrics_cfg = cfg.setdefault("METRICS", {})
    metrics_cfg.setdefault("mape_threshold", 1.0)
    metrics_cfg.setdefault("mape_eps", 1e-5)
    metrics_cfg.setdefault("mape_as_percent", True)
    scaler_cfg = cfg.setdefault("SCALER", {})
    scaler_cfg.setdefault("enabled", True)
    scaler_cfg.setdefault("type", "zscore")
    scaler_cfg.setdefault("norm_each_channel", False)
    scaler_cfg.setdefault("rescale", True)
    scaler_cfg.setdefault("eps", 1e-5)
    train_cfg.setdefault("no_decay_for_bias_norm_emb", False)
    train_cfg.setdefault("torch_num_threads", None)
    train_cfg.setdefault("curriculum_enabled", False)
    train_cfg.setdefault("curriculum_start_horizon", 3)
    train_cfg.setdefault("curriculum_full_horizon_epoch", 30)
    train_cfg.setdefault("teacher_forcing_enabled", False)
    train_cfg.setdefault("tf_decay_steps", 2000)
    train_cfg.setdefault("resume_allow_missing_pseudo_env", True)
    loss_cfg.setdefault("warmup_epochs", 0)
    loss_cfg.setdefault("aux_ramp_epochs", 0)
    loss_cfg.setdefault("mask_value_mode", cfg["DATASET"].get("mask_value_mode", "null_val"))
    grad_consensus_cfg = loss_cfg.setdefault("grad_consensus", {})
    grad_consensus_cfg.setdefault("enabled", False)
    grad_consensus_cfg.setdefault("target", "z_seq")
    grad_consensus_cfg.setdefault("apply_to", "inv_branch")
    grad_consensus_cfg.setdefault("mode", "time_channel")
    grad_consensus_cfg.setdefault("rho_max", 0.1)
    grad_consensus_cfg.setdefault("gamma", 1.0)
    grad_consensus_cfg.setdefault("ema_beta", 0.95)
    grad_consensus_cfg.setdefault("warmup_epochs", 10)
    grad_consensus_cfg.setdefault("eps", 1e-8)
    grad_consensus_cfg.setdefault("sand_alpha", 1.0)
    grad_consensus_cfg.setdefault("use_ema", True)
    grad_consensus_cfg.setdefault("loss_type", "mse")
    grad_consensus_cfg.setdefault("log_stats", True)
    grad_surgery_cfg = loss_cfg.setdefault("grad_surgery", {})
    grad_surgery_cfg.setdefault("enabled", False)
    grad_surgery_cfg.setdefault("method", "pcgrad")
    grad_surgery_cfg.setdefault("apply_to", "inv_encoder")
    grad_surgery_cfg.setdefault("primary_losses", ["pred", "inv"])
    grad_surgery_cfg.setdefault("aux_losses", ["envpred", "future_mi", "swap", "sep", "sparse"])
    z_inv_ib_cfg = loss_cfg.setdefault("z_inv_bottleneck", {})
    z_inv_ib_cfg.setdefault("enabled", False)
    z_inv_ib_cfg.setdefault("type", "vib")
    z_inv_ib_cfg.setdefault("beta", 1.0e-4)
    z_inv_ib_cfg.setdefault("noise_std", 0.05)
    z_inv_ib_cfg.setdefault("kl_free_bits", 0.0)
    z_inv_ib_cfg.setdefault("apply_to", "z_inv")
    z_inv_ib_cfg.setdefault("predict_from_sampled_z", True)
    z_inv_ib_cfg["type"] = str(z_inv_ib_cfg.get("type", "vib") or "vib").lower()
    z_inv_ib_cfg["apply_to"] = str(z_inv_ib_cfg.get("apply_to", "z_inv") or "z_inv").lower()
    if z_inv_ib_cfg["type"] not in {"vib", "gaussian_noise", "l2_norm"}:
        raise ValueError("LOSS.z_inv_bottleneck.type must be one of: vib, gaussian_noise, l2_norm")
    if z_inv_ib_cfg["apply_to"] != "z_inv":
        raise ValueError("LOSS.z_inv_bottleneck.apply_to currently supports 'z_inv' only")
    if bool(z_inv_ib_cfg.get("enabled", False)) and bool(grad_surgery_cfg.get("enabled", False)):
        primary_losses = list(grad_surgery_cfg.get("primary_losses", ["pred", "inv"]) or [])
        aux_losses = list(grad_surgery_cfg.get("aux_losses", []) or [])
        if "z_inv_ib" not in primary_losses and "z_inv_ib" not in aux_losses:
            primary_losses.append("z_inv_ib")
        grad_surgery_cfg["primary_losses"] = primary_losses
    model_cfg["z_inv_bottleneck"] = dict(z_inv_ib_cfg)
    loss_cfg.setdefault("use_pseudo_env_heads", False)
    loss_cfg.setdefault("pseudo_env_k", 3)
    loss_cfg.setdefault("pseudo_env_tau", 1.0)
    loss_cfg.setdefault("pseudo_env_lambda_head", 0.0)
    loss_cfg.setdefault("pseudo_env_lambda_var", 0.0)
    loss_cfg.setdefault("pseudo_env_lambda_balance", 0.0)
    loss_cfg.setdefault("pseudo_env_lambda_entropy", 0.0)
    loss_cfg.setdefault("pseudo_env_lambda_diverse", 0.0)
    loss_cfg.setdefault("pseudo_env_warmup_epochs", 0)
    loss_cfg.setdefault("pseudo_env_update_interval", 1)
    loss_cfg.setdefault("pseudo_env_detach_assignment", True)
    loss_cfg.setdefault("pseudo_env_use_global_cache", True)
    loss_cfg.setdefault("pseudo_env_use_temporal_smoothing", True)
    loss_cfg.setdefault("pseudo_env_smooth_radius", 2)
    loss_cfg.setdefault("pseudo_env_assignment_mode", "cached_soft")
    loss_cfg.setdefault("pseudo_env_level", "window")
    loss_cfg["pseudo_env_k"] = int(loss_cfg.get("pseudo_env_k", 3))
    if loss_cfg["pseudo_env_k"] < 1:
        raise ValueError("LOSS.pseudo_env_k must be >= 1")
    loss_cfg["pseudo_env_assignment_mode"] = str(loss_cfg.get("pseudo_env_assignment_mode", "cached_soft")).lower()
    if loss_cfg["pseudo_env_assignment_mode"] not in {"soft", "hard", "cached_soft", "cached_hard"}:
        raise ValueError("LOSS.pseudo_env_assignment_mode must be soft/hard/cached_soft/cached_hard")
    loss_cfg["pseudo_env_level"] = str(loss_cfg.get("pseudo_env_level", "window")).lower()
    if loss_cfg["pseudo_env_level"] not in {"window", "node"}:
        raise ValueError("LOSS.pseudo_env_level must be 'window' or 'node'")
    if bool(loss_cfg.get("use_pseudo_env_heads", False)) and bool(grad_surgery_cfg.get("enabled", False)):
        aux_losses = list(grad_surgery_cfg.get("aux_losses", []) or [])
        primary_losses = list(grad_surgery_cfg.get("primary_losses", ["pred", "inv"]) or [])
        if "pseudo_env" not in aux_losses and "pseudo_env" not in primary_losses:
            aux_losses.append("pseudo_env")
        grad_surgery_cfg["aux_losses"] = aux_losses
    model_cfg["pseudo_env"] = {
        "enabled": bool(loss_cfg.get("use_pseudo_env_heads", False)),
        "k": int(loss_cfg.get("pseudo_env_k", 3)),
        "hidden_dim": int(loss_cfg.get("pseudo_env_hidden_dim", model_cfg.get("residual_hidden_dim", 64))),
        "dropout": float(loss_cfg.get("pseudo_env_dropout", model_cfg.get("residual_dropout", 0.1))),
    }
    loss_cfg.setdefault("use_env_routed_inv_heads", False)
    loss_cfg.setdefault("env_route_k", 3)
    loss_cfg.setdefault("env_route_tau", 1.0)
    loss_cfg.setdefault("env_route_oracle_tau", 0.3)
    loss_cfg.setdefault("env_route_mode", "confidence_mix")
    loss_cfg.setdefault("env_route_replace_final", False)
    loss_cfg.setdefault("env_route_lambda_final", 1.0)
    loss_cfg.setdefault("env_route_lambda_global", 0.2)
    loss_cfg.setdefault("env_route_lambda_route_soft", 0.5)
    loss_cfg.setdefault("env_route_lambda_expert", 0.2)
    loss_cfg.setdefault("env_route_lambda_router_oracle", 0.5)
    loss_cfg.setdefault("env_route_lambda_balance", 0.01)
    loss_cfg.setdefault("env_route_lambda_diverse", 0.001)
    loss_cfg.setdefault("env_route_lambda_entropy", 0.0)
    loss_cfg.setdefault("env_route_warmup_epochs", 5)
    loss_cfg.setdefault("env_route_detach_q_for_expert", True)
    loss_cfg.setdefault("env_route_use_oracle_weight_for_expert", True)
    loss_cfg.setdefault("env_route_alpha_detach", False)
    loss_cfg["env_route_k"] = int(loss_cfg.get("env_route_k", 3))
    if loss_cfg["env_route_k"] < 1:
        raise ValueError("LOSS.env_route_k must be >= 1")
    loss_cfg["env_route_mode"] = str(loss_cfg.get("env_route_mode", "confidence_mix")).lower()
    if loss_cfg["env_route_mode"] not in {"soft", "hard", "confidence_mix", "uniform", "random"}:
        raise ValueError("LOSS.env_route_mode must be soft/hard/confidence_mix/uniform/random")
    if bool(loss_cfg.get("use_env_routed_inv_heads", False)) and bool(grad_surgery_cfg.get("enabled", False)):
        aux_losses = list(grad_surgery_cfg.get("aux_losses", []) or [])
        primary_losses = list(grad_surgery_cfg.get("primary_losses", ["pred", "inv"]) or [])
        if "env_route" not in aux_losses and "env_route" not in primary_losses:
            aux_losses.append("env_route")
        grad_surgery_cfg["aux_losses"] = aux_losses
    model_cfg["env_routed_inv_heads"] = {
        "enabled": bool(loss_cfg.get("use_env_routed_inv_heads", False)),
        "k": int(loss_cfg.get("env_route_k", 3)),
        "tau": float(loss_cfg.get("env_route_tau", 1.0)),
        "mode": str(loss_cfg.get("env_route_mode", "confidence_mix")),
        "replace_final": bool(loss_cfg.get("env_route_replace_final", False)),
        "alpha_detach": bool(loss_cfg.get("env_route_alpha_detach", False)),
        "hidden_dim": int(loss_cfg.get("env_route_hidden_dim", model_cfg.get("residual_hidden_dim", 64))),
        "dropout": float(loss_cfg.get("env_route_dropout", model_cfg.get("residual_dropout", 0.1))),
    }
    loss_cfg.setdefault("peak_weight_enabled", False)
    loss_cfg.setdefault("peak_quantile", 0.75)
    loss_cfg.setdefault("peak_weight", 0.2)
    loss_cfg.setdefault("swap_detach_inv", True)
    loss_cfg.setdefault("swap_detach_env", False)
    ds_cfg.setdefault("null_to_num", 0.0)
    ds_cfg.setdefault("frequency_minutes", 5)
    ds_cfg.setdefault("auto_generate_timestamps", True)
    resolved_null_val = ds_cfg.get("null_val", None)
    if resolved_null_val is None and loss_cfg.get("null_val", None) is not None:
        resolved_null_val = loss_cfg.get("null_val")
    if ds_cfg.get("null_val", None) is not None and loss_cfg.get("null_val", None) not in {None, ds_cfg.get("null_val")}:
        warnings.warn(
            f"DATASET.null_val={ds_cfg.get('null_val')!r} overrides LOSS.null_val={loss_cfg.get('null_val')!r}.",
            RuntimeWarning,
        )
    ds_cfg["null_val"] = resolved_null_val
    loss_cfg["null_val"] = resolved_null_val
    train_loss_scale = str(loss_cfg.get("train_loss_scale", "normalized")).lower()
    if train_loss_scale not in {"normalized", "original"}:
        raise ValueError("LOSS.train_loss_scale must be 'normalized' or 'original'")
    loss_cfg["train_loss_scale"] = train_loss_scale

    if model_cfg.get("use_time_embedding", False):
        raise NotImplementedError(
            "MODEL.use_time_embedding=True is not implemented yet. "
            "Current invariant backbones use historical values and optional node embeddings only."
        )
    if model_cfg.get("adaptive_adj", False):
        raise NotImplementedError(
            "MODEL.adaptive_adj=True is not implemented at the NUE-STG top level. "
            "Use backbone-specific adaptive adjacency options, such as MODEL.backbone.graph_wavenet.addaptadj."
        )
    env_neighbor_mix = model_cfg.get("env_neighbor_mix", "static_adj")
    if env_neighbor_mix not in (None, "static_adj"):
        raise NotImplementedError(
            f"MODEL.env_neighbor_mix={env_neighbor_mix!r} is not implemented. "
            "Current EnvEncoder supports self-only fallback and static_adj neighbor aggregation."
        )
    if loss_cfg.get("gate_label_mode", "potential_gain") != "potential_gain":
        raise NotImplementedError(
            "Only LOSS.gate_label_mode='potential_gain' is implemented. "
            "NUE-STG gate labels must use y_potential = y_inv + r_env."
        )
    swap_mode = swap_cfg.get("mode", "batch_node_random")
    if swap_mode != "batch_node_random":
        raise NotImplementedError(
            f"SWAP.mode={swap_mode!r} is not implemented. "
            "Current swap is batch_node_random; concept-shift pair mining is future work."
        )
    if swap_cfg.get("pair_mining", False):
        raise NotImplementedError(
            "SWAP.pair_mining=True is not implemented yet. "
            "Future work: history-similar/future-different and history-similar/future-similar pair mining."
        )
    if int(swap_cfg.get("num_swaps", 1)) != 1:
        raise NotImplementedError("SWAP.num_swaps other than 1 is not implemented in current random swap.")
    if loss_cfg.get("use_env_consistency", False) and swap_mode == "batch_node_random":
        raise ValueError(
            "LOSS.use_env_consistency=True is not allowed with SWAP.mode='batch_node_random'. "
            "Random env_perm should not be pulled toward the original env; enable this only with same-pair mining."
        )
    separation_cfg = model_cfg.setdefault("separation", {})
    separation_cfg.setdefault("enabled", False)
    separation_cfg.setdefault("mode", "none")
    if not separation_cfg.get("enabled", False):
        separation_cfg["mode"] = "none"
    if separation_cfg.get("mode") not in {
        "none",
        "orthogonal_projection",
        "basis_projection",
        "lowrank_residual",
    }:
        raise NotImplementedError(
            f"MODEL.separation.mode={separation_cfg.get('mode')!r} is not implemented."
        )
    if separation_cfg.get("mode") == "lowrank_residual":
        lowrank_target = (separation_cfg.get("lowrank", {}) or {}).get("target", "hidden")
        if lowrank_target != "hidden":
            raise NotImplementedError("Only MODEL.separation.lowrank.target='hidden' is implemented.")
    model_cfg["use_separated_z_for_y_inv"] = bool(
        model_cfg.get(
            "use_separated_z_for_y_inv",
            separation_cfg.get("use_separated_z_for_y_inv", True),
        )
    )
    separation_cfg["use_separated_z_for_y_inv"] = model_cfg["use_separated_z_for_y_inv"]

    for key in ["input_len", "output_len", "input_dim", "output_dim", "num_nodes"]:
        model_cfg[key] = ds_cfg[key]
    method_variant = str(model_cfg.get("method_variant", "nue")).lower()
    model_cfg["method_variant"] = method_variant
    if method_variant == "fpem":
        model_cfg["env_token_mode"] = True
    model_cfg["adj_path"] = ds_cfg.get("adj_path", model_cfg.get("adj_path", ""))
    backbone_cfg = model_cfg.setdefault("backbone", {})
    model_cfg["backbone_name"] = model_cfg.get("backbone_name", backbone_cfg.get("name", "stid_mlp"))
    backbone_cfg["name"] = model_cfg["backbone_name"]
    backbone_name_lower = str(model_cfg["backbone_name"]).lower()
    if train_cfg.get("teacher_forcing_enabled", False) and backbone_name_lower == "agcrn":
        warnings.warn(
            "TRAIN.teacher_forcing_enabled=True was requested for AGCRN, but the current AGCRN adapter "
            "uses a direct multi-horizon head and does not implement the official decoder scheduled sampling protocol.",
            RuntimeWarning,
        )
    if isinstance(model_cfg.get("GWNET"), dict):
        backbone_cfg.setdefault("graph_wavenet", {})
        backbone_cfg["graph_wavenet"].update(model_cfg["GWNET"])
    if isinstance(model_cfg.get("STNORM"), dict):
        stnorm_preset = dict(model_cfg["STNORM"])
        if "snorm" in stnorm_preset:
            stnorm_preset["snorm_bool"] = bool(stnorm_preset.pop("snorm"))
        if "tnorm" in stnorm_preset:
            stnorm_preset["tnorm_bool"] = bool(stnorm_preset.pop("tnorm"))
        backbone_cfg.setdefault("stnorm_wavenet", {})
        backbone_cfg["stnorm_wavenet"].update(stnorm_preset)
    model_cfg["baseline_name"] = model_cfg.get("baseline_name") or run_cfg.get("method") or model_cfg.get("name", "")
    model_cfg["reference_status"] = (
        model_cfg.get("reference_status")
        or run_cfg.get("reference_status")
        or "native_adapter"
    )
    run_cfg["reference_status"] = model_cfg["reference_status"]
    representation_dim = int(backbone_cfg.get("representation_dim", model_cfg.get("hidden_dim", 64)))
    backbone_key = str(model_cfg["backbone_name"]).lower()
    if backbone_key in {"graphwavenet", "gwnet"}:
        backbone_key = "graph_wavenet"
    if backbone_key in {"graphwavenet_full", "gwnet_full", "graphwavenet-full"}:
        backbone_key = "graph_wavenet_full"
    if backbone_key in {"mlp", "stid_like"}:
        backbone_key = "stid_mlp"
    if backbone_key in {"official_stid"}:
        backbone_key = "stid"
    if backbone_key in {"st_norm", "stnorm"}:
        backbone_key = "stnorm_wavenet"
    selected_cfg = backbone_cfg.get(backbone_key, {})
    if isinstance(selected_cfg, dict) and selected_cfg.get("representation_dim") is not None:
        representation_dim = int(selected_cfg["representation_dim"])
    backbone_cfg["representation_dim"] = representation_dim
    model_cfg["hidden_dim"] = representation_dim
    swap_cfg = cfg.setdefault("SWAP", {})
    if "detach_env" in swap_cfg:
        loss_cfg["swap_detach_env"] = bool(swap_cfg["detach_env"])
    else:
        swap_cfg["detach_env"] = bool(loss_cfg.get("swap_detach_env", False))
    swap_cfg.setdefault("freeze_predictor", True)
    model_cfg["swap"] = swap_cfg
    model_cfg["swap_detach_inv"] = bool(loss_cfg.get("swap_detach_inv", True))
    model_cfg["swap_detach_env"] = bool(swap_cfg.get("detach_env", False))
    cfg["LOSS"]["z_dim"] = representation_dim
    cfg["LOSS"]["env_dim"] = int(model_cfg.get("env_dim", 32))
    return cfg


def get_device(train_cfg: Dict) -> torch.device:
    requested = train_cfg.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def configure_torch_runtime(train_cfg: Dict) -> None:
    num_threads = train_cfg.get("torch_num_threads")
    if num_threads in (None, "", False):
        return
    torch.set_num_threads(max(1, int(num_threads)))


class IndexedForecastingDataset(Dataset):
    """Add original window-order sample_index while preserving BasicTS items."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def data(self):
        return getattr(self.dataset, "data")

    def __getitem__(self, index: int):
        item = self.dataset[index]
        sample_index = torch.as_tensor(index, dtype=torch.long)
        if isinstance(item, dict):
            out = dict(item)
            out["sample_index"] = sample_index
            return out
        if isinstance(item, tuple):
            return (*item, sample_index)
        if isinstance(item, list):
            return [*item, sample_index]
        raise TypeError(f"Unsupported dataset item type for sample_index wrapping: {type(item)!r}")


def build_dataset(cfg: Dict, split: str) -> BasicTSForecastingDataset:
    ds_cfg = cfg["DATASET"]
    maybe_generate_timestamp_file(ds_cfg["data_file_path"], split, ds_cfg)
    dataset = BasicTSForecastingDataset(
        dataset_name=ds_cfg["name"],
        input_len=ds_cfg["input_len"],
        output_len=ds_cfg["output_len"],
        mode=split,
        use_timestamps=ds_cfg.get("use_timestamps", False),
        data_file_path=ds_cfg["data_file_path"],
        memmap=ds_cfg.get("memmap", True),
    )
    if split == "train" and pseudo_env_enabled(cfg):
        return IndexedForecastingDataset(dataset)
    return dataset


def build_loader(cfg: Dict, split: str, shuffle: bool) -> DataLoader:
    train_cfg = cfg["TRAIN"]
    drop_last_default = split == "train"
    drop_last = bool(train_cfg.get(f"drop_last_{split}", drop_last_default))
    return DataLoader(
        build_dataset(cfg, split),
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", True) and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def resolve_target_mask_value(targets: torch.Tensor, cfg: Dict):
    ds_cfg = cfg["DATASET"]
    loss_cfg = cfg["LOSS"]
    null_val = ds_cfg.get("null_val", loss_cfg.get("null_val"))
    mode = str(loss_cfg.get("mask_value_mode", ds_cfg.get("mask_value_mode", "null_val")) or "null_val").lower()
    if mode in {"null_val", "config", "default"}:
        return null_val
    if mode in {"stexpert_min", "st_expert_min", "batch_min_if_lt_one"}:
        finite = targets.detach()[torch.isfinite(targets.detach())]
        if finite.numel() > 0:
            min_value = finite.min()
            if min_value < 1:
                return min_value.to(device=targets.device, dtype=targets.dtype)
        return targets.new_tensor(0.0)
    raise ValueError(
        "LOSS.mask_value_mode must be 'null_val' or 'stexpert_min', "
        f"got {mode!r}"
    )


def get_scaler_cfg(cfg: Dict) -> Dict:
    scaler_cfg = cfg.get("SCALER")
    if scaler_cfg is None:
        scaler_cfg = cfg.get("DATASET", {}).get("scaler", {})
    return scaler_cfg or {}


def build_data_scaler(cfg: Dict, device: torch.device) -> ZScoreDataScaler:
    scaler_cfg = get_scaler_cfg(cfg)
    if not scaler_cfg.get("enabled", True):
        return ZScoreDataScaler.identity().to(device)
    scaler_type = str(scaler_cfg.get("type", "zscore")).lower()
    if scaler_type not in {"zscore", "standard", "standardization"}:
        raise NotImplementedError(f"Only zscore/standard scaler is implemented, got {scaler_type!r}")
    if scaler_cfg.get("mean") is not None and scaler_cfg.get("std") is not None:
        return ZScoreDataScaler(
            scaler_cfg["mean"],
            scaler_cfg["std"],
            enabled=True,
            eps=float(scaler_cfg.get("eps", 1e-5)),
        ).to(device)
    train_data = build_dataset(cfg, "train").data
    scaler = ZScoreDataScaler.fit(
        train_data,
        null_val=cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
        norm_each_channel=bool(scaler_cfg.get("norm_each_channel", False)),
        eps=float(scaler_cfg.get("eps", 1e-5)),
    ).to(device)
    return scaler


def preprocess_batch(
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
    data_scaler: ZScoreDataScaler,
) -> Dict[str, torch.Tensor]:
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    null_val = cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val"))
    null_to_num = float(cfg["DATASET"].get("null_to_num", 0.0))
    processed = dict(batch)

    inputs_mask = make_valid_mask(batch[input_key], null_val)
    target_null_val = resolve_target_mask_value(batch[target_key], cfg)
    targets_mask = make_valid_mask(batch[target_key], target_null_val)
    inputs_scaled = data_scaler.transform(batch[input_key], inputs_mask)
    targets_scaled = data_scaler.transform(batch[target_key], targets_mask)

    processed[input_key] = torch.where(
        inputs_mask,
        inputs_scaled,
        torch.as_tensor(null_to_num, dtype=inputs_scaled.dtype, device=inputs_scaled.device),
    )
    processed[target_key] = torch.where(
        targets_mask,
        targets_scaled,
        torch.as_tensor(null_to_num, dtype=targets_scaled.dtype, device=targets_scaled.device),
    )
    processed["inputs_mask"] = inputs_mask
    processed["targets_mask"] = targets_mask
    return processed


def to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def get_time_kwargs(batch: Dict[str, torch.Tensor], cfg: Dict, include_future: bool) -> Dict[str, torch.Tensor]:
    ds_cfg = cfg["DATASET"]
    seq_key = ds_cfg.get("input_timestamp_key", "inputs_timestamps")
    future_key = ds_cfg.get("target_timestamp_key", "targets_timestamps")
    cur_key = ds_cfg.get("current_timestamp_key", "current_timestamps")
    seq_time = batch.get(seq_key)
    cur_time = batch.get(cur_key)
    if cur_time is None and isinstance(seq_time, torch.Tensor) and seq_time.dim() == 3:
        cur_time = seq_time[:, -1]
    out = {"seq_time": seq_time, "cur_time": cur_time}
    if include_future:
        out["future_time"] = batch.get(future_key)
    else:
        out["future_time"] = None
    return out


def pseudo_env_is_active(cfg: Dict, epoch: int) -> bool:
    loss_cfg = cfg.get("LOSS", {})
    return bool(loss_cfg.get("use_pseudo_env_heads", False)) and int(epoch) >= int(loss_cfg.get("pseudo_env_warmup_epochs", 0))


class PseudoEnvCache:
    """CPU cache for pseudo-env assignments keyed by original sample_index."""

    def __init__(self, num_samples: int, num_heads: int) -> None:
        self.num_samples = int(num_samples)
        self.num_heads = int(num_heads)
        self.cached_q = torch.full((self.num_samples, self.num_heads), 1.0 / self.num_heads, dtype=torch.float32)
        self.cached_hard_env = torch.zeros(self.num_samples, dtype=torch.long)
        self.cached_loss_head = torch.zeros((self.num_samples, self.num_heads), dtype=torch.float32)
        self.updated = torch.zeros(self.num_samples, dtype=torch.bool)

    def read(self, sample_index: torch.Tensor, device: torch.device, hard: bool = False) -> torch.Tensor:
        idx = sample_index.detach().long().cpu().clamp(0, self.num_samples - 1)
        if hard:
            q = torch.nn.functional.one_hot(self.cached_hard_env[idx], num_classes=self.num_heads).float()
        else:
            q = self.cached_q[idx].float()
        return q.to(device=device)

    def write(self, sample_index: torch.Tensor, loss_head: torch.Tensor, q_env: torch.Tensor) -> None:
        idx = sample_index.detach().long().cpu().clamp(0, self.num_samples - 1)
        self.cached_loss_head[idx] = loss_head.detach().float().cpu()
        self.cached_q[idx] = q_env.detach().float().cpu()
        self.cached_hard_env[idx] = q_env.detach().argmax(dim=-1).long().cpu()
        self.updated[idx] = True

    def smooth_hard_assignments(self, radius: int) -> None:
        radius = max(0, int(radius))
        if radius <= 0 or self.num_samples == 0:
            return
        hard = self.cached_hard_env.clone()
        smoothed = hard.clone()
        for index in range(self.num_samples):
            lo = max(0, index - radius)
            hi = min(self.num_samples, index + radius + 1)
            counts = torch.bincount(hard[lo:hi], minlength=self.num_heads)
            smoothed[index] = counts.argmax()
        self.cached_hard_env = smoothed
        self.cached_q = torch.nn.functional.one_hot(smoothed, num_classes=self.num_heads).float()

    def state_dict(self) -> Dict[str, torch.Tensor | int]:
        return {
            "num_samples": self.num_samples,
            "num_heads": self.num_heads,
            "cached_q": self.cached_q,
            "cached_hard_env": self.cached_hard_env,
            "cached_loss_head": self.cached_loss_head,
            "updated": self.updated,
        }

    def load_state_dict(self, state: Dict) -> None:
        if not isinstance(state, dict):
            return
        if int(state.get("num_samples", self.num_samples)) != self.num_samples:
            warnings.warn("PseudoEnvCache sample count differs from checkpoint; ignoring cached assignments.", RuntimeWarning)
            return
        if int(state.get("num_heads", self.num_heads)) != self.num_heads:
            warnings.warn("PseudoEnvCache head count differs from checkpoint; ignoring cached assignments.", RuntimeWarning)
            return
        for name in ["cached_q", "cached_hard_env", "cached_loss_head", "updated"]:
            value = state.get(name)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(getattr(self, name).shape):
                setattr(self, name, value.detach().cpu().clone())

    def summary(self) -> Dict[str, float | list[int]]:
        active = self.updated if bool(self.updated.any()) else torch.ones_like(self.updated, dtype=torch.bool)
        q = self.cached_q[active]
        hard = self.cached_hard_env[active]
        counts = torch.bincount(hard, minlength=self.num_heads).tolist()
        entropy = (-(q * (q + 1e-8).log()).sum(dim=-1).mean()).item() if q.numel() else 0.0
        qmax = q.max(dim=-1).values.mean().item() if q.numel() else 0.0
        return {
            "counts": [int(value) for value in counts],
            "entropy": float(entropy),
            "qmax": float(qmax),
            "updated": int(active.sum().item()),
        }


def _ensure_bhnc_for_pseudo(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(-1)
    if tensor.dim() == 4:
        return tensor
    raise AssertionError(f"pseudo-env cache target must be [B,H,N] or [B,H,N,C], got {tuple(tensor.shape)}")


def compute_pseudo_env_loss_head_for_cache(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    targets_mask: torch.Tensor | None,
    cfg: Dict,
    data_scaler: ZScoreDataScaler,
) -> torch.Tensor:
    head_pred = output.get("pseudo_env_head_pred")
    if not isinstance(head_pred, torch.Tensor):
        raise RuntimeError("Pseudo-env cache update requires output['pseudo_env_head_pred'].")
    if str(cfg["LOSS"].get("train_loss_scale", "normalized")).lower() == "original":
        head_pred = data_scaler.inverse_transform(head_pred)
    targets = _ensure_bhnc_for_pseudo(targets)
    horizon = head_pred.shape[2]
    if targets.shape[1] != horizon:
        targets = targets[:, :horizon]
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


def maybe_attach_pseudo_env_assignment(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
    cache: PseudoEnvCache | None,
    epoch: int,
    cache_updated: bool,
) -> None:
    if not pseudo_env_is_active(cfg, epoch):
        return
    loss_cfg = cfg["LOSS"]
    output["pseudo_env_cache_updated"] = bool(cache_updated)
    output["pseudo_env_smoothing_enabled"] = bool(loss_cfg.get("pseudo_env_use_temporal_smoothing", True))
    if (
        cache is None
        or not bool(loss_cfg.get("pseudo_env_use_global_cache", True))
        or str(loss_cfg.get("pseudo_env_level", "window")).lower() != "window"
        or "sample_index" not in batch
    ):
        return
    mode = str(loss_cfg.get("pseudo_env_assignment_mode", "cached_soft")).lower()
    if mode not in {"cached_soft", "cached_hard"}:
        return
    output["pseudo_env_q_weight"] = cache.read(
        batch["sample_index"],
        device=output["prediction"].device,
        hard=(mode == "cached_hard"),
    )


@torch.no_grad()
def update_pseudo_env_cache(
    model: torch.nn.Module,
    train_dataset: Dataset,
    cfg: Dict,
    device: torch.device,
    data_scaler: ZScoreDataScaler,
    cache: PseudoEnvCache,
) -> None:
    loss_cfg = cfg["LOSS"]
    if str(loss_cfg.get("pseudo_env_level", "window")).lower() != "window":
        warnings.warn("Pseudo-env global cache currently supports window-level assignments only; using batch assignments.", RuntimeWarning)
        return
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["TRAIN"]["batch_size"],
        shuffle=False,
        num_workers=cfg["TRAIN"].get("num_workers", 0),
        pin_memory=cfg["TRAIN"].get("pin_memory", True) and torch.cuda.is_available(),
        drop_last=False,
    )
    was_training = model.training
    model.eval()
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    tau = max(float(loss_cfg.get("pseudo_env_tau", 1.0)), 1e-6)
    for raw_batch in loader:
        raw_batch = to_device_batch(raw_batch, device)
        if "sample_index" not in raw_batch:
            continue
        batch = preprocess_batch(raw_batch, cfg, data_scaler)
        output = model(
            batch[input_key],
            y_true=batch[target_key],
            **get_time_kwargs(batch, cfg, include_future=True),
        )
        targets_for_loss = raw_batch[target_key] if str(loss_cfg.get("train_loss_scale", "normalized")).lower() == "original" else batch[target_key]
        loss_head = compute_pseudo_env_loss_head_for_cache(
            output,
            targets_for_loss,
            batch.get("targets_mask"),
            cfg,
            data_scaler,
        )
        q_env = torch.softmax(-loss_head / tau, dim=-1)
        cache.write(raw_batch["sample_index"], loss_head, q_env)
    if bool(loss_cfg.get("pseudo_env_use_temporal_smoothing", True)):
        cache.smooth_hard_assignments(int(loss_cfg.get("pseudo_env_smooth_radius", 2)))
    if was_training:
        model.train()


class TimeChannelSoftGradientConsensus:
    """Gradient-level TC-SGC regularizer for invariant representations.

    TC-SGC does not construct environments or add an explicit loss. It is
    registered only on invariant encoder outputs and softly interpolates the
    grad_z returned to that encoder. Downstream modules still compute their own
    parameter gradients, losses, and optimizer updates through the original
    training graph; no feature is hard-masked, thresholded, or removed.
    """

    def __init__(self, cfg: Optional[Dict]) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.target = str(cfg.get("target", "z_seq") or "z_seq")
        self.apply_to = str(cfg.get("apply_to", "inv_branch") or "inv_branch")
        self.mode = str(cfg.get("mode", "time_channel") or "time_channel")
        self.rho_max = max(0.0, float(cfg.get("rho_max", 0.1)))
        self.gamma = max(0.0, float(cfg.get("gamma", 1.0)))
        self.ema_beta = min(max(float(cfg.get("ema_beta", 0.95)), 0.0), 0.999999)
        self.warmup_epochs = int(cfg.get("warmup_epochs", 10))
        self.eps = max(float(cfg.get("eps", 1e-8)), 1e-12)
        self.sand_alpha = max(0.0, float(cfg.get("sand_alpha", cfg.get("alpha", 1.0))))
        self.use_ema = bool(cfg.get("use_ema", True))
        self.log_stats = bool(cfg.get("log_stats", True))
        self.loss_type = str(cfg.get("loss_type", "mse") or "mse")
        self.ema: Optional[torch.Tensor] = None
        self.latest_stats = self._empty_stats(False)
        self._warned: set[str] = set()

    @staticmethod
    def _empty_stats(using_fallback: bool) -> Dict[str, float]:
        return {
            "grad_consensus/agree_mean": 0.0,
            "grad_consensus/agree_std": 0.0,
            "grad_consensus/agree_min": 0.0,
            "grad_consensus/agree_max": 0.0,
            "grad_consensus/rho_mean": 0.0,
            "grad_consensus/rho_max": 0.0,
            "grad_consensus/using_fallback_z_inv": float(using_fallback),
            "grad_consensus/ema_agree_mean": 0.0,
            "grad_consensus/mag_var_mean": 0.0,
            "grad_consensus/mag_stability_mean": 0.0,
        }

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, RuntimeWarning)
            self._warned.add(key)

    def _to_btnd(self, grad: torch.Tensor):
        if grad.dim() == 3:
            return grad.unsqueeze(1), lambda item: item.squeeze(1)
        if grad.dim() != 4:
            return None, None
        # Expected z_seq is [B,T,N,D]. Some backbones may expose [B,D,N,T];
        # use the channel/time size heuristic and restore the original layout.
        if grad.shape[1] > grad.shape[-1]:
            return grad.permute(0, 3, 2, 1).contiguous(), lambda item: item.permute(0, 3, 2, 1).contiguous()
        return grad, lambda item: item

    def _make_hook(self, current_epoch: Optional[int], using_fallback: bool):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if current_epoch is not None and int(current_epoch) <= self.warmup_epochs:
                self.latest_stats = self._empty_stats(using_fallback)
                return grad
            with torch.no_grad():
                grad_btnd, restore = self._to_btnd(grad)
                if grad_btnd is None or restore is None:
                    self._warn_once(
                        "bad_grad_shape",
                        f"TC-SGC expected grad rank 3 or 4, got {tuple(grad.shape)}; skipping consensus.",
                    )
                    self.latest_stats = self._empty_stats(using_fallback)
                    return grad

                grad_float = grad_btnd.detach().float()
                sign_g = grad_float.sign()
                mag_var = None
                mag_stability = None
                if self.mode == "sand_tc":
                    # Advanced SAND-TC mode stays strictly inside the z hook:
                    # it uses only the batch-node distribution of grad_z for
                    # each time-channel location, with no environment/domain/
                    # group labels, proxy task, or virtual partition.
                    mean_sign = sign_g.mean(dim=(0, 2))
                    sign_agree = mean_sign.abs().clamp(0.0, 1.0)
                    abs_grad = grad_float.abs()
                    mag_var = abs_grad.var(dim=(0, 2), unbiased=False)
                    norm_mag_var = mag_var / mag_var.mean().clamp_min(self.eps)
                    mag_stability = torch.exp(-self.sand_alpha * norm_mag_var).clamp(0.0, 1.0)
                    agreement_input = (sign_agree * mag_stability).detach()
                    if self.ema is None or tuple(self.ema.shape) != tuple(agreement_input.shape):
                        self.ema = torch.zeros_like(agreement_input)
                    self.ema = self.ema.to(device=agreement_input.device, dtype=agreement_input.dtype)
                    self.ema.mul_(self.ema_beta).add_(agreement_input, alpha=1.0 - self.ema_beta)
                    agreement = self.ema.abs().clamp(0.0, 1.0)
                    direction = grad_float.mean(dim=(0, 2)).sign()
                    rho = (self.rho_max * agreement.clamp_min(0.0).pow(self.gamma)).clamp(0.0, self.rho_max)
                    view_shape = (1, agreement.shape[0], 1, agreement.shape[1])
                    consensus_grad = grad_float.abs() * direction.view(view_shape)
                    rho_view = rho.view(view_shape)
                    new_grad = (1.0 - rho_view) * grad_float + rho_view * consensus_grad
                elif self.mode == "time_channel":
                    m_td = sign_g.mean(dim=(0, 2))
                    if self.use_ema:
                        if self.ema is None or tuple(self.ema.shape) != tuple(m_td.shape):
                            self.ema = torch.zeros_like(m_td)
                        self.ema = self.ema.to(device=m_td.device, dtype=m_td.dtype)
                        self.ema.mul_(self.ema_beta).add_(m_td.detach(), alpha=1.0 - self.ema_beta)
                        consensus_map = self.ema
                    else:
                        consensus_map = m_td.detach()

                    agreement = consensus_map.abs().clamp(0.0, 1.0)
                    direction = consensus_map.sign()
                    rho = (self.rho_max * agreement.clamp_min(0.0).pow(self.gamma)).clamp(0.0, self.rho_max)

                    view_shape = (1, agreement.shape[0], 1, agreement.shape[1])
                    consensus_grad = grad_float.abs() * direction.view(view_shape)
                    rho_view = rho.view(view_shape)
                    new_grad = (1.0 - rho_view) * grad_float + rho_view * consensus_grad
                elif self.mode == "channel":
                    m_d = sign_g.mean(dim=(0, 1, 2))
                    m_td = m_d.unsqueeze(0).expand(grad_float.shape[1], -1)
                    if self.use_ema:
                        if self.ema is None or tuple(self.ema.shape) != tuple(m_td.shape):
                            self.ema = torch.zeros_like(m_td)
                        self.ema = self.ema.to(device=m_td.device, dtype=m_td.dtype)
                        self.ema.mul_(self.ema_beta).add_(m_td.detach(), alpha=1.0 - self.ema_beta)
                        consensus_map = self.ema
                    else:
                        consensus_map = m_td.detach()

                    agreement = consensus_map.abs().clamp(0.0, 1.0)
                    direction = consensus_map.sign()
                    rho = (self.rho_max * agreement.clamp_min(0.0).pow(self.gamma)).clamp(0.0, self.rho_max)

                    view_shape = (1, agreement.shape[0], 1, agreement.shape[1])
                    consensus_grad = grad_float.abs() * direction.view(view_shape)
                    rho_view = rho.view(view_shape)
                    new_grad = (1.0 - rho_view) * grad_float + rho_view * consensus_grad
                else:
                    self._warn_once(
                        "bad_mode",
                        f"TC-SGC mode={self.mode!r} is unsupported; expected 'time_channel', 'channel', or 'sand_tc'.",
                    )
                    self.latest_stats = self._empty_stats(using_fallback)
                    return grad

                if self.log_stats:
                    agreement_cpu = agreement.detach().cpu()
                    rho_cpu = rho.detach().cpu()
                    mag_var_mean = float(mag_var.detach().mean().cpu()) if mag_var is not None else 0.0
                    mag_stability_mean = (
                        float(mag_stability.detach().mean().cpu()) if mag_stability is not None else 0.0
                    )
                    self.latest_stats = {
                        "grad_consensus/agree_mean": float(agreement_cpu.mean()),
                        "grad_consensus/agree_std": float(agreement_cpu.std(unbiased=False)),
                        "grad_consensus/agree_min": float(agreement_cpu.min()),
                        "grad_consensus/agree_max": float(agreement_cpu.max()),
                        "grad_consensus/rho_mean": float(rho_cpu.mean()),
                        "grad_consensus/rho_max": float(rho_cpu.max()),
                        "grad_consensus/using_fallback_z_inv": float(using_fallback),
                        "grad_consensus/ema_agree_mean": (
                            float(self.ema.detach().abs().mean().cpu())
                            if self.ema is not None and (self.use_ema or self.mode == "sand_tc")
                            else 0.0
                        ),
                        "grad_consensus/mag_var_mean": mag_var_mean,
                        "grad_consensus/mag_stability_mean": mag_stability_mean,
                    }
                return restore(new_grad.to(dtype=grad.dtype))

        return hook

    def register(self, z_tensor: Optional[torch.Tensor], current_epoch: Optional[int] = None, using_fallback: bool = False):
        self.latest_stats = self._empty_stats(using_fallback)
        if not self.enabled:
            return None
        if self.apply_to != "inv_branch":
            self._warn_once("apply_to", "TC-SGC currently supports apply_to='inv_branch' only; skipping consensus.")
            return None
        if not isinstance(z_tensor, torch.Tensor):
            self._warn_once("missing_tensor", "TC-SGC target tensor is missing; skipping consensus.")
            return None
        if not z_tensor.requires_grad:
            self._warn_once("no_grad", "TC-SGC target tensor does not require grad; skipping consensus.")
            return None
        # The hook only replaces the gradient sent to this tensor's creators.
        # Parameter gradients in environment, mask, fusion, predictor, losses,
        # and optimizer logic are left to the original backward pass.
        return z_tensor.register_hook(self._make_hook(current_epoch, using_fallback))

    def log_tensors(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.enabled or not self.log_stats:
            return {}
        return {
            key: like.new_tensor(float(value))
            for key, value in self.latest_stats.items()
        }


def register_grad_consensus_hook(
    output: Dict[str, torch.Tensor],
    grad_consensus: TimeChannelSoftGradientConsensus,
    epoch: Optional[int],
) -> None:
    if not grad_consensus.enabled:
        return
    grad_consensus.latest_stats = grad_consensus._empty_stats(False)
    target = grad_consensus.target
    using_fallback = False
    if target == "z_seq":
        z_tensor = output.get("grad_consensus_z_seq")
        if not isinstance(z_tensor, torch.Tensor):
            z_tensor = output.get("grad_consensus_z_inv")
            using_fallback = True
            grad_consensus._warn_once(
                "fallback_z_inv",
                "TC-SGC hook target z_seq is unavailable; falling back to invariant-encoder z_inv as T=1.",
            )
    elif target == "z_inv":
        z_tensor = output.get("grad_consensus_z_inv")
    else:
        z_tensor = None
        grad_consensus._warn_once(
            "bad_target",
            "TC-SGC target must be 'z_seq' or 'z_inv'; skipping consensus.",
        )
    if not isinstance(z_tensor, torch.Tensor):
        grad_consensus.latest_stats = grad_consensus._empty_stats(using_fallback)
        grad_consensus._warn_once(
            "missing_target",
            f"TC-SGC hook tensor for target {target!r} is unavailable; skipping consensus.",
        )
        return
    grad_consensus.register(z_tensor, current_epoch=epoch, using_fallback=using_fallback)


class InvariantGradientSurgery:
    """Conflict-aware gradient surgery for invariant encoder parameters only.

    This module does not alter the forward graph and does not introduce any
    environment/domain/group/proxy partition. It computes primary and auxiliary
    gradients for the selected invariant encoder parameters, lets the normal
    total loss backward populate every module's gradients, then replaces only
    those invariant encoder gradients with the PCGrad-composed result.
    """

    def __init__(self, cfg: Optional[Dict]) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.method = str(cfg.get("method", "pcgrad") or "pcgrad").lower()
        self.apply_to = str(cfg.get("apply_to", "inv_encoder") or "inv_encoder")
        self.primary_losses = list(cfg.get("primary_losses", ["pred", "inv"]) or [])
        self.aux_losses = list(cfg.get("aux_losses", ["envpred", "future_mi", "swap", "sep", "sparse"]) or [])
        self.eps = max(float(cfg.get("eps", 1e-12)), 1e-12)
        self.latest_stats = self._empty_stats()
        self._warned: set[str] = set()

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {
            "grad_surgery/enabled": 0.0,
            "grad_surgery/conflict_cos": 0.0,
            "grad_surgery/conflict_dot": 0.0,
            "grad_surgery/projection_norm": 0.0,
            "grad_surgery/aux_grad_norm": 0.0,
            "grad_surgery/primary_grad_norm": 0.0,
        }

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, RuntimeWarning)
            self._warned.add(key)

    def select_params(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        if not self.enabled:
            return []
        if self.apply_to != "inv_encoder":
            self._warn_once(
                "apply_to",
                "grad_surgery currently supports apply_to='inv_encoder' only; skipping surgery.",
            )
            return []
        modules = []
        backbone = getattr(model, "backbone", None)
        if isinstance(backbone, torch.nn.Module):
            modules.append(backbone)
        z_time_adapter = getattr(model, "z_time_adapter", None)
        if isinstance(z_time_adapter, torch.nn.Module):
            modules.append(z_time_adapter)
        params: list[torch.nn.Parameter] = []
        seen = set()
        for module in modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        if not params:
            self._warn_once("no_params", "grad_surgery found no invariant encoder parameters; skipping surgery.")
        return params

    @staticmethod
    def _sum_terms(terms: Dict[str, torch.Tensor], names: list[str], like: torch.Tensor) -> torch.Tensor:
        total = like.new_zeros(())
        for name in names:
            value = terms.get(name)
            if isinstance(value, torch.Tensor):
                total = total + value
        return total

    def _flat_grads(
        self,
        loss: torch.Tensor,
        params: list[torch.nn.Parameter],
        scale: float,
        retain_graph: bool,
    ) -> torch.Tensor:
        if not loss.requires_grad:
            flat_zeros = [
                torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1)
                for param in params
            ]
            if not flat_zeros:
                return loss.new_zeros((0,))
            return torch.cat(flat_zeros)
        grads = torch.autograd.grad(
            loss * scale,
            params,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        flat = []
        for param, grad in zip(params, grads):
            if grad is None:
                flat.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        if not flat:
            return loss.new_zeros((0,))
        return torch.cat(flat)

    def prepare(
        self,
        loss_terms: Optional[Dict[str, torch.Tensor]],
        params: list[torch.nn.Parameter],
        scale: float,
        like: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        self.latest_stats = self._empty_stats()
        if not self.enabled:
            return None
        if self.method == "cagrad":
            self._warn_once("cagrad", "grad_surgery method='cagrad' is not implemented; falling back to pcgrad.")
        elif self.method != "pcgrad":
            raise ValueError("LOSS.grad_surgery.method must be 'pcgrad' or 'cagrad'")
        if not params or not isinstance(loss_terms, dict):
            return None

        primary_loss = self._sum_terms(loss_terms, self.primary_losses, like)
        aux_loss = self._sum_terms(loss_terms, self.aux_losses, like)
        primary_grad = self._flat_grads(primary_loss, params, scale, retain_graph=True)
        aux_grad = self._flat_grads(aux_loss, params, scale, retain_graph=True)
        if primary_grad.numel() == 0:
            return None

        dot = torch.dot(aux_grad.float(), primary_grad.float())
        primary_norm = primary_grad.float().norm()
        aux_norm = aux_grad.float().norm()
        projection_norm = primary_grad.new_zeros(())
        aux_projected = aux_grad
        has_conflict = bool((dot < 0).detach().item()) and bool((primary_norm > self.eps).detach().item())
        if has_conflict:
            projection = (dot / primary_norm.pow(2).clamp_min(self.eps)).to(aux_grad.dtype) * primary_grad
            aux_projected = aux_grad - projection
            projection_norm = projection.float().norm().to(primary_grad.dtype)
        final_grad = primary_grad + aux_projected
        cos = dot / (primary_norm * aux_norm).clamp_min(self.eps)
        self.latest_stats = {
            "grad_surgery/enabled": 1.0,
            "grad_surgery/conflict_cos": float(cos.detach().cpu()),
            "grad_surgery/conflict_dot": float(dot.detach().cpu()),
            "grad_surgery/projection_norm": float(projection_norm.detach().cpu()),
            "grad_surgery/aux_grad_norm": float(aux_norm.detach().cpu()),
            "grad_surgery/primary_grad_norm": float(primary_norm.detach().cpu()),
        }
        return {"final_grad": final_grad.detach()}

    def apply(self, params: list[torch.nn.Parameter], prepared: Optional[Dict[str, torch.Tensor]]) -> None:
        if not self.enabled or prepared is None:
            return
        flat = prepared["final_grad"]
        offset = 0
        for param in params:
            numel = param.numel()
            grad = flat[offset: offset + numel].view_as(param).to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = grad.clone(memory_format=torch.preserve_format)
            else:
                param.grad.detach().copy_(grad)
            offset += numel

    def log_tensors(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        return {key: like.new_tensor(float(value)) for key, value in self.latest_stats.items()}


class BackboneOnlyForecastModel(torch.nn.Module):
    """Forecast directly from the configured backbone without NUE/FPEM modules."""

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.backbone = build_backbone({"MODEL": model_cfg})
        self.output_len = int(model_cfg["output_len"])
        self.output_dim = int(model_cfg["output_dim"])
        self.env_dim = int(model_cfg.get("env_dim", 1))
        self.baseline_name = model_cfg.get("baseline_name") or model_cfg.get("name") or model_cfg.get("backbone_name", "")
        self.reference_status = model_cfg.get("reference_status", "")

    def forward(
        self,
        inputs: torch.Tensor,
        y_true: torch.Tensor | None = None,
        seq_time: torch.Tensor | None = None,
        cur_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor | None]:
        del y_true, kwargs
        x = ensure_blnc(inputs, "inputs")
        backbone_out = self.backbone(
            x,
            seq_time=seq_time,
            cur_time=cur_time,
            future_time=future_time,
        )
        prediction = backbone_out["y_inv"]
        z_inv = backbone_out["z_inv"]
        z_seq = backbone_out.get("z_seq")
        batch_size, _, num_nodes, _ = prediction.shape
        env = prediction.new_zeros(batch_size, num_nodes, self.env_dim)
        rho = prediction.new_zeros(batch_size, self.output_len, num_nodes, 1)
        r_env = prediction.new_zeros(batch_size, self.output_len, num_nodes, self.output_dim)
        return {
            "method_variant": "backbone_only",
            "baseline_name": self.baseline_name,
            "reference_status": self.reference_status,
            "prediction": prediction,
            "y_inv": prediction,
            "y_potential": prediction,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "z_raw": z_inv,
            "z_seq": z_seq,
            "grad_consensus_z_inv": z_inv,
            "grad_consensus_z_seq": z_seq,
            "env_mu": env,
            "env_logvar": env,
            "env": env,
            "env_hist": env,
            "env_raw": env,
            "y_inv_raw": prediction,
            "separation_mode": "none",
            "separation_extra": {},
            "env_fut": None,
            "persist_q": None,
            "persist_k": None,
            "persist_score": None,
            "persistence_enabled": False,
            "prediction_swap": None,
            "rho_swap": None,
            "env_perm": None,
            "env_perm_index": None,
            "backbone_aux_losses": backbone_out.get("backbone_aux_losses", {}),
            "backbone_aux_weights": backbone_out.get("backbone_aux_weights", {}),
        }


def build_model_and_loss(cfg: Dict, device: torch.device) -> Tuple[torch.nn.Module, NUESTGLoss]:
    method_variant = str(cfg["MODEL"].get("method_variant", "nue") or "nue").lower()
    if method_variant in {"backbone_only", "backbone", "graphwavenet_only", "pure_graphwavenet"}:
        model = BackboneOnlyForecastModel(cfg["MODEL"]).to(device)
    else:
        model = NUESTG(NUESTGConfig(**cfg["MODEL"])).to(device)
    loss_fn = NUESTGLoss(**cfg["LOSS"]).to(device)
    return model, loss_fn


def build_optimizer(cfg: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = cfg["TRAIN"]
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    learning_rate = train_cfg["learning_rate"]
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_kwargs = {"lr": learning_rate}
    if weight_decay > 0 and bool(train_cfg.get("no_decay_for_bias_norm_emb", False)):
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            no_decay = (
                param.ndim <= 1
                or lowered.endswith(".bias")
                or "norm" in lowered
                or "embedding" in lowered
                or "emb" in lowered
            )
            if no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        params = []
        if decay_params:
            params.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            params.append({"params": no_decay_params, "weight_decay": 0.0})
    else:
        params = model.parameters()
        optimizer_kwargs["weight_decay"] = weight_decay
    if optimizer_name == "adam":
        return torch.optim.Adam(params, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer={optimizer_name!r}")


def build_lr_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer):
    train_cfg = cfg["TRAIN"]
    name = str(train_cfg.get("lr_scheduler", "none") or "none").lower()
    if name == "none":
        return None
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(train_cfg.get("lr_milestones", [30, 60, 80])),
            gamma=float(train_cfg.get("lr_gamma", 0.3)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(train_cfg.get("epochs", 1)), 1),
            eta_min=float(train_cfg.get("lr_min", 1e-5)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_gamma", 0.3)),
            patience=int(train_cfg.get("lr_plateau_patience", 5)),
            min_lr=float(train_cfg.get("lr_min", 1e-5)),
        )
    raise ValueError(f"Unsupported TRAIN.lr_scheduler={name!r}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


def curriculum_horizon(train_cfg: Dict, epoch: int, full_horizon: int) -> int:
    if not bool(train_cfg.get("curriculum_enabled", False)):
        return int(full_horizon)
    full_horizon = int(full_horizon)
    start = max(1, min(int(train_cfg.get("curriculum_start_horizon", full_horizon)), full_horizon))
    full_epoch = max(1, int(train_cfg.get("curriculum_full_horizon_epoch", 1)))
    if epoch >= full_epoch or start >= full_horizon:
        return full_horizon
    progress = max(epoch - 1, 0) / max(full_epoch - 1, 1)
    horizon = int(np.ceil(start + (full_horizon - start) * progress))
    return max(start, min(horizon, full_horizon))


def slice_for_train_horizon(
    output: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    targets_mask: torch.Tensor | None,
    raw_y_true: torch.Tensor | None,
    horizon: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    full_horizon = int(output["prediction"].shape[1])
    horizon = max(1, min(int(horizon), full_horizon))
    if horizon >= full_horizon:
        return output, y_true, targets_mask, raw_y_true
    horizon_keys = {
        "prediction",
        "y_inv",
        "y_inv_aug",
        "y_potential",
        "r_env",
        "rho",
        "y_inv_raw",
        "prediction_swap",
        "rho_swap",
        "env_fut_tokens",
        "env_fut_mu_tokens",
        "env_fut_logvar_tokens",
        "pred_fut_mu",
        "pred_fut_logvar",
        "pred_fut_mu_minus",
        "pred_fut_logvar_minus",
        "future_time_emb",
        "pseudo_env_head_pred",
        "y_route_soft",
        "y_route_hard",
        "y_route",
        "y_route_selected",
        "y_global",
        "y_route_final",
        "prediction_fused",
    }
    sliced = dict(output)
    for key in horizon_keys:
        value = sliced.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == full_horizon:
            sliced[key] = value[:, :horizon]
    pseudo_head = sliced.get("pseudo_env_head_pred")
    if isinstance(pseudo_head, torch.Tensor) and pseudo_head.dim() == 5 and pseudo_head.shape[2] == full_horizon:
        sliced["pseudo_env_head_pred"] = pseudo_head[:, :, :horizon]
    route_heads = sliced.get("y_route_heads")
    if isinstance(route_heads, torch.Tensor) and route_heads.dim() == 5 and route_heads.shape[2] == full_horizon:
        sliced["y_route_heads"] = route_heads[:, :, :horizon]
    y_true_sliced = y_true[:, :horizon] if isinstance(y_true, torch.Tensor) and y_true.shape[1] >= horizon else y_true
    mask_sliced = (
        targets_mask[:, :horizon]
        if isinstance(targets_mask, torch.Tensor) and targets_mask.shape[1] >= horizon
        else targets_mask
    )
    raw_sliced = (
        raw_y_true[:, :horizon]
        if isinstance(raw_y_true, torch.Tensor) and raw_y_true.shape[1] >= horizon
        else raw_y_true
    )
    return sliced, y_true_sliced, mask_sliced, raw_sliced


def describe_backbone_features(model: NUESTG) -> Dict[str, bool]:
    backbone = getattr(model, "backbone", None)
    inner_backbone = getattr(backbone, "model", None)
    return {
        "node_embedding": bool(
            getattr(backbone, "node_emb", None) is not None
            or getattr(backbone, "node_embeddings", None) is not None
            or getattr(backbone, "nodevec1", None) is not None
            or getattr(inner_backbone, "nodevec1", None) is not None
        ),
        "time_of_day_embedding": bool(getattr(backbone, "time_in_day_emb", None) is not None),
        "day_of_week_embedding": bool(getattr(backbone, "day_in_week_emb", None) is not None),
        "time_of_day_channel": bool(
            getattr(backbone, "use_time_of_day_channel", False)
            and int(getattr(backbone, "in_dim", getattr(backbone, "input_dim", 0)))
            > int(getattr(backbone, "input_dim", 0))
        ),
        "day_of_week_channel": bool(
            getattr(backbone, "use_day_of_week_channel", False)
            and int(getattr(backbone, "in_dim", getattr(backbone, "input_dim", 0)))
            > int(getattr(backbone, "input_dim", 0)) + 1
        ),
    }


def check_output_shapes(output: Dict[str, torch.Tensor], targets: torch.Tensor, cfg: Dict) -> None:
    batch_size = targets.shape[0]
    input_len = cfg["DATASET"]["input_len"]
    output_len = cfg["DATASET"]["output_len"]
    num_nodes = cfg["DATASET"]["num_nodes"]
    output_dim = cfg["DATASET"]["output_dim"]
    representation_dim = int(cfg["MODEL"].get("backbone", {}).get("representation_dim", cfg["MODEL"]["hidden_dim"]))
    method_variant = output.get("method_variant", cfg["MODEL"].get("method_variant", "nue"))
    print(f"method_variant: {method_variant}")
    print(f"baseline_name: {output.get('baseline_name', cfg['MODEL'].get('baseline_name', ''))}")
    print(f"reference_status: {output.get('reference_status', cfg['MODEL'].get('reference_status', ''))}")
    if method_variant == "fpem":
        expected = {
            "prediction": (batch_size, output_len, num_nodes, output_dim),
            "y_inv": (batch_size, output_len, num_nodes, output_dim),
            "y_potential": (batch_size, output_len, num_nodes, output_dim),
            "r_env": (batch_size, output_len, num_nodes, output_dim),
            "rho": (batch_size, output_len, num_nodes, 1),
            "z_inv": (batch_size, num_nodes, representation_dim),
            "z_raw": (batch_size, num_nodes, representation_dim),
            "env_mu": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_logvar": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_tokens": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_hist_tokens": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_hist_mu_tokens": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_hist_logvar_tokens": (batch_size, input_len, num_nodes, cfg["MODEL"]["env_dim"]),
            "env": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_hist": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_hist_bar": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_raw": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_plus": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "env_minus": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
            "mask": (batch_size, input_len, num_nodes, 1),
            "y_inv_raw": (batch_size, output_len, num_nodes, output_dim),
        }
        for key, expected_shape in expected.items():
            value = output[key]
            print(f"{key}: {tuple(value.shape)}")
            if tuple(value.shape) != expected_shape:
                raise AssertionError(f"{key} expected {expected_shape}, got {tuple(value.shape)}")
            assert_finite(value, key)
        for key in [
            "z_seq",
            "env_fut",
            "env_fut_tokens",
            "env_fut_mu_tokens",
            "env_fut_logvar_tokens",
            "pred_fut_mu",
            "pred_fut_logvar",
            "prediction_swap",
            "env_perm",
            "fusion_gamma",
            "fusion_beta",
            "seq_time_emb",
            "cur_time_emb",
            "future_time_emb",
            "pseudo_env_head_pred",
            "y_route_heads",
            "env_route_logits",
            "env_route_q",
            "env_route_entropy",
            "env_route_alpha",
            "y_route_soft",
            "y_route_hard",
            "y_route",
            "y_route_selected",
            "y_global",
            "y_route_final",
            "route_confidence",
            "prediction_fused",
        ]:
            value = output.get(key)
            if value is None:
                print(f"{key}: None")
            else:
                print(f"{key}: {tuple(value.shape)}")
                assert_finite(value, key)
        if pseudo_env_enabled(cfg):
            expected_pseudo = (batch_size, int(cfg["LOSS"].get("pseudo_env_k", 3)), output_len, num_nodes, output_dim)
            value = output.get("pseudo_env_head_pred")
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_pseudo:
                raise AssertionError(f"pseudo_env_head_pred expected {expected_pseudo}, got {None if value is None else tuple(value.shape)}")
        if env_route_enabled(cfg):
            k = int(cfg["LOSS"].get("env_route_k", 3))
            expected_heads = (batch_size, k, output_len, num_nodes, output_dim)
            expected_pred = (batch_size, output_len, num_nodes, output_dim)
            expected_q = (batch_size, k)
            for key in ["y_route_heads"]:
                value = output.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_heads:
                    raise AssertionError(f"{key} expected {expected_heads}, got {None if value is None else tuple(value.shape)}")
            for key in ["env_route_logits", "env_route_q"]:
                value = output.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_q:
                    raise AssertionError(f"{key} expected {expected_q}, got {None if value is None else tuple(value.shape)}")
            for key in ["y_route_soft", "y_route", "y_global", "y_route_final"]:
                value = output.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_pred:
                    raise AssertionError(f"{key} expected {expected_pred}, got {None if value is None else tuple(value.shape)}")
            for key in ["env_route_entropy", "env_route_alpha", "route_confidence"]:
                value = output.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,):
                    raise AssertionError(f"{key} expected {(batch_size,)}, got {None if value is None else tuple(value.shape)}")
            alpha = output["env_route_alpha"].detach()
            if bool((alpha < 0).any()) or bool((alpha > 1).any()):
                raise AssertionError("env_route_alpha must stay in [0,1]")
        aligned = align_target(targets, output["prediction"])
        print(f"aligned_targets: {tuple(aligned.shape)}")
        return

    expected = {
        "prediction": (batch_size, output_len, num_nodes, output_dim),
        "y_inv": (batch_size, output_len, num_nodes, output_dim),
        "y_potential": (batch_size, output_len, num_nodes, output_dim),
        "r_env": (batch_size, output_len, num_nodes, output_dim),
        "rho": (batch_size, output_len, num_nodes, 1),
        "z_inv": (batch_size, num_nodes, representation_dim),
        "z_raw": (batch_size, num_nodes, representation_dim),
        "env_mu": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env_logvar": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env_hist": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env_raw": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "y_inv_raw": (batch_size, output_len, num_nodes, output_dim),
    }
    for key, expected_shape in expected.items():
        value = output[key]
        print(f"{key}: {tuple(value.shape)}")
        if tuple(value.shape) != expected_shape:
            raise AssertionError(f"{key} expected {expected_shape}, got {tuple(value.shape)}")
        assert_finite(value, key)
    for key in [
        "z_seq",
        "prediction_swap",
        "rho_swap",
        "env_perm",
        "pseudo_env_head_pred",
        "y_route_heads",
        "env_route_logits",
        "env_route_q",
        "env_route_entropy",
        "env_route_alpha",
        "y_route_soft",
        "y_route_hard",
        "y_route",
        "y_route_selected",
        "y_global",
        "y_route_final",
        "route_confidence",
        "prediction_fused",
    ]:
        value = output.get(key)
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {tuple(value.shape)}")
            assert_finite(value, key)
    if pseudo_env_enabled(cfg):
        expected_pseudo = (batch_size, int(cfg["LOSS"].get("pseudo_env_k", 3)), output_len, num_nodes, output_dim)
        value = output.get("pseudo_env_head_pred")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_pseudo:
            raise AssertionError(f"pseudo_env_head_pred expected {expected_pseudo}, got {None if value is None else tuple(value.shape)}")
    if env_route_enabled(cfg):
        k = int(cfg["LOSS"].get("env_route_k", 3))
        expected_heads = (batch_size, k, output_len, num_nodes, output_dim)
        expected_pred = (batch_size, output_len, num_nodes, output_dim)
        expected_q = (batch_size, k)
        for key in ["y_route_heads"]:
            value = output.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_heads:
                raise AssertionError(f"{key} expected {expected_heads}, got {None if value is None else tuple(value.shape)}")
        for key in ["env_route_logits", "env_route_q"]:
            value = output.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_q:
                raise AssertionError(f"{key} expected {expected_q}, got {None if value is None else tuple(value.shape)}")
        for key in ["y_route_soft", "y_route", "y_global", "y_route_final"]:
            value = output.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_pred:
                raise AssertionError(f"{key} expected {expected_pred}, got {None if value is None else tuple(value.shape)}")
        for key in ["env_route_entropy", "env_route_alpha", "route_confidence"]:
            value = output.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,):
                raise AssertionError(f"{key} expected {(batch_size,)}, got {None if value is None else tuple(value.shape)}")
        alpha = output["env_route_alpha"].detach()
        if bool((alpha < 0).any()) or bool((alpha > 1).any()):
            raise AssertionError("env_route_alpha must stay in [0,1]")
    for key in ["env_fut", "persist_q", "persist_k", "persist_score"]:
        value = output.get(key)
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {tuple(value.shape)}")
            assert_finite(value, key)
    aligned = align_target(targets, output["prediction"])
    print(f"aligned_targets: {tuple(aligned.shape)}")


def _finite_stats(tensor: torch.Tensor) -> Dict[str, float]:
    values = tensor.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
    }


def _format_stats(name: str, tensor: torch.Tensor) -> str:
    stats = _finite_stats(tensor)
    return (
        f"metrics_debug {name} "
        f"min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
    )


def print_metric_debug_diagnostics(
    cfg: Dict,
    raw_batch: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    data_scaler: ZScoreDataScaler,
) -> None:
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    null_val = cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val"))
    metrics_cfg = cfg.get("METRICS", {})
    threshold = float(metrics_cfg.get("mape_threshold", 1.0))

    pred_norm = output["prediction"]
    pred_raw = data_scaler.inverse_transform(pred_norm)
    target_raw = align_target(raw_batch[target_key], pred_raw)
    target_norm = align_target(batch[target_key], pred_norm)
    mask = batch.get("targets_mask")
    mask_valid = make_valid_mask(target_raw, null_val, mask)
    mape_valid = make_mape_valid_mask(pred_raw, target_raw, null_val, mask, threshold=threshold)
    total = int(target_raw.numel())
    finite_target = torch.isfinite(target_raw)
    finite_total = int(finite_target.sum().detach().cpu())

    print(_format_stats("target_raw", target_raw))
    print(_format_stats("pred_raw", pred_raw))
    print(_format_stats("target_norm", target_norm))
    print(_format_stats("pred_norm", pred_norm))
    for small_threshold in (0.0, 1.0, 5.0):
        small = finite_target & (target_raw.abs() <= small_threshold)
        ratio = float(small.float().mean().detach().cpu()) if total else float("nan")
        print(f"metrics_debug target_raw_abs_le_{small_threshold:g}_ratio={ratio:.6f}")
    print(
        "metrics_debug "
        f"mape_valid_count={int(mape_valid.sum().detach().cpu())}/{total} "
        f"mape_threshold={threshold:g} finite_target_count={finite_total}/{total}"
    )
    print(
        "metrics_debug "
        f"mask_valid_count={int(mask_valid.sum().detach().cpu())}/{total} "
        f"mask_shape={tuple(mask.shape) if isinstance(mask, torch.Tensor) else None}"
    )

    if bool(cfg["MODEL"].get("required_timestamp", False)) or str(cfg["MODEL"].get("method_variant", "")).lower() == "fpem":
        seq_key = cfg["DATASET"].get("input_timestamp_key", "inputs_timestamps")
        cur_key = cfg["DATASET"].get("current_timestamp_key", "current_timestamps")
        missing_time_keys = [
            key for key in [
                seq_key,
                cfg["DATASET"].get("target_timestamp_key", "targets_timestamps"),
                cur_key,
            ]
            if not isinstance(batch.get(key), torch.Tensor)
        ]
        if cur_key in missing_time_keys and isinstance(batch.get(seq_key), torch.Tensor):
            missing_time_keys.remove(cur_key)
        if missing_time_keys:
            print(f"WARNING: FPEM requires timestamps but batch is missing {missing_time_keys}")


def debug_batch(cfg: Dict) -> None:
    cfg = finalize_config(cfg)
    train_cfg = cfg["TRAIN"]
    configure_torch_runtime(train_cfg)
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg)
    data_scaler = build_data_scaler(cfg, device)
    loader = build_loader(cfg, "train", shuffle=True)
    try:
        model, loss_fn = build_model_and_loss(cfg, device)
    except OfficialBaselineSkip as exc:
        print(str(exc))
        print(f"reference_status: {exc.reference_status}")
        print(f"unsupported_reason: {exc.reason}")
        return
    grad_consensus = TimeChannelSoftGradientConsensus(cfg["LOSS"].get("grad_consensus", {}))
    grad_surgery = InvariantGradientSurgery(cfg["LOSS"].get("grad_surgery", {}))
    grad_surgery_params = grad_surgery.select_params(model)
    loss_fn.set_epoch(1)
    model.train()

    raw_batch = to_device_batch(next(iter(loader)), device)
    batch = preprocess_batch(raw_batch, cfg, data_scaler)
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    scaler_summary = data_scaler.summary()
    backbone_name = cfg["MODEL"].get("backbone_name", "stid_mlp")
    ablations = cfg.get("RUN", {}).get("ablations", [])
    print(f"ablations: {','.join(ablations) if ablations else 'none'}")
    print(f"method_variant: {cfg['MODEL'].get('method_variant', 'nue')}")
    print(f"backbone_name: {backbone_name}")
    print(f"backbone_description: {BACKBONE_DESCRIPTIONS.get(str(backbone_name).lower(), 'custom invariant backbone')}")
    backbone_features = describe_backbone_features(model)
    print(f"backbone_uses_node_embedding: {backbone_features['node_embedding']}")
    print(f"backbone_uses_time_of_day_embedding: {backbone_features['time_of_day_embedding']}")
    print(f"backbone_uses_day_of_week_embedding: {backbone_features['day_of_week_embedding']}")
    print(f"backbone_uses_time_of_day_channel: {backbone_features['time_of_day_channel']}")
    print(f"backbone_uses_day_of_week_channel: {backbone_features['day_of_week_channel']}")
    if cfg["MODEL"].get("required_timestamp", False) and not (
        backbone_features["time_of_day_embedding"]
        or backbone_features["day_of_week_embedding"]
        or backbone_features["time_of_day_channel"]
        or backbone_features["day_of_week_channel"]
    ):
        print(
            "WARNING: current backbone does not consume TOD/DOW identity embeddings directly; "
            "FPEM still uses timestamp embeddings in z/env/mask/future branches."
        )
    print(f"teacher_forcing_enabled: {cfg['TRAIN'].get('teacher_forcing_enabled', False)}")
    print(f"tf_decay_steps: {cfg['TRAIN'].get('tf_decay_steps', '')}")
    if str(backbone_name).lower() == "agcrn":
        print(
            "WARNING: AGCRN adapter uses a direct multi-horizon head; official decoder scheduled sampling "
            "is not implemented in this local wrapper."
        )
    print(
        "loss_schedule: "
        f"warmup_epochs={cfg['LOSS'].get('warmup_epochs', 0)} "
        f"aux_ramp_epochs={cfg['LOSS'].get('aux_ramp_epochs', 0)}"
    )
    print(
        "horizon_curriculum: "
        f"enabled={cfg['TRAIN'].get('curriculum_enabled', False)} "
        f"start={cfg['TRAIN'].get('curriculum_start_horizon', '')} "
        f"full_epoch={cfg['TRAIN'].get('curriculum_full_horizon_epoch', '')}"
    )
    print(f"gate_label_mode: {cfg['LOSS'].get('gate_label_mode', 'potential_gain')}")
    print(f"swap_mode: {cfg.get('SWAP', {}).get('mode', 'batch_node_random')}")
    print(f"pair_mining: {cfg.get('SWAP', {}).get('pair_mining', False)}")
    print(f"env_consistency_enabled: {cfg['LOSS'].get('use_env_consistency', False)}")
    print(f"separation_mode: {cfg['MODEL'].get('separation', {}).get('mode', 'none')}")
    print(f"use_separated_z_for_y_inv: {cfg['MODEL'].get('use_separated_z_for_y_inv', True)}")
    print(f"persistence_enabled: {cfg['MODEL'].get('persistence', {}).get('enabled', False)}")
    print(f"persistence_affects_gate: {cfg['LOSS'].get('persistence_affects_gate', False)}")
    print(
        "scaler: "
        f"enabled={scaler_summary['enabled']} "
        f"mean={scaler_summary['mean']:.6f} std={scaler_summary['std']:.6f}"
    )
    print(f"mask_value_mode: {cfg['LOSS'].get('mask_value_mode', 'null_val')}")
    print(
        "grad_consensus: "
        f"enabled={grad_consensus.enabled} "
        f"target={grad_consensus.target} "
        f"mode={grad_consensus.mode} "
        f"rho_max={grad_consensus.rho_max} "
        f"warmup_epochs={grad_consensus.warmup_epochs} "
        f"use_ema={grad_consensus.use_ema}"
    )
    print(
        "grad_surgery: "
        f"enabled={grad_surgery.enabled} "
        f"method={grad_surgery.method} "
        f"apply_to={grad_surgery.apply_to} "
        f"num_params={len(grad_surgery_params)}"
    )
    z_inv_ib_cfg = cfg["LOSS"].get("z_inv_bottleneck", {})
    if bool(z_inv_ib_cfg.get("enabled", False)):
        print(
            "z_inv_bottleneck: "
            f"enabled=True "
            f"type={z_inv_ib_cfg.get('type', 'vib')} "
            f"beta={float(z_inv_ib_cfg.get('beta', 1.0e-4))} "
            f"predict_from_sampled_z={bool(z_inv_ib_cfg.get('predict_from_sampled_z', True))}"
        )
    if pseudo_env_enabled(cfg):
        print(
            "pseudo_env_heads: "
            f"enabled=True "
            f"k={cfg['LOSS'].get('pseudo_env_k', 3)} "
            f"tau={cfg['LOSS'].get('pseudo_env_tau', 1.0)} "
            f"assignment_mode={cfg['LOSS'].get('pseudo_env_assignment_mode', 'cached_soft')} "
            f"global_cache={cfg['LOSS'].get('pseudo_env_use_global_cache', True)} "
            f"temporal_smoothing={cfg['LOSS'].get('pseudo_env_use_temporal_smoothing', True)} "
            f"level={cfg['LOSS'].get('pseudo_env_level', 'window')}"
        )
    if env_route_enabled(cfg):
        print(
            "env_routed_inv_heads: "
            f"enabled=True "
            f"k={cfg['LOSS'].get('env_route_k', 3)} "
            f"tau={cfg['LOSS'].get('env_route_tau', 1.0)} "
            f"oracle_tau={cfg['LOSS'].get('env_route_oracle_tau', 0.3)} "
            f"mode={cfg['LOSS'].get('env_route_mode', 'confidence_mix')} "
            f"replace_final={cfg['LOSS'].get('env_route_replace_final', False)} "
            f"lambda_route_soft={cfg['LOSS'].get('env_route_lambda_route_soft', 0.5)} "
            f"lambda_router_oracle={cfg['LOSS'].get('env_route_lambda_router_oracle', 0.5)} "
            f"detach_q_for_expert={cfg['LOSS'].get('env_route_detach_q_for_expert', True)} "
            f"use_oracle_weight_for_expert={cfg['LOSS'].get('env_route_use_oracle_weight_for_expert', True)} "
            f"alpha_detach={cfg['LOSS'].get('env_route_alpha_detach', False)}"
        )
    print(f"{input_key}_raw: {tuple(raw_batch[input_key].shape)}")
    print(f"{target_key}_raw: {tuple(raw_batch[target_key].shape)}")
    print(f"{input_key}_scaled: {tuple(batch[input_key].shape)}")
    print(f"{target_key}_scaled: {tuple(batch[target_key].shape)}")
    print(f"{input_key}: {tuple(batch[input_key].shape)}")
    print(f"{target_key}: {tuple(batch[target_key].shape)}")
    print(f"batch_keys: {sorted(batch.keys())}")
    for time_key in [
        cfg["DATASET"].get("input_timestamp_key", "inputs_timestamps"),
        cfg["DATASET"].get("target_timestamp_key", "targets_timestamps"),
        cfg["DATASET"].get("current_timestamp_key", "current_timestamps"),
    ]:
        value = batch.get(time_key)
        print(f"{time_key}: {tuple(value.shape) if isinstance(value, torch.Tensor) else None}")
    print(
        f"{input_key}_raw_mean={raw_batch[input_key].float().mean().item():.6f} "
        f"{input_key}_scaled_mean={batch[input_key].float().mean().item():.6f}"
    )
    print(
        f"{target_key}_raw_mean={raw_batch[target_key].float().mean().item():.6f} "
        f"{target_key}_scaled_mean={batch[target_key].float().mean().item():.6f}"
    )
    print(f"inputs_after_align: {tuple(batch[input_key].shape)}")
    print(f"targets_before_align: {tuple(batch[target_key].shape)}")

    output = model(
        batch[input_key],
        y_true=batch[target_key],
        **get_time_kwargs(batch, cfg, include_future=True),
    )
    maybe_attach_pseudo_env_assignment(output, batch, cfg, None, epoch=1, cache_updated=False)
    register_grad_consensus_hook(output, grad_consensus, epoch=1)
    check_output_shapes(output, batch[target_key], cfg)
    debug_horizon = curriculum_horizon(train_cfg, 1, cfg["DATASET"]["output_len"])
    loss_output, loss_targets, loss_mask, loss_raw_targets = slice_for_train_horizon(
        output,
        batch[target_key],
        batch.get("targets_mask"),
        raw_batch[target_key],
        debug_horizon,
    )
    loss, logs = loss_fn(
        loss_output,
        loss_targets,
        loss_mask,
        raw_y_true=loss_raw_targets,
        data_scaler=data_scaler,
    )
    if pseudo_env_enabled(cfg):
        head_pred = loss_output.get("pseudo_env_head_pred")
        if not isinstance(head_pred, torch.Tensor):
            raise AssertionError("pseudo_env_heads enabled but pseudo_env_head_pred is missing.")
        loss_cfg = cfg["LOSS"]
        targets_for_debug = (
            loss_raw_targets
            if str(loss_cfg.get("train_loss_scale", "normalized")).lower() == "original"
            else loss_targets
        )
        loss_head = compute_pseudo_env_loss_head_for_cache(
            loss_output,
            targets_for_debug,
            loss_mask,
            cfg,
            data_scaler,
        )
        expected_shape = (head_pred.shape[0], int(loss_cfg.get("pseudo_env_k", 3)))
        if tuple(loss_head.shape) != expected_shape:
            raise AssertionError(f"loss_head must be [B,K]={expected_shape}, got {tuple(loss_head.shape)}")
        q_env = torch.softmax(-loss_head / max(float(loss_cfg.get("pseudo_env_tau", 1.0)), 1e-6), dim=-1)
        q_row_sum_error = (q_env.sum(dim=-1) - 1.0).abs().max().item()
        if q_row_sum_error > 1e-5:
            raise AssertionError(f"pseudo_env q_env rows do not sum to 1 (max_error={q_row_sum_error:.6e})")
        print(
            "pseudo_env_debug: "
            f"head_pred_shape={tuple(head_pred.shape)} "
            f"loss_head_shape={tuple(loss_head.shape)} "
            f"q_row_sum_max_error={q_row_sum_error:.6e}"
        )
    if env_route_enabled(cfg):
        y_route_heads = loss_output.get("y_route_heads")
        q = loss_output.get("env_route_q")
        if not isinstance(y_route_heads, torch.Tensor) or not isinstance(q, torch.Tensor):
            raise AssertionError("env_routed_inv_heads enabled but y_route_heads/env_route_q is missing.")
        loss_cfg = cfg["LOSS"]
        targets_for_debug = (
            loss_raw_targets
            if str(loss_cfg.get("train_loss_scale", "normalized")).lower() == "original"
            else loss_targets
        )
        y_route_heads_for_debug = (
            data_scaler.inverse_transform(y_route_heads)
            if str(loss_cfg.get("train_loss_scale", "normalized")).lower() == "original"
            else y_route_heads
        )
        loss_head = loss_fn._env_route_head_losses(y_route_heads_for_debug, targets_for_debug, loss_mask)
        expected_shape = (y_route_heads.shape[0], int(loss_cfg.get("env_route_k", 3)))
        if tuple(loss_head.shape) != expected_shape:
            raise AssertionError(f"env route loss_head must be [B,K]={expected_shape}, got {tuple(loss_head.shape)}")
        q_row_sum_error = (q.sum(dim=-1) - 1.0).abs().max().item()
        if q_row_sum_error > 1e-5:
            raise AssertionError(f"env_route_q rows do not sum to 1 (max_error={q_row_sum_error:.6e})")
        alpha = loss_output.get("env_route_alpha")
        if not isinstance(alpha, torch.Tensor):
            raise AssertionError("env_routed_inv_heads enabled but env_route_alpha is missing.")
        alpha_min = float(alpha.detach().min().cpu())
        alpha_max = float(alpha.detach().max().cpu())
        if alpha_min < -1e-6 or alpha_max > 1.0 + 1e-6:
            raise AssertionError(f"env_route_alpha must stay in [0,1], got min={alpha_min:.6e} max={alpha_max:.6e}")
        if not bool(loss_cfg.get("env_route_replace_final", False)):
            fused = output.get("prediction_fused")
            if not isinstance(fused, torch.Tensor):
                raise AssertionError("env_route_replace_final=False expected prediction_fused for debug comparison.")
            max_delta = (output["prediction"].detach() - fused.detach()).abs().max().item()
            if max_delta != 0.0:
                raise AssertionError(f"replace_final=False changed main prediction (max_delta={max_delta:.6e})")
        print(
            "env_route_debug: "
            f"heads_shape={tuple(y_route_heads.shape)} "
            f"loss_head_shape={tuple(loss_head.shape)} "
            f"q_shape={tuple(q.shape)} "
            f"q_row_sum_max_error={q_row_sum_error:.6e} "
            f"alpha_mean={float(alpha.detach().mean().cpu()):.6f} "
            f"alpha_min={alpha_min:.6f} "
            f"alpha_max={alpha_max:.6f} "
            f"replace_final={bool(loss_cfg.get('env_route_replace_final', False))}"
        )
    loss_terms = logs.pop("__loss_terms__", None)
    surgery_prepared = grad_surgery.prepare(
        loss_terms,
        grad_surgery_params,
        scale=1.0,
        like=output["prediction"],
    )
    perturb_enabled = bool(cfg["MODEL"].get("perturb_enabled", False))
    print(f"perturb_enabled={perturb_enabled}")
    if perturb_enabled:
        perturb_types = [
            name
            for name, enabled in [
                ("value_jitter", cfg["MODEL"].get("perturb_value_jitter", True)),
                ("value_scale", cfg["MODEL"].get("perturb_value_scale", True)),
                ("time_node_mask", cfg["MODEL"].get("perturb_time_node_mask", True)),
                ("temporal_block", cfg["MODEL"].get("perturb_temporal_block", True)),
                ("edge_dropout", cfg["MODEL"].get("perturb_edge_dropout", False)),
            ]
            if enabled
        ]
        perturb_info = output.get("perturb_info") or {}

        def _debug_float(value) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        print(f"perturb_types_enabled={','.join(perturb_types) if perturb_types else 'none'}")
        print(f"perturb_applied={perturb_info.get('applied', False)}")
        x_aug_stats = perturb_info.get("x_aug_stats") if isinstance(perturb_info, dict) else None
        if isinstance(x_aug_stats, dict):
            print(
                "x_aug "
                f"min={_debug_float(x_aug_stats['min']):.6f} "
                f"max={_debug_float(x_aug_stats['max']):.6f} "
                f"mean={_debug_float(x_aug_stats['mean']):.6f} "
                f"std={_debug_float(x_aug_stats['std']):.6f}"
            )
        print(
            "perturb_consistency "
            f"z_cons_loss={_debug_float(logs.get('z_cons_loss', output['prediction'].new_zeros(()))):.6f} "
            f"y_cons_loss={_debug_float(logs.get('y_cons_loss', output['prediction'].new_zeros(()))):.6f}"
        )
    logs["curriculum_horizon"] = output["prediction"].new_tensor(float(debug_horizon))
    loss.backward()
    non_surgery_grad_snapshots = []
    surgery_grad_snapshots = []
    if grad_surgery.enabled:
        surgery_param_ids = {id(param) for param in grad_surgery_params}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if id(param) in surgery_param_ids:
                surgery_grad_snapshots.append((name, param, param.grad.detach().clone()))
            else:
                non_surgery_grad_snapshots.append((name, param, param.grad.detach().clone()))
    grad_surgery.apply(grad_surgery_params, surgery_prepared)
    if grad_surgery.enabled:
        max_non_surgery_delta = 0.0
        for _name, param, grad_before in non_surgery_grad_snapshots:
            if param.grad is None:
                delta = grad_before.float().abs().max()
            else:
                delta = (param.grad.detach().float() - grad_before.float()).abs().max()
            max_non_surgery_delta = max(max_non_surgery_delta, float(delta.cpu()))
        surgery_delta_sq = 0.0
        for _name, param, grad_before in surgery_grad_snapshots:
            if param.grad is None:
                delta = grad_before.float()
            else:
                delta = param.grad.detach().float() - grad_before.float()
            surgery_delta_sq += float(delta.pow(2).sum().cpu())
        surgery_delta_norm = math.sqrt(surgery_delta_sq)
        print(
            "grad_surgery_debug: "
            f"non_inv_grad_max_delta={max_non_surgery_delta:.6e} "
            f"inv_grad_delta_norm={surgery_delta_norm:.6e}"
        )
        if max_non_surgery_delta > 0.0:
            raise AssertionError(
                "grad_surgery changed gradients outside invariant encoder parameters "
                f"(max_delta={max_non_surgery_delta:.6e})"
            )
    logs.update(grad_consensus.log_tensors(output["prediction"]))
    logs.update(grad_surgery.log_tensors(output["prediction"]))
    assert_finite(loss, "total_loss")
    assert_finite(output["rho"], "rho")
    pred_original = data_scaler.inverse_transform(output["prediction"])
    original_mae = masked_mae_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
        batch.get("targets_mask"),
    )
    original_mse = masked_mse_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
        batch.get("targets_mask"),
    )
    original_rmse = masked_rmse_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
        batch.get("targets_mask"),
    )
    metrics_cfg = cfg.get("METRICS", {})
    original_mape = masked_mape_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
        batch.get("targets_mask"),
        eps=float(metrics_cfg.get("mape_eps", 1e-5)),
        threshold=float(metrics_cfg.get("mape_threshold", 1.0)),
        as_percent=bool(metrics_cfg.get("mape_as_percent", True)),
    )
    print(f"debug_original_scale_mae={float(original_mae.detach().cpu()):.6f}")
    print(f"debug_original_scale_mse={float(original_mse.detach().cpu()):.6f}")
    print(f"debug_original_scale_rmse={float(original_rmse.detach().cpu()):.6f}")
    print(f"debug_original_scale_mape={float(original_mape.detach().cpu()):.6f}")
    print_metric_debug_diagnostics(cfg, raw_batch, batch, output, data_scaler)
    ib_enabled = z_inv_bottleneck_enabled(cfg)
    ib_log_keys = [key for key in logs if key == "loss_z_inv_ib" or key.startswith("z_inv_ib/")]
    if ib_enabled:
        if "loss_z_inv_ib" not in logs:
            raise AssertionError("z_inv_bottleneck is enabled but loss_z_inv_ib was not logged.")
        ib_type = str(cfg["LOSS"]["z_inv_bottleneck"].get("type", "vib")).lower()
        ib_loss_value = float(logs["loss_z_inv_ib"].detach().cpu())
        print(f"z_inv_bottleneck_debug: enabled=True type={ib_type} loss_z_inv_ib={ib_loss_value:.6e}")
        if ib_type != "gaussian_noise" and ib_loss_value <= 0.0:
            raise AssertionError("z_inv_bottleneck expected a positive loss_z_inv_ib for vib/l2_norm debug mode.")
    elif ib_log_keys:
        raise AssertionError(f"z_inv_bottleneck is disabled but logged keys appeared: {ib_log_keys}")
    debug_log_keys = log_keys_for_config(cfg)
    if not perturb_enabled:
        debug_log_keys = [
            key for key in debug_log_keys
            if key not in {"z_cons_loss", "y_cons_loss", "effective_lambda_z_cons", "effective_lambda_y_cons"}
        ]
    print(format_logs(logs, debug_log_keys))
    print("debug_batch ok: forward/loss/backward finished without NaN or shape errors")


@torch.no_grad()
def compute_metric_dict(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val,
    existing_mask: torch.Tensor | None = None,
    metrics_cfg: Dict | None = None,
) -> Dict[str, float]:
    metrics_cfg = metrics_cfg or {}
    mape_eps = float(metrics_cfg.get("mape_eps", 1e-5))
    mape_threshold = float(metrics_cfg.get("mape_threshold", 1.0))
    mape_as_percent = bool(metrics_cfg.get("mape_as_percent", True))
    metrics = {
        "mae": float(masked_mae_value(prediction, targets, null_val, existing_mask).detach().cpu()),
        "mse": float(masked_mse_value(prediction, targets, null_val, existing_mask).detach().cpu()),
        "mape": float(
            masked_mape_value(
                prediction,
                targets,
                null_val,
                existing_mask,
                eps=mape_eps,
                threshold=mape_threshold,
                as_percent=mape_as_percent,
            ).detach().cpu()
        ),
        "wape": float(
            masked_wape_value(
                prediction,
                targets,
                null_val,
                existing_mask,
                eps=mape_eps,
                as_percent=mape_as_percent,
            ).detach().cpu()
        ),
    }
    metrics["rmse"] = float(np.sqrt(max(metrics["mse"], 0.0)))
    horizon = prediction.shape[1]
    eval_horizon = min(12, horizon)
    avg_mask = existing_mask[:, :eval_horizon] if existing_mask is not None else None
    metrics["mae_avg12"] = float(
        masked_mae_value(prediction[:, :eval_horizon], targets[:, :eval_horizon], null_val, avg_mask).detach().cpu()
    )
    metrics["mse_avg12"] = float(
        masked_mse_value(prediction[:, :eval_horizon], targets[:, :eval_horizon], null_val, avg_mask).detach().cpu()
    )
    metrics["rmse_avg12"] = float(np.sqrt(max(metrics["mse_avg12"], 0.0)))
    metrics["mape_avg12"] = float(
        masked_mape_value(
            prediction[:, :eval_horizon],
            targets[:, :eval_horizon],
            null_val,
            avg_mask,
            eps=mape_eps,
            threshold=mape_threshold,
            as_percent=mape_as_percent,
        ).detach().cpu()
    )
    metrics["wape_avg12"] = float(
        masked_wape_value(
            prediction[:, :eval_horizon],
            targets[:, :eval_horizon],
            null_val,
            avg_mask,
            eps=mape_eps,
            as_percent=mape_as_percent,
        ).detach().cpu()
    )
    for step in HORIZON_EVAL_STEPS:
        if horizon >= step:
            pred_h = prediction[:, step - 1 : step]
            target_h = targets[:, step - 1 : step]
            mask_h = existing_mask[:, step - 1 : step] if existing_mask is not None else None
            mse_key = f"mse_h{step}"
            metrics[f"mae_h{step}"] = float(masked_mae_value(pred_h, target_h, null_val, mask_h).detach().cpu())
            metrics[mse_key] = float(masked_mse_value(pred_h, target_h, null_val, mask_h).detach().cpu())
            metrics[f"rmse_h{step}"] = float(np.sqrt(max(metrics[mse_key], 0.0)))
            metrics[f"mape_h{step}"] = float(
                masked_mape_value(
                    pred_h,
                    target_h,
                    null_val,
                    mask_h,
                    eps=mape_eps,
                    threshold=mape_threshold,
                    as_percent=mape_as_percent,
                ).detach().cpu()
            )
            metrics[f"wape_h{step}"] = float(
                masked_wape_value(
                    pred_h,
                    target_h,
                    null_val,
                    mask_h,
                    eps=mape_eps,
                    as_percent=mape_as_percent,
                ).detach().cpu()
            )
    return metrics


@torch.no_grad()
def evaluate(
    model: NUESTG,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict,
    max_batches,
    data_scaler: ZScoreDataScaler,
) -> Dict[str, float]:
    model.eval()
    values: Dict[str, list] = {}
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    null_val = cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val"))
    full_eval = max_batches is None or int(max_batches) < 0
    metric_aggregation = str(cfg.get("EVAL", {}).get("metric_aggregation", "batch_mean") or "batch_mean").lower()
    concat_eval = metric_aggregation in {"concat", "stexpert", "stexpert_concat"}
    pred_chunks = []
    target_chunks = []
    mask_chunks = []
    needs_future_time = bool(
        cfg["MODEL"].get("required_timestamp", False)
        or str(cfg["MODEL"].get("method_variant", "")).lower() == "fpem"
    )
    for step, batch in enumerate(loader):
        if not full_eval and step >= int(max_batches):
            break
        raw_batch = to_device_batch(batch, device)
        batch = preprocess_batch(raw_batch, cfg, data_scaler)
        output = model(
            batch[input_key],
            **get_time_kwargs(batch, cfg, include_future=needs_future_time),
        )
        prediction = data_scaler.inverse_transform(output["prediction"])
        targets = raw_batch[target_key]
        if concat_eval:
            pred_chunks.append(prediction.detach())
            target_chunks.append(targets.detach())
            if isinstance(batch.get("targets_mask"), torch.Tensor):
                mask_chunks.append(batch["targets_mask"].detach())
            continue
        batch_metrics = compute_metric_dict(
            prediction,
            targets,
            null_val,
            batch.get("targets_mask"),
            cfg.get("METRICS", {}),
        )
        for key, value in batch_metrics.items():
            values.setdefault(key, []).append(value)
    model.train()
    if concat_eval:
        if not pred_chunks:
            return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan"), "wape": float("nan")}
        prediction = torch.cat(pred_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        existing_mask = torch.cat(mask_chunks, dim=0) if len(mask_chunks) == len(pred_chunks) else None
        metric_null_val = null_val
        mode = str(cfg["LOSS"].get("mask_value_mode", cfg["DATASET"].get("mask_value_mode", "null_val")) or "null_val").lower()
        if mode in {"stexpert_min", "st_expert_min", "batch_min_if_lt_one"}:
            metric_null_val = resolve_target_mask_value(targets, cfg)
            existing_mask = None
        return compute_metric_dict(prediction, targets, metric_null_val, existing_mask, cfg.get("METRICS", {}))
    if not values.get("mae"):
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan"), "wape": float("nan")}
    result = {key: float(np.mean(item_values)) for key, item_values in values.items()}
    for suffix in ["", "_avg12", *[f"_h{step}" for step in HORIZON_EVAL_STEPS]]:
        mse_key = f"mse{suffix}"
        rmse_key = f"rmse{suffix}"
        if mse_key in result:
            result[rmse_key] = float(np.sqrt(max(result[mse_key], 0.0)))
    return result


@torch.no_grad()
def save_test_diagnostics(
    model: NUESTG,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict,
    max_batches,
    data_scaler: ZScoreDataScaler,
    ckpt_dir: Path,
) -> Dict[str, float]:
    if not bool(cfg.get("EVAL", {}).get("save_test_diagnostics", False)):
        return {}
    model.eval()
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    null_val = cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val"))
    metrics_cfg = cfg.get("METRICS", {})
    full_eval = max_batches is None or int(max_batches) < 0
    arrays: Dict[str, list] = {
        "prediction_raw": [],
        "target_raw": [],
        "mask": [],
        "env_mask": [],
        "env_plus": [],
        "env_minus": [],
    }
    stats: Dict[str, list] = {
        "mask_density": [],
        "env_plus_future_nll": [],
        "env_minus_future_nll": [],
        "swap_delta_mae": [],
    }
    for step, batch in enumerate(loader):
        if not full_eval and step >= int(max_batches):
            break
        raw_batch = to_device_batch(batch, device)
        batch = preprocess_batch(raw_batch, cfg, data_scaler)
        output = model(
            batch[input_key],
            y_true=batch[target_key],
            **get_time_kwargs(batch, cfg, include_future=True),
            compute_aux=True,
        )
        prediction_raw = data_scaler.inverse_transform(output["prediction"])
        target_raw = align_target(raw_batch[target_key], prediction_raw)
        target_mask = make_valid_mask(target_raw, null_val, batch.get("targets_mask"))
        arrays["prediction_raw"].append(prediction_raw.detach().cpu().numpy().astype(np.float32))
        arrays["target_raw"].append(target_raw.detach().cpu().numpy().astype(np.float32))
        arrays["mask"].append(target_mask.detach().cpu().numpy().astype(np.bool_))
        if isinstance(output.get("mask"), torch.Tensor):
            env_mask = output["mask"].detach()
            arrays["env_mask"].append(env_mask.cpu().numpy().astype(np.float32))
            stats["mask_density"].append(float(env_mask.mean().cpu()))
        for key in ["env_plus", "env_minus"]:
            if isinstance(output.get(key), torch.Tensor):
                arrays[key].append(output[key].detach().cpu().numpy().astype(np.float32))
        pred_swap = output.get("prediction_swap")
        if isinstance(pred_swap, torch.Tensor):
            swap_raw = data_scaler.inverse_transform(pred_swap)
            full_mae = masked_mae_value(prediction_raw, target_raw, null_val, target_mask)
            swap_mae = masked_mae_value(swap_raw, target_raw, null_val, target_mask)
            stats["swap_delta_mae"].append(float((swap_mae - full_mae).detach().cpu()))
        env_fut_tokens = output.get("env_fut_tokens")
        pred_fut_mu = output.get("pred_fut_mu")
        pred_fut_logvar = output.get("pred_fut_logvar")
        pred_fut_mu_minus = output.get("pred_fut_mu_minus")
        pred_fut_logvar_minus = output.get("pred_fut_logvar_minus")
        if isinstance(env_fut_tokens, torch.Tensor) and isinstance(pred_fut_mu, torch.Tensor) and isinstance(pred_fut_logvar, torch.Tensor):
            nll_plus = 0.5 * ((env_fut_tokens - pred_fut_mu).pow(2) * torch.exp(-pred_fut_logvar) + pred_fut_logvar)
            stats["env_plus_future_nll"].append(float(nll_plus.mean().detach().cpu()))
        if isinstance(env_fut_tokens, torch.Tensor) and isinstance(pred_fut_mu_minus, torch.Tensor) and isinstance(pred_fut_logvar_minus, torch.Tensor):
            nll_minus = 0.5 * (
                (env_fut_tokens - pred_fut_mu_minus).pow(2) * torch.exp(-pred_fut_logvar_minus) + pred_fut_logvar_minus
            )
            stats["env_minus_future_nll"].append(float(nll_minus.mean().detach().cpu()))

    if not arrays["prediction_raw"]:
        model.train()
        return {}

    save_payload = {}
    for key, chunks in arrays.items():
        if chunks:
            save_payload[key] = np.concatenate(chunks, axis=0)
    np.savez_compressed(ckpt_dir / "test_outputs.npz", **save_payload)

    prediction = torch.from_numpy(save_payload["prediction_raw"]).to(device)
    targets = torch.from_numpy(save_payload["target_raw"]).to(device)
    mask = torch.from_numpy(save_payload["mask"]).to(device)
    horizon_rows = []
    for horizon_idx in range(prediction.shape[1]):
        pred_h = prediction[:, horizon_idx : horizon_idx + 1]
        target_h = targets[:, horizon_idx : horizon_idx + 1]
        mask_h = mask[:, horizon_idx : horizon_idx + 1]
        row_metrics = compute_metric_dict(pred_h, target_h, null_val, mask_h, metrics_cfg)
        horizon_rows.append({
            "horizon": horizon_idx + 1,
            "mae": row_metrics.get("mae", float("nan")),
            "mse": row_metrics.get("mse", float("nan")),
            "rmse": row_metrics.get("rmse", float("nan")),
            "mape": row_metrics.get("mape", float("nan")),
            "wape": row_metrics.get("wape", float("nan")),
        })
    with (ckpt_dir / "test_metrics_by_horizon.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "mae", "mse", "rmse", "mape", "wape"])
        writer.writeheader()
        writer.writerows(horizon_rows)

    valid_targets = targets[mask]
    range_rows = []
    if valid_targets.numel() > 0:
        q_low = torch.quantile(valid_targets.float(), 1.0 / 3.0)
        q_high = torch.quantile(valid_targets.float(), 2.0 / 3.0)
        ranges = [
            ("low", mask & (targets <= q_low)),
            ("mid", mask & (targets > q_low) & (targets <= q_high)),
            ("high", mask & (targets > q_high)),
        ]
        for label, range_mask in ranges:
            row_metrics = compute_metric_dict(prediction, targets, null_val, range_mask, metrics_cfg)
            range_rows.append({
                "range": label,
                "count": int(range_mask.sum().detach().cpu()),
                "mae": row_metrics.get("mae", float("nan")),
                "mse": row_metrics.get("mse", float("nan")),
                "rmse": row_metrics.get("rmse", float("nan")),
                "mape": row_metrics.get("mape", float("nan")),
                "wape": row_metrics.get("wape", float("nan")),
            })
    with (ckpt_dir / "test_metrics_by_range.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["range", "count", "mae", "mse", "rmse", "mape", "wape"])
        writer.writeheader()
        writer.writerows(range_rows)

    summary = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in stats.items()
    }
    save_metrics_json(ckpt_dir / "test_env_diagnostics.json", summary)
    model.train()
    return summary


def build_metrics_payload(
    cfg: Dict,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    ckpt_path: Path,
    split: str = "val",
) -> Dict:
    run_cfg = cfg.get("RUN", {})
    model_cfg = cfg["MODEL"]
    payload = {
        "dataset": cfg["DATASET"]["name"],
        "setting": run_cfg.get("setting", "forecasting"),
        "method": run_cfg.get("method", model_cfg.get("name", "NUE-STG")),
        "display_name": run_cfg.get("display_name", run_cfg.get("method", model_cfg.get("name", "NUE-STG"))),
        "category": run_cfg.get("category", "plugin_ours"),
        "backbone": model_cfg.get("backbone_name", ""),
        "ablation": ",".join(run_cfg.get("ablations", [])) if run_cfg.get("ablations") else run_cfg.get("ablation", ""),
        "reference_status": model_cfg.get("reference_status", run_cfg.get("reference_status", "")),
        "is_official": run_cfg.get("is_official", model_cfg.get("is_official", "")),
        "is_adapter": run_cfg.get("is_adapter", model_cfg.get("is_adapter", "")),
        "main_table_safe": run_cfg.get("main_table_safe", model_cfg.get("main_table_safe", "")),
        "unsupported_reason": run_cfg.get("unsupported_reason", model_cfg.get("unsupported_reason", "")),
        "seed": cfg["TRAIN"].get("seed"),
        "epoch": epoch,
        "global_step": global_step,
        "split": split,
        "mae": metrics.get("mae", float("nan")),
        "mse": metrics.get("mse", float("nan")),
        "rmse": metrics.get("rmse", float("nan")),
        "mape": metrics.get("mape", float("nan")),
        "wape": metrics.get("wape", float("nan")),
        "config_path": run_cfg.get("config_path", ""),
        "ckpt_path": str(ckpt_path),
        "status": run_cfg.get("status", "runnable"),
        "notes": run_cfg.get("notes", ""),
    }
    for key, value in metrics.items():
        if key not in payload:
            payload[key] = value
    return payload


def save_metrics_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def atomic_torch_save(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def torch_load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dict, got {type(checkpoint)!r}: {path}")
    return checkpoint


def resolve_resume_path(train_cfg: Dict, ckpt_dir: Path) -> Path | None:
    resume_from = train_cfg.get("resume_from", "")
    auto_resume = bool(train_cfg.get("auto_resume", False))
    if resume_from in (None, "", False):
        if not auto_resume:
            return None
        candidates = [ckpt_dir / "last.pt"]
        if bool(train_cfg.get("resume_fallback_best", True)):
            candidates.append(ckpt_dir / "best.pt")
    elif str(resume_from).lower() == "auto":
        candidates = [ckpt_dir / "last.pt"]
        if bool(train_cfg.get("resume_fallback_best", True)):
            candidates.append(ckpt_dir / "best.pt")
    else:
        path = Path(str(resume_from))
        candidates = [path if path.is_absolute() else (Path.cwd() / path)]

    for path in candidates:
        if path.exists():
            return path
    if resume_from not in (None, "", False) and str(resume_from).lower() != "auto":
        raise FileNotFoundError(f"TRAIN.resume_from checkpoint not found: {candidates[0]}")
    return None


def read_best_score_from_metrics(ckpt_dir: Path, train_cfg: Dict) -> float:
    select_split = str(train_cfg.get("best_select_split", "test") or "test").lower()
    metric_name = str(train_cfg.get("best_select_metric", "mae") or "mae").lower()
    metrics_path = ckpt_dir / ("best_test_metrics.json" if select_split == "test" else "best_metrics.json")
    if not metrics_path.exists():
        return float("inf")
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        value = float(payload.get(metric_name, float("inf")))
        return value if np.isfinite(value) else float("inf")
    except Exception:
        return float("inf")


def make_checkpoint_payload(
    cfg: Dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    data_scaler: ZScoreDataScaler,
    amp_scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val: float,
    patience: int,
    pseudo_env_cache: PseudoEnvCache | None = None,
    extra: Dict | None = None,
) -> Dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "config": cfg,
        "scaler": data_scaler.state_dict(),
        "amp_scaler": amp_scaler.state_dict() if amp_scaler is not None else None,
        "epoch": epoch,
        "epoch_complete": True,
        "global_step": global_step,
        "best_score": best_val,
        "best_val": best_val,
        "best_select_split": cfg["TRAIN"].get("best_select_split", "test"),
        "best_select_metric": cfg["TRAIN"].get("best_select_metric", "mae"),
        "patience": patience,
    }
    if pseudo_env_cache is not None:
        payload["pseudo_env_cache"] = pseudo_env_cache.state_dict()
    if extra:
        payload.update(extra)
    return payload


def save_run_complete_marker(
    ckpt_dir: Path,
    cfg: Dict,
    epoch: int,
    global_step: int,
    best_val: float,
    stopped_early: bool,
) -> None:
    if not cfg["TRAIN"].get("save_completion_marker", True):
        return
    save_metrics_json(
        ckpt_dir / "run_complete.json",
        {
            "status": "complete",
            "epoch": epoch,
            "target_epochs": cfg["TRAIN"].get("epochs"),
            "global_step": global_step,
            "best_score": best_val,
            "best_val": best_val,
            "best_select_split": cfg["TRAIN"].get("best_select_split", "test"),
            "best_select_metric": cfg["TRAIN"].get("best_select_metric", "mae"),
            "stopped_early": stopped_early,
            "best_metrics_path": str(ckpt_dir / "best_metrics.json"),
            "best_test_metrics_path": str(ckpt_dir / "best_test_metrics.json"),
            "last_checkpoint_path": str(ckpt_dir / "last.pt"),
        },
    )


def restore_training_state(
    checkpoint: Dict,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    amp_scaler: torch.cuda.amp.GradScaler,
    train_cfg: Dict,
) -> tuple[int, int, float, int]:
    strict = bool(train_cfg.get("resume_strict", True))
    try:
        model.load_state_dict(checkpoint["model"], strict=strict)
    except RuntimeError as exc:
        if not strict or not bool(train_cfg.get("resume_allow_missing_pseudo_env", True)):
            raise
        warnings.warn(
            f"Checkpoint {path} is missing or has extra optional module parameters; "
            "loading model weights with strict=False.",
            RuntimeWarning,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    if bool(train_cfg.get("resume_load_optimizer", True)) and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if (
        bool(train_cfg.get("resume_load_scheduler", True))
        and lr_scheduler is not None
        and checkpoint.get("scheduler") is not None
    ):
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
    if (
        bool(train_cfg.get("resume_load_amp_scaler", True))
        and amp_scaler is not None
        and checkpoint.get("amp_scaler") is not None
    ):
        amp_scaler.load_state_dict(checkpoint["amp_scaler"])

    resume_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = resume_epoch + 1 if bool(checkpoint.get("epoch_complete", True)) else resume_epoch
    global_step = int(checkpoint.get("global_step", 0))
    best_val = float(checkpoint.get("best_score", checkpoint.get("best_val", checkpoint.get("val_mae", float("inf")))))
    patience = int(checkpoint.get("patience", 0))
    print(
        f"[resume] loaded {path} epoch={resume_epoch} start_epoch={start_epoch} "
        f"global_step={global_step} best_val={best_val} patience={patience}"
    )
    return start_epoch, global_step, best_val, patience


def make_val_metric_row(metrics: Dict[str, float], lr: float | None = None) -> Dict[str, object]:
    row = {
        "val_mae": metrics.get("mae", ""),
        "val_mse": metrics.get("mse", ""),
        "val_rmse": metrics.get("rmse", ""),
        "val_mape": metrics.get("mape", ""),
        "val_wape": metrics.get("wape", ""),
        "val_mae_avg12": metrics.get("mae_avg12", metrics.get("mae", "")),
        "val_rmse_avg12": metrics.get("rmse_avg12", metrics.get("rmse", "")),
        "val_mape_avg12": metrics.get("mape_avg12", metrics.get("mape", "")),
        "val_wape_avg12": metrics.get("wape_avg12", metrics.get("wape", "")),
        "lr": "" if lr is None else lr,
    }
    for step in HORIZON_EVAL_STEPS:
        row[f"val_mae_h{step}"] = metrics.get(f"mae_h{step}", "")
        row[f"val_rmse_h{step}"] = metrics.get(f"rmse_h{step}", "")
        row[f"val_mape_h{step}"] = metrics.get(f"mape_h{step}", "")
        row[f"val_wape_h{step}"] = metrics.get(f"wape_h{step}", "")
    return row


def append_train_log(cfg: Dict, row: Dict) -> None:
    if not cfg["LOGGING"].get("save_csv_log", True):
        return
    csv_path = Path(cfg["TRAIN"]["ckpt_dir"]) / cfg["LOGGING"].get("csv_log_path", "train_log.csv")
    append_csv_log(csv_path, row, csv_fields_for_config(cfg))


def train_local(cfg: Dict) -> None:
    cfg = finalize_config(cfg)
    train_cfg = cfg["TRAIN"]
    configure_torch_runtime(train_cfg)
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg)
    data_scaler = build_data_scaler(cfg, device)
    cfg.setdefault("SCALER", {})
    cfg["SCALER"]["stats"] = data_scaler.state_dict()
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    test_loader = build_loader(cfg, "test", shuffle=False)
    try:
        model, loss_fn = build_model_and_loss(cfg, device)
    except OfficialBaselineSkip as exc:
        print(str(exc))
        print(f"reference_status: {exc.reference_status}")
        print(f"unsupported_reason: {exc.reason}")
        return
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    grad_consensus = TimeChannelSoftGradientConsensus(cfg["LOSS"].get("grad_consensus", {}))
    grad_surgery = InvariantGradientSurgery(cfg["LOSS"].get("grad_surgery", {}))
    grad_surgery_params = grad_surgery.select_params(model)
    pseudo_env_cache = None
    if (
        pseudo_env_enabled(cfg)
        and bool(cfg["LOSS"].get("pseudo_env_use_global_cache", True))
        and str(cfg["LOSS"].get("pseudo_env_level", "window")).lower() == "window"
    ):
        pseudo_env_cache = PseudoEnvCache(len(train_loader.dataset), int(cfg["LOSS"].get("pseudo_env_k", 3)))
        print(
            "[pseudo-env] global cache enabled "
            f"samples={pseudo_env_cache.num_samples} heads={pseudo_env_cache.num_heads} "
            f"assignment_mode={cfg['LOSS'].get('pseudo_env_assignment_mode', 'cached_soft')}"
        )
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", False) and device.type == "cuda")
    autocast_ctx = (
        torch.cuda.amp.autocast
        if train_cfg.get("amp", False) and device.type == "cuda"
        else nullcontext
    )

    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg["LOGGING"].get("save_config", True):
        save_resolved_config(cfg, ckpt_dir / "resolved_config.json")

    best_select_split = str(train_cfg.get("best_select_split", "test") or "test").lower()
    best_select_metric = str(train_cfg.get("best_select_metric", "mae") or "mae").lower()
    if best_select_split not in {"val", "test"}:
        raise ValueError("TRAIN.best_select_split must be 'val' or 'test'")
    if best_select_metric != "mae":
        raise ValueError("TRAIN.best_select_metric currently supports only 'mae'")
    print(f"[best-select] split={best_select_split} metric={best_select_metric}")
    best_val = read_best_score_from_metrics(ckpt_dir, train_cfg)
    patience = 0
    global_step = 0
    start_epoch = 1
    resume_path = resolve_resume_path(train_cfg, ckpt_dir)
    if resume_path is not None:
        try:
            checkpoint = torch_load_checkpoint(resume_path, device)
            start_epoch, global_step, best_val, patience = restore_training_state(
                checkpoint,
                resume_path,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                train_cfg,
            )
            if pseudo_env_cache is not None and isinstance(checkpoint.get("pseudo_env_cache"), dict):
                pseudo_env_cache.load_state_dict(checkpoint["pseudo_env_cache"])
                print(f"[pseudo-env] restored cache from {resume_path}")
            ckpt_select_split = checkpoint.get("best_select_split")
            ckpt_select_metric = checkpoint.get("best_select_metric")
            if ckpt_select_split != best_select_split or ckpt_select_metric != best_select_metric:
                best_val = read_best_score_from_metrics(ckpt_dir, train_cfg)
                print(
                    "[resume] best selection config differs from checkpoint or is missing; "
                    f"using {best_select_split}/{best_select_metric} score from metrics: {best_val}"
                )
        except Exception as exc:
            explicit_resume = train_cfg.get("resume_from", "") not in (None, "", False, "auto")
            if explicit_resume or bool(train_cfg.get("resume_raise_on_error", False)):
                raise
            print(f"[resume] WARNING: failed to load {resume_path}: {exc}. Starting from scratch.")
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    if start_epoch > int(train_cfg["epochs"]):
        print(
            f"[resume] checkpoint already reached epoch {start_epoch - 1}; "
            f"TRAIN.epochs={train_cfg['epochs']}. Nothing to train."
        )
        save_run_complete_marker(ckpt_dir, cfg, start_epoch - 1, global_step, best_val, stopped_early=False)
        return

    last_epoch_ran = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        last_epoch_ran = epoch
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        pseudo_env_cache_updated = False
        if pseudo_env_cache is not None and pseudo_env_is_active(cfg, epoch):
            update_interval = max(1, int(cfg["LOSS"].get("pseudo_env_update_interval", 1)))
            if epoch % update_interval == 0:
                update_pseudo_env_cache(model, train_loader.dataset, cfg, device, data_scaler, pseudo_env_cache)
                pseudo_env_cache_updated = True
                cache_summary = pseudo_env_cache.summary()
                print(
                    "[pseudo-env] "
                    f"epoch={epoch} cache_updated=True "
                    f"counts={cache_summary['counts']} "
                    f"entropy={cache_summary['entropy']:.6f} "
                    f"qmax={cache_summary['qmax']:.6f}"
                )
        model.train()
        meters = AverageMeterDict()
        for batch_idx, batch in enumerate(train_loader, start=1):
            if train_cfg.get("max_train_batches") is not None and batch_idx > int(train_cfg["max_train_batches"]):
                break
            global_step += 1
            raw_batch = to_device_batch(batch, device)
            batch = preprocess_batch(raw_batch, cfg, data_scaler)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                output = model(
                    batch[input_key],
                    y_true=batch[target_key],
                    **get_time_kwargs(batch, cfg, include_future=True),
                )
                maybe_attach_pseudo_env_assignment(
                    output,
                    batch,
                    cfg,
                    pseudo_env_cache,
                    epoch=epoch,
                    cache_updated=pseudo_env_cache_updated,
                )
                register_grad_consensus_hook(output, grad_consensus, epoch=epoch)
                train_horizon = curriculum_horizon(train_cfg, epoch, cfg["DATASET"]["output_len"])
                loss_output, loss_targets, loss_mask, loss_raw_targets = slice_for_train_horizon(
                    output,
                    batch[target_key],
                    batch.get("targets_mask"),
                    raw_batch[target_key],
                    train_horizon,
                )
                loss, logs = loss_fn(
                    loss_output,
                    loss_targets,
                    loss_mask,
                    raw_y_true=loss_raw_targets,
                    data_scaler=data_scaler,
                )
                loss_terms = logs.pop("__loss_terms__", None)
                surgery_prepared = grad_surgery.prepare(
                    loss_terms,
                    grad_surgery_params,
                    scale=float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                    like=output["prediction"],
                )
                logs["curriculum_horizon"] = output["prediction"].new_tensor(float(train_horizon))
            scaler.scale(loss).backward()
            grad_surgery.apply(grad_surgery_params, surgery_prepared)
            logs.update(grad_consensus.log_tensors(output["prediction"]))
            logs.update(grad_surgery.log_tensors(output["prediction"]))
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            scalar_logs = {key: float(value.detach().cpu()) for key, value in logs.items()}
            meters.update(scalar_logs)
            if global_step % train_cfg.get("log_interval", 20) == 0:
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "train",
                    **scalar_logs,
                    **make_val_metric_row({}, lr=current_lr(optimizer)),
                }
                append_train_log(cfg, row)
                print(f"epoch={epoch} step={global_step} {format_logs(scalar_logs, log_keys_for_config(cfg))}")

        epoch_logs = meters.mean()
        append_train_log(
            cfg,
            {
                "epoch": epoch,
                "step": global_step,
                "split": "train_epoch",
                **epoch_logs,
                **make_val_metric_row({}, lr=current_lr(optimizer)),
            },
        )

        should_stop = False
        if epoch % train_cfg.get("val_interval", 1) == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                cfg,
                train_cfg.get("val_batches", 50),
                data_scaler,
            )
            print(
                f"epoch={epoch} val_mae={val_metrics['mae']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_mape={val_metrics['mape']:.6f} "
                f"val_wape={val_metrics.get('wape', float('nan')):.6f} "
                f"val_mae_h3={val_metrics.get('mae_h3', float('nan')):.6f} "
                f"val_mae_h6={val_metrics.get('mae_h6', float('nan')):.6f} "
                f"val_mae_h12={val_metrics.get('mae_h12', float('nan')):.6f} "
                f"lr={current_lr(optimizer):.8f}"
            )
            if lr_scheduler is not None:
                if str(train_cfg.get("lr_scheduler", "none")).lower() == "plateau":
                    lr_scheduler.step(val_metrics["mae"])
                else:
                    lr_scheduler.step()
            append_train_log(
                cfg,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "val",
                    **make_val_metric_row(val_metrics, lr=current_lr(optimizer)),
                },
            )
            last_payload = build_metrics_payload(
                cfg,
                epoch,
                global_step,
                val_metrics,
                ckpt_dir / "last.pt",
                split="val",
            )
            save_metrics_json(ckpt_dir / "last_metrics.json", last_payload)

            test_metrics = None
            if best_select_split == "test":
                test_metrics = evaluate(
                    model,
                    test_loader,
                    device,
                    cfg,
                    train_cfg.get("test_batches", None),
                    data_scaler,
                )
                append_train_log(
                    cfg,
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "split": "test_select",
                        **make_val_metric_row(test_metrics, lr=current_lr(optimizer)),
                    },
                )
                print(
                    f"epoch={epoch} test_select_mae={test_metrics['mae']:.6f} "
                    f"test_select_mse={test_metrics['mse']:.6f} "
                    f"test_select_rmse={test_metrics['rmse']:.6f} "
                    f"test_select_mape={test_metrics['mape']:.6f} "
                    f"test_select_wape={test_metrics.get('wape', float('nan')):.6f}"
                )

            select_metrics = test_metrics if best_select_split == "test" else val_metrics
            select_score = select_metrics[best_select_metric]
            improved = select_score < best_val
            if improved:
                best_val = select_score
                patience = 0
                if train_cfg.get("save_best", True):
                    best_ckpt_path = ckpt_dir / "best.pt"
                    atomic_torch_save(
                        make_checkpoint_payload(
                            cfg,
                            model,
                            optimizer,
                            lr_scheduler,
                            data_scaler,
                            scaler,
                            epoch,
                            global_step,
                            best_val,
                            patience,
                            pseudo_env_cache=pseudo_env_cache,
                            extra={
                                "best_select_split": best_select_split,
                                "best_select_metric": best_select_metric,
                                "best_select_score": best_val,
                                "val_mae": val_metrics["mae"],
                                "val_mse": val_metrics["mse"],
                                "val_rmse": val_metrics["rmse"],
                                "val_mape": val_metrics["mape"],
                                "test_mae": test_metrics["mae"] if test_metrics is not None else None,
                                "test_mse": test_metrics["mse"] if test_metrics is not None else None,
                                "test_rmse": test_metrics["rmse"] if test_metrics is not None else None,
                                "test_mape": test_metrics["mape"] if test_metrics is not None else None,
                            },
                        ),
                        best_ckpt_path,
                    )
                    best_payload = build_metrics_payload(
                        cfg,
                        epoch,
                        global_step,
                        val_metrics,
                        best_ckpt_path,
                        split="val",
                    )
                    best_payload["best_select_split"] = best_select_split
                    best_payload["best_select_metric"] = best_select_metric
                    best_payload["best_select_score"] = best_val
                    save_metrics_json(ckpt_dir / "best_metrics.json", best_payload)
                    if train_cfg.get("eval_test_on_best", True) or best_select_split == "test":
                        if test_metrics is None:
                            test_metrics = evaluate(
                                model,
                                test_loader,
                                device,
                                cfg,
                                train_cfg.get("test_batches", None),
                                data_scaler,
                            )
                        test_metrics_for_save = dict(test_metrics)
                        diagnostic_metrics = save_test_diagnostics(
                            model,
                            test_loader,
                            device,
                            cfg,
                            train_cfg.get("test_batches", None),
                            data_scaler,
                            ckpt_dir,
                        )
                        for diag_key, diag_value in diagnostic_metrics.items():
                            test_metrics_for_save[f"diag_{diag_key}"] = diag_value
                        test_payload = build_metrics_payload(
                            cfg,
                            epoch,
                            global_step,
                            test_metrics_for_save,
                            best_ckpt_path,
                            split="test",
                        )
                        test_payload["best_select_split"] = best_select_split
                        test_payload["best_select_metric"] = best_select_metric
                        test_payload["best_select_score"] = best_val
                        save_metrics_json(ckpt_dir / "best_test_metrics.json", test_payload)
                        append_train_log(
                            cfg,
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "split": "test",
                                **make_val_metric_row(test_metrics_for_save, lr=current_lr(optimizer)),
                            },
                        )
                        print(
                            f"best_by_{best_select_split} {best_select_metric}={best_val:.6f} "
                            f"test_mae={test_metrics_for_save['mae']:.6f} "
                            f"test_mse={test_metrics_for_save['mse']:.6f} "
                            f"test_rmse={test_metrics_for_save['rmse']:.6f} "
                            f"test_mape={test_metrics_for_save['mape']:.6f} "
                            f"test_wape={test_metrics_for_save.get('wape', float('nan')):.6f}"
                        )
                    print(f"saved best checkpoint: {best_ckpt_path}")
            else:
                patience += 1
                early_stop_patience = train_cfg.get("early_stop_patience")
                if early_stop_patience and patience >= early_stop_patience:
                    print(
                        f"early stopping at epoch={epoch}, "
                        f"best_{best_select_split}_{best_select_metric}={best_val:.6f}"
                    )
                    should_stop = True
                    stopped_early = True
        if train_cfg.get("save_last", True):
            atomic_torch_save(
                make_checkpoint_payload(
                    cfg,
                    model,
                    optimizer,
                    lr_scheduler,
                    data_scaler,
                    scaler,
                    epoch,
                    global_step,
                    best_val,
                    patience,
                    pseudo_env_cache=pseudo_env_cache,
                ),
                ckpt_dir / "last.pt",
            )
        if should_stop:
            break
    save_run_complete_marker(ckpt_dir, cfg, last_epoch_ran, global_step, best_val, stopped_early)


def train_with_basicts_launcher(cfg: Dict) -> None:
    from basicts import BasicTSLauncher
    from basicts.configs import BasicTSForecastingConfig

    cfg = finalize_config(cfg)
    if get_scaler_cfg(cfg).get("enabled", True):
        raise NotImplementedError(
            "The local runner now enforces GraphWaveNet/BasicTS-style scaling. "
            "BasicTS launcher scaler wiring for NUE-STG dict outputs is not supported here; "
            "use --runner local for scaled experiments."
        )
    warnings.warn(
        "BasicTS launcher support is experimental for NUE-STG. "
        "The local runner is the recommended experiment path because it guarantees dict-output auxiliary losses, "
        "full loss logging, and gate diagnostics.",
        RuntimeWarning,
    )
    ds_cfg = cfg["DATASET"]
    train_cfg = cfg["TRAIN"]
    device = train_cfg.get("device", "cpu")
    gpus = device.split(":", 1)[1] if device.startswith("cuda:") else None

    basicts_cfg = BasicTSForecastingConfig(
        model=NUESTG,
        model_config=NUESTGConfig(**cfg["MODEL"]),
        dataset_name=ds_cfg["name"],
        dataset_params={
            "input_len": ds_cfg["input_len"],
            "output_len": ds_cfg["output_len"],
            "use_timestamps": ds_cfg.get("use_timestamps", False),
            "data_file_path": ds_cfg["data_file_path"],
            "memmap": ds_cfg.get("memmap", True),
        },
        input_len=ds_cfg["input_len"],
        output_len=ds_cfg["output_len"],
        gpus=gpus,
        scaler=None,
        loss=make_basicts_loss(cfg["LOSS"]),
        metrics=[],
        target_metric="loss",
        num_epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        optimizer_params={"lr": train_cfg["learning_rate"], "weight_decay": train_cfg.get("weight_decay", 0.0)},
        ckpt_save_dir=train_cfg["ckpt_dir"] + "_basicts",
        eval_after_train=False,
    )
    BasicTSLauncher.launch_training(basicts_cfg)


def _parse_optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def apply_pseudo_env_cli_args(cfg: Dict, args: argparse.Namespace) -> None:
    loss_cfg = cfg.setdefault("LOSS", {})
    bool_fields = {
        "use_pseudo_env_heads": "use_pseudo_env_heads",
        "pseudo_env_detach_assignment": "pseudo_env_detach_assignment",
        "pseudo_env_use_global_cache": "pseudo_env_use_global_cache",
        "pseudo_env_use_temporal_smoothing": "pseudo_env_use_temporal_smoothing",
        "use_env_routed_inv_heads": "use_env_routed_inv_heads",
        "env_route_replace_final": "env_route_replace_final",
        "env_route_detach_q_for_expert": "env_route_detach_q_for_expert",
        "env_route_use_oracle_weight_for_expert": "env_route_use_oracle_weight_for_expert",
        "env_route_alpha_detach": "env_route_alpha_detach",
    }
    int_fields = {
        "pseudo_env_k": "pseudo_env_k",
        "pseudo_env_warmup_epochs": "pseudo_env_warmup_epochs",
        "pseudo_env_update_interval": "pseudo_env_update_interval",
        "pseudo_env_smooth_radius": "pseudo_env_smooth_radius",
        "env_route_k": "env_route_k",
        "env_route_warmup_epochs": "env_route_warmup_epochs",
    }
    float_fields = {
        "pseudo_env_tau": "pseudo_env_tau",
        "pseudo_env_lambda_head": "pseudo_env_lambda_head",
        "pseudo_env_lambda_var": "pseudo_env_lambda_var",
        "pseudo_env_lambda_balance": "pseudo_env_lambda_balance",
        "pseudo_env_lambda_entropy": "pseudo_env_lambda_entropy",
        "pseudo_env_lambda_diverse": "pseudo_env_lambda_diverse",
        "env_route_tau": "env_route_tau",
        "env_route_oracle_tau": "env_route_oracle_tau",
        "env_route_lambda_final": "env_route_lambda_final",
        "env_route_lambda_global": "env_route_lambda_global",
        "env_route_lambda_route_soft": "env_route_lambda_route_soft",
        "env_route_lambda_expert": "env_route_lambda_expert",
        "env_route_lambda_router_oracle": "env_route_lambda_router_oracle",
        "env_route_lambda_balance": "env_route_lambda_balance",
        "env_route_lambda_diverse": "env_route_lambda_diverse",
        "env_route_lambda_entropy": "env_route_lambda_entropy",
    }
    str_fields = {
        "pseudo_env_assignment_mode": "pseudo_env_assignment_mode",
        "pseudo_env_level": "pseudo_env_level",
        "env_route_mode": "env_route_mode",
    }
    for arg_name, cfg_name in bool_fields.items():
        parsed = _parse_optional_bool(getattr(args, arg_name, None))
        if parsed is not None:
            loss_cfg[cfg_name] = parsed
    for arg_name, cfg_name in int_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            loss_cfg[cfg_name] = int(value)
    for arg_name, cfg_name in float_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            loss_cfg[cfg_name] = float(value)
    for arg_name, cfg_name in str_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            loss_cfg[cfg_name] = str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="NUE-STG experiment entrypoint.")
    parser.add_argument("--config", "--config_file", dest="config", required=True, help="Path to Python config file.")
    parser.add_argument("--debug_batch", action="store_true", help="Run one forward/loss/backward smoke test.")
    parser.add_argument("--runner", choices=["local", "basicts"], default="local")
    parser.add_argument("--set", dest="dotlist", action="append", default=[], help="Override config, e.g. LOSS.lambda_kl=1e-5")
    parser.add_argument("--ablation", action="append", default=[], help="Apply named ablation.")
    parser.add_argument("--use_pseudo_env_heads", nargs="?", const="True", default=None)
    parser.add_argument("--pseudo_env_k", type=int, default=None)
    parser.add_argument("--pseudo_env_tau", type=float, default=None)
    parser.add_argument("--pseudo_env_lambda_head", type=float, default=None)
    parser.add_argument("--pseudo_env_lambda_var", type=float, default=None)
    parser.add_argument("--pseudo_env_lambda_balance", type=float, default=None)
    parser.add_argument("--pseudo_env_lambda_entropy", type=float, default=None)
    parser.add_argument("--pseudo_env_lambda_diverse", type=float, default=None)
    parser.add_argument("--pseudo_env_warmup_epochs", type=int, default=None)
    parser.add_argument("--pseudo_env_update_interval", type=int, default=None)
    parser.add_argument("--pseudo_env_detach_assignment", nargs="?", const="True", default=None)
    parser.add_argument("--pseudo_env_use_global_cache", nargs="?", const="True", default=None)
    parser.add_argument("--pseudo_env_use_temporal_smoothing", nargs="?", const="True", default=None)
    parser.add_argument("--pseudo_env_smooth_radius", type=int, default=None)
    parser.add_argument("--pseudo_env_assignment_mode", default=None)
    parser.add_argument("--pseudo_env_level", default=None)
    parser.add_argument("--use_env_routed_inv_heads", nargs="?", const="True", default=None)
    parser.add_argument("--env_route_k", type=int, default=None)
    parser.add_argument("--env_route_tau", type=float, default=None)
    parser.add_argument("--env_route_oracle_tau", type=float, default=None)
    parser.add_argument("--env_route_mode", default=None)
    parser.add_argument("--env_route_replace_final", nargs="?", const="True", default=None)
    parser.add_argument("--env_route_lambda_final", type=float, default=None)
    parser.add_argument("--env_route_lambda_global", type=float, default=None)
    parser.add_argument("--env_route_lambda_route_soft", type=float, default=None)
    parser.add_argument("--env_route_lambda_expert", type=float, default=None)
    parser.add_argument("--env_route_lambda_router_oracle", type=float, default=None)
    parser.add_argument("--env_route_lambda_balance", type=float, default=None)
    parser.add_argument("--env_route_lambda_diverse", type=float, default=None)
    parser.add_argument("--env_route_lambda_entropy", type=float, default=None)
    parser.add_argument("--env_route_warmup_epochs", type=int, default=None)
    parser.add_argument("--env_route_detach_q_for_expert", nargs="?", const="True", default=None)
    parser.add_argument("--env_route_use_oracle_weight_for_expert", nargs="?", const="True", default=None)
    parser.add_argument("--env_route_alpha_detach", nargs="?", const="True", default=None)
    args = parser.parse_args()

    cfg = resolve_cli_config(args.config, args.ablation, args.dotlist)
    apply_pseudo_env_cli_args(cfg, args)
    if args.debug_batch:
        debug_batch(cfg)
        return
    if args.runner == "basicts":
        train_with_basicts_launcher(cfg)
    else:
        train_local(cfg)


if __name__ == "__main__":
    main()
