from __future__ import annotations

import argparse
import json
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicts.data import BasicTSForecastingDataset

from losses import NUESTGLoss, make_basicts_loss, nue_mae_metric
from models import NUESTG, NUESTGConfig
from utils import (
    AverageMeterDict,
    align_target,
    append_csv_log,
    assert_finite,
    format_logs,
    masked_mae_value,
    masked_mape_value,
    masked_rmse_value,
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
    "ind_loss",
    "sparse_loss",
    "entropy_loss",
    "residual_norm_loss",
    "env_consistency_loss",
    "persistence_mi_loss",
    "envpred_loss",
    "future_mi_loss",
    "future_mi_type",
    "env_fut_nll",
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
]
HORIZON_EVAL_STEPS = (3, 6, 12)
METRIC_FIELDS = [
    "val_mae",
    "val_rmse",
    "val_mape",
    "val_mae_h3",
    "val_rmse_h3",
    "val_mape_h3",
    "val_mae_h6",
    "val_rmse_h6",
    "val_mape_h6",
    "val_mae_h12",
    "val_rmse_h12",
    "val_mape_h12",
    "val_mae_avg12",
    "val_rmse_avg12",
    "val_mape_avg12",
    "lr",
]
CSV_FIELDS = ["epoch", "step", "split", *LOG_KEYS, *METRIC_FIELDS]

BACKBONE_DESCRIPTIONS = {
    "stid": "faithful_native_adapter STID backbone with spatial identity and TOD/DOW embeddings",
    "official_stid": "faithful_native_adapter STID backbone with spatial identity and TOD/DOW embeddings",
    "stid_mlp": "lightweight STID-like temporal MLP + node embedding",
    "mlp": "lightweight STID-like temporal MLP + node embedding",
    "stid_like": "lightweight STID-like temporal MLP + node embedding",
    "graphwavenet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "graph_wavenet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "gwnet": "graphwavenet_native_adapter backbone adapted to the shared interface",
    "agcrn": "faithful_native_adapter AGCRN backbone adapted to the shared interface",
    "stgcn": "reference_native STGCN backbone adapted from hazdzz/STGCN",
    "stnorm": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "st_norm": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "stnorm_wavenet": "stnorm_wavenet_adapter backbone with model-internal ST-Norm",
    "d2stgnn": "official_local_wrapper D2STGNN backbone adapted to the shared interface",
    "cast": "cast_fixed_node_simplified_adapter; not full official PyG/ST-OOD pipeline",
    "stone": "stone_fixed_node_simplified_adapter with learnable semantic fallback",
    "stop": "stop_architecture_adapter_without_sood_protocol adapted from STOP architecture",
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
    run_cfg = cfg.setdefault("RUN", {})
    swap_cfg = cfg.get("SWAP", {})
    cfg.setdefault("EVAL", {}).setdefault("horizon_metrics", True)
    scaler_cfg = cfg.setdefault("SCALER", {})
    scaler_cfg.setdefault("enabled", True)
    scaler_cfg.setdefault("type", "zscore")
    scaler_cfg.setdefault("norm_each_channel", False)
    scaler_cfg.setdefault("rescale", True)
    scaler_cfg.setdefault("eps", 1e-5)
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
    model_cfg["swap"] = cfg.get("SWAP", {})
    model_cfg["swap_detach_inv"] = cfg.get("LOSS", {}).get("swap_detach_inv", True)
    cfg["LOSS"]["z_dim"] = representation_dim
    cfg["LOSS"]["env_dim"] = int(model_cfg.get("env_dim", 32))
    return cfg


def get_device(train_cfg: Dict) -> torch.device:
    requested = train_cfg.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: Dict, split: str) -> BasicTSForecastingDataset:
    ds_cfg = cfg["DATASET"]
    maybe_generate_timestamp_file(ds_cfg["data_file_path"], split, ds_cfg)
    return BasicTSForecastingDataset(
        dataset_name=ds_cfg["name"],
        input_len=ds_cfg["input_len"],
        output_len=ds_cfg["output_len"],
        mode=split,
        use_timestamps=ds_cfg.get("use_timestamps", False),
        data_file_path=ds_cfg["data_file_path"],
        memmap=ds_cfg.get("memmap", True),
    )


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
    targets_mask = make_valid_mask(batch[target_key], null_val)
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
            out[key] = value.to(device=device, dtype=torch.float32)
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


def build_model_and_loss(cfg: Dict, device: torch.device) -> Tuple[NUESTG, NUESTGLoss]:
    model = NUESTG(NUESTGConfig(**cfg["MODEL"])).to(device)
    loss_fn = NUESTGLoss(**cfg["LOSS"]).to(device)
    return model, loss_fn


def build_optimizer(cfg: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = cfg["TRAIN"]
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    params = {
        "lr": train_cfg["learning_rate"],
        "weight_decay": train_cfg.get("weight_decay", 0.0),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
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
        ]:
            value = output.get(key)
            if value is None:
                print(f"{key}: None")
            else:
                print(f"{key}: {tuple(value.shape)}")
                assert_finite(value, key)
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
    for key in ["prediction_swap", "rho_swap", "env_perm"]:
        value = output.get(key)
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {tuple(value.shape)}")
            assert_finite(value, key)
    for key in ["env_fut", "persist_q", "persist_k", "persist_score"]:
        value = output.get(key)
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {tuple(value.shape)}")
            assert_finite(value, key)
    aligned = align_target(targets, output["prediction"])
    print(f"aligned_targets: {tuple(aligned.shape)}")


def debug_batch(cfg: Dict) -> None:
    cfg = finalize_config(cfg)
    train_cfg = cfg["TRAIN"]
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg)
    data_scaler = build_data_scaler(cfg, device)
    loader = build_loader(cfg, "train", shuffle=True)
    model, loss_fn = build_model_and_loss(cfg, device)
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
    check_output_shapes(output, batch[target_key], cfg)
    loss, logs = loss_fn(
        output,
        batch[target_key],
        batch.get("targets_mask"),
        raw_y_true=raw_batch[target_key],
        data_scaler=data_scaler,
    )
    loss.backward()
    assert_finite(loss, "total_loss")
    assert_finite(output["rho"], "rho")
    pred_original = data_scaler.inverse_transform(output["prediction"])
    original_mae = masked_mae_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
    )
    original_rmse = masked_rmse_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
    )
    original_mape = masked_mape_value(
        pred_original,
        raw_batch[target_key],
        cfg["DATASET"].get("null_val", cfg["LOSS"].get("null_val")),
    )
    print(f"debug_original_scale_mae={float(original_mae.detach().cpu()):.6f}")
    print(f"debug_original_scale_rmse={float(original_rmse.detach().cpu()):.6f}")
    print(f"debug_original_scale_mape={float(original_mape.detach().cpu()):.6f}")
    print(format_logs(logs, LOG_KEYS))
    print("debug_batch ok: forward/loss/backward finished without NaN or shape errors")


@torch.no_grad()
def compute_metric_dict(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    null_val,
    existing_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    metrics = {
        "mae": float(masked_mae_value(prediction, targets, null_val, existing_mask).detach().cpu()),
        "rmse": float(masked_rmse_value(prediction, targets, null_val, existing_mask).detach().cpu()),
        "mape": float(masked_mape_value(prediction, targets, null_val, existing_mask).detach().cpu()),
    }
    horizon = prediction.shape[1]
    eval_horizon = min(12, horizon)
    avg_mask = existing_mask[:, :eval_horizon] if existing_mask is not None else None
    metrics["mae_avg12"] = float(
        masked_mae_value(prediction[:, :eval_horizon], targets[:, :eval_horizon], null_val, avg_mask).detach().cpu()
    )
    metrics["rmse_avg12"] = float(
        masked_rmse_value(prediction[:, :eval_horizon], targets[:, :eval_horizon], null_val, avg_mask).detach().cpu()
    )
    metrics["mape_avg12"] = float(
        masked_mape_value(prediction[:, :eval_horizon], targets[:, :eval_horizon], null_val, avg_mask).detach().cpu()
    )
    for step in HORIZON_EVAL_STEPS:
        if horizon >= step:
            pred_h = prediction[:, step - 1 : step]
            target_h = targets[:, step - 1 : step]
            mask_h = existing_mask[:, step - 1 : step] if existing_mask is not None else None
            metrics[f"mae_h{step}"] = float(masked_mae_value(pred_h, target_h, null_val, mask_h).detach().cpu())
            metrics[f"rmse_h{step}"] = float(masked_rmse_value(pred_h, target_h, null_val, mask_h).detach().cpu())
            metrics[f"mape_h{step}"] = float(masked_mape_value(pred_h, target_h, null_val, mask_h).detach().cpu())
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
    for step, batch in enumerate(loader):
        if not full_eval and step >= int(max_batches):
            break
        raw_batch = to_device_batch(batch, device)
        batch = preprocess_batch(raw_batch, cfg, data_scaler)
        output = model(
            batch[input_key],
            **get_time_kwargs(batch, cfg, include_future=False),
        )
        prediction = data_scaler.inverse_transform(output["prediction"])
        targets = raw_batch[target_key]
        batch_metrics = compute_metric_dict(prediction, targets, null_val, batch.get("targets_mask"))
        for key, value in batch_metrics.items():
            values.setdefault(key, []).append(value)
    model.train()
    if not values.get("mae"):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    return {key: float(np.mean(item_values)) for key, item_values in values.items()}


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
        "category": run_cfg.get("category", "plugin_ours"),
        "backbone": model_cfg.get("backbone_name", ""),
        "ablation": ",".join(run_cfg.get("ablations", [])) if run_cfg.get("ablations") else run_cfg.get("ablation", ""),
        "reference_status": model_cfg.get("reference_status", run_cfg.get("reference_status", "")),
        "seed": cfg["TRAIN"].get("seed"),
        "epoch": epoch,
        "global_step": global_step,
        "split": split,
        "mae": metrics.get("mae", float("nan")),
        "rmse": metrics.get("rmse", float("nan")),
        "mape": metrics.get("mape", float("nan")),
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


def make_val_metric_row(metrics: Dict[str, float], lr: float | None = None) -> Dict[str, object]:
    row = {
        "val_mae": metrics.get("mae", ""),
        "val_rmse": metrics.get("rmse", ""),
        "val_mape": metrics.get("mape", ""),
        "val_mae_avg12": metrics.get("mae_avg12", metrics.get("mae", "")),
        "val_rmse_avg12": metrics.get("rmse_avg12", metrics.get("rmse", "")),
        "val_mape_avg12": metrics.get("mape_avg12", metrics.get("mape", "")),
        "lr": "" if lr is None else lr,
    }
    for step in HORIZON_EVAL_STEPS:
        row[f"val_mae_h{step}"] = metrics.get(f"mae_h{step}", "")
        row[f"val_rmse_h{step}"] = metrics.get(f"rmse_h{step}", "")
        row[f"val_mape_h{step}"] = metrics.get(f"mape_h{step}", "")
    return row


def append_train_log(cfg: Dict, row: Dict) -> None:
    if not cfg["LOGGING"].get("save_csv_log", True):
        return
    csv_path = Path(cfg["TRAIN"]["ckpt_dir"]) / cfg["LOGGING"].get("csv_log_path", "train_log.csv")
    append_csv_log(csv_path, row, CSV_FIELDS)


def train_local(cfg: Dict) -> None:
    cfg = finalize_config(cfg)
    train_cfg = cfg["TRAIN"]
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg)
    data_scaler = build_data_scaler(cfg, device)
    cfg.setdefault("SCALER", {})
    cfg["SCALER"]["stats"] = data_scaler.state_dict()
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    test_loader = build_loader(cfg, "test", shuffle=False)
    model, loss_fn = build_model_and_loss(cfg, device)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
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

    best_val = float("inf")
    patience = 0
    global_step = 0
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")

    for epoch in range(1, train_cfg["epochs"] + 1):
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
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
                loss, logs = loss_fn(
                    output,
                    batch[target_key],
                    batch.get("targets_mask"),
                    raw_y_true=raw_batch[target_key],
                    data_scaler=data_scaler,
                )
            scaler.scale(loss).backward()
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
                print(f"epoch={epoch} step={global_step} {format_logs(scalar_logs, LOG_KEYS)}")

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

        if train_cfg.get("save_last", True):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "config": cfg,
                    "scaler": data_scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                ckpt_dir / "last.pt",
            )

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
                f"val_rmse={val_metrics['rmse']:.6f} val_mape={val_metrics['mape']:.6f} "
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
            improved = val_metrics["mae"] < best_val
            if improved:
                best_val = val_metrics["mae"]
                patience = 0
                if train_cfg.get("save_best", True):
                    best_ckpt_path = ckpt_dir / "best.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                            "config": cfg,
                            "scaler": data_scaler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_mae": val_metrics["mae"],
                            "val_rmse": val_metrics["rmse"],
                            "val_mape": val_metrics["mape"],
                        },
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
                    save_metrics_json(ckpt_dir / "best_metrics.json", best_payload)
                    if train_cfg.get("eval_test_on_best", True):
                        test_metrics = evaluate(
                            model,
                            test_loader,
                            device,
                            cfg,
                            train_cfg.get("test_batches", None),
                            data_scaler,
                        )
                        test_payload = build_metrics_payload(
                            cfg,
                            epoch,
                            global_step,
                            test_metrics,
                            best_ckpt_path,
                            split="test",
                        )
                        save_metrics_json(ckpt_dir / "best_test_metrics.json", test_payload)
                        append_train_log(
                            cfg,
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "split": "test",
                                **make_val_metric_row(test_metrics, lr=current_lr(optimizer)),
                            },
                        )
                        print(
                            f"best test_mae={test_metrics['mae']:.6f} "
                            f"test_rmse={test_metrics['rmse']:.6f} test_mape={test_metrics['mape']:.6f}"
                        )
                    print(f"saved best checkpoint: {best_ckpt_path}")
            else:
                patience += 1
                early_stop_patience = train_cfg.get("early_stop_patience")
                if early_stop_patience and patience >= early_stop_patience:
                    print(f"early stopping at epoch={epoch}, best_val_mae={best_val:.6f}")
                    break


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


def main() -> None:
    parser = argparse.ArgumentParser(description="NUE-STG experiment entrypoint.")
    parser.add_argument("--config", required=True, help="Path to Python config file.")
    parser.add_argument("--debug_batch", action="store_true", help="Run one forward/loss/backward smoke test.")
    parser.add_argument("--runner", choices=["local", "basicts"], default="local")
    parser.add_argument("--set", dest="dotlist", action="append", default=[], help="Override config, e.g. LOSS.lambda_kl=1e-5")
    parser.add_argument("--ablation", action="append", default=[], help="Apply named ablation.")
    args = parser.parse_args()

    cfg = resolve_cli_config(args.config, args.ablation, args.dotlist)
    if args.debug_batch:
        debug_batch(cfg)
        return
    if args.runner == "basicts":
        train_with_basicts_launcher(cfg)
    else:
        train_local(cfg)


if __name__ == "__main__":
    main()
