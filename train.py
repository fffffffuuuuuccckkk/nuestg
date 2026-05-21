from __future__ import annotations

import argparse
import random
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
    resolve_cli_config,
    save_resolved_config,
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
    "delta_gain_std",
    "delta_gain_pos_ratio",
    "s_gain_mean",
    "potential_gain_mean",
    "swap_delta_mean",
    "env_mu_abs_mean",
    "env_std_mean",
    "r_env_abs_mean",
    "y_inv_mae",
    "y_potential_mae",
    "y_hat_mae",
]
CSV_FIELDS = ["epoch", "step", "split", *LOG_KEYS, "val_mae"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finalize_config(cfg: Dict) -> Dict:
    """Keep duplicated model/dataset shape fields synchronized."""
    ds_cfg = cfg["DATASET"]
    model_cfg = cfg["MODEL"]
    for key in ["input_len", "output_len", "input_dim", "output_dim", "num_nodes"]:
        model_cfg[key] = ds_cfg[key]
    model_cfg["adj_path"] = ds_cfg.get("adj_path", model_cfg.get("adj_path", ""))
    model_cfg["swap"] = cfg.get("SWAP", {})
    model_cfg["swap_detach_inv"] = cfg.get("LOSS", {}).get("swap_detach_inv", True)
    cfg["LOSS"]["null_val"] = ds_cfg.get("null_val", cfg["LOSS"].get("null_val"))
    return cfg


def get_device(train_cfg: Dict) -> torch.device:
    requested = train_cfg.get("device", "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: Dict, split: str) -> BasicTSForecastingDataset:
    ds_cfg = cfg["DATASET"]
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
    return DataLoader(
        build_dataset(cfg, split),
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", True) and torch.cuda.is_available(),
    )


def to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device=device, dtype=torch.float32)
        else:
            out[key] = value
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


def check_output_shapes(output: Dict[str, torch.Tensor], targets: torch.Tensor, cfg: Dict) -> None:
    batch_size = targets.shape[0]
    output_len = cfg["DATASET"]["output_len"]
    num_nodes = cfg["DATASET"]["num_nodes"]
    output_dim = cfg["DATASET"]["output_dim"]
    expected = {
        "prediction": (batch_size, output_len, num_nodes, output_dim),
        "y_inv": (batch_size, output_len, num_nodes, output_dim),
        "y_potential": (batch_size, output_len, num_nodes, output_dim),
        "r_env": (batch_size, output_len, num_nodes, output_dim),
        "rho": (batch_size, output_len, num_nodes, 1),
        "z_inv": (batch_size, num_nodes, cfg["MODEL"]["hidden_dim"]),
        "env_mu": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env_logvar": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
        "env": (batch_size, num_nodes, cfg["MODEL"]["env_dim"]),
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
    aligned = align_target(targets, output["prediction"])
    print(f"aligned_targets: {tuple(aligned.shape)}")


def debug_batch(cfg: Dict) -> None:
    cfg = finalize_config(cfg)
    train_cfg = cfg["TRAIN"]
    set_seed(train_cfg["seed"])
    device = get_device(train_cfg)
    loader = build_loader(cfg, "train", shuffle=True)
    model, loss_fn = build_model_and_loss(cfg, device)
    loss_fn.set_epoch(1)
    model.train()

    batch = to_device_batch(next(iter(loader)), device)
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    print(f"{input_key}: {tuple(batch[input_key].shape)}")
    print(f"{target_key}: {tuple(batch[target_key].shape)}")
    print(f"inputs_after_align: {tuple(batch[input_key].shape)}")
    print(f"targets_before_align: {tuple(batch[target_key].shape)}")

    output = model(batch[input_key])
    check_output_shapes(output, batch[target_key], cfg)
    loss, logs = loss_fn(output, batch[target_key])
    loss.backward()
    assert_finite(loss, "total_loss")
    assert_finite(output["rho"], "rho")
    print(format_logs(logs, LOG_KEYS))
    print("debug_batch ok: forward/loss/backward finished without NaN or shape errors")


@torch.no_grad()
def evaluate(model: NUESTG, loader: DataLoader, device: torch.device, cfg: Dict, max_batches: int) -> float:
    model.eval()
    values = []
    input_key = cfg["DATASET"].get("input_key", "inputs")
    target_key = cfg["DATASET"].get("target_key", "targets")
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        batch = to_device_batch(batch, device)
        output = model(batch[input_key])
        mae = nue_mae_metric(output["prediction"], batch[target_key])
        values.append(float(mae.detach().cpu()))
    model.train()
    return float(np.mean(values)) if values else float("nan")


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
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    model, loss_fn = build_model_and_loss(cfg, device)
    optimizer = build_optimizer(cfg, model)
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
            batch = to_device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                output = model(batch[input_key])
                loss, logs = loss_fn(output, batch[target_key])
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
                row = {"epoch": epoch, "step": global_step, "split": "train", **scalar_logs, "val_mae": ""}
                append_train_log(cfg, row)
                print(f"epoch={epoch} step={global_step} {format_logs(scalar_logs, LOG_KEYS)}")

        epoch_logs = meters.mean()
        append_train_log(cfg, {"epoch": epoch, "step": global_step, "split": "train_epoch", **epoch_logs, "val_mae": ""})

        if train_cfg.get("save_last", True):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "global_step": global_step,
                },
                ckpt_dir / "last.pt",
            )

        if epoch % train_cfg.get("val_interval", 1) == 0:
            val_mae = evaluate(model, val_loader, device, cfg, train_cfg.get("val_batches", 50))
            print(f"epoch={epoch} val_mae={val_mae:.6f}")
            append_train_log(cfg, {"epoch": epoch, "step": global_step, "split": "val", "val_mae": val_mae})
            improved = val_mae < best_val
            if improved:
                best_val = val_mae
                patience = 0
                if train_cfg.get("save_best", True):
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": cfg,
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_mae": val_mae,
                        },
                        ckpt_dir / "best.pt",
                    )
                    print(f"saved best checkpoint: {ckpt_dir / 'best.pt'}")
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
