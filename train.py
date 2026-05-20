from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicts.data import BasicTSForecastingDataset

from losses import NUESTGLoss, make_basicts_loss, nue_mae_metric
from models import NUESTG, NUESTGConfig
from utils import align_target, assert_finite


def load_config(config_path: str) -> Dict:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_config"):
        return module.get_config()
    if hasattr(module, "CONFIG"):
        return module.CONFIG
    raise AttributeError(f"{path} must define CONFIG or get_config()")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        pin_memory=torch.cuda.is_available(),
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


def format_logs(logs: Dict[str, torch.Tensor]) -> str:
    parts = []
    for key in [
        "pred_loss",
        "inv_loss",
        "gate_loss",
        "swap_loss",
        "kl_loss",
        "ind_loss",
        "sparse_loss",
        "rho_mean",
        "rho_std",
        "delta_gain_mean",
        "total_loss",
    ]:
        value = logs[key]
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        parts.append(f"{key}={value:.6f}")
    return " ".join(parts)


def check_output_shapes(output: Dict[str, torch.Tensor], targets: torch.Tensor) -> None:
    expected_keys = ["prediction", "y_inv", "r_env", "rho", "z_inv", "env_mu", "env_logvar", "env"]
    for key in expected_keys:
        print(f"{key}: {tuple(output[key].shape)}")
        assert_finite(output[key], key)
    y_true = align_target(targets, output["prediction"])
    print(f"aligned_targets: {tuple(y_true.shape)}")


def debug_batch(cfg: Dict) -> None:
    train_cfg = cfg["TRAIN"]
    set_seed(train_cfg["seed"])
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    loader = build_loader(cfg, "train", shuffle=True)
    model, loss_fn = build_model_and_loss(cfg, device)
    model.train()

    batch = to_device_batch(next(iter(loader)), device)
    print(f"inputs: {tuple(batch['inputs'].shape)}")
    print(f"targets: {tuple(batch['targets'].shape)}")

    output = model(batch["inputs"])
    check_output_shapes(output, batch["targets"])
    loss, logs = loss_fn(output, batch["targets"])
    loss.backward()
    assert_finite(loss, "loss")
    print(format_logs(logs))
    print("debug_batch ok: forward/loss/backward finished without NaN or shape errors")


@torch.no_grad()
def evaluate(model: NUESTG, loss_fn: NUESTGLoss, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    values = []
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        batch = to_device_batch(batch, device)
        output = model(batch["inputs"])
        mae = nue_mae_metric(output["prediction"], batch["targets"])
        values.append(float(mae.detach().cpu()))
    model.train()
    return float(np.mean(values)) if values else float("nan")


def train_local(cfg: Dict) -> None:
    train_cfg = cfg["TRAIN"]
    set_seed(train_cfg["seed"])
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    model, loss_fn = build_model_and_loss(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    global_step = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = to_device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["inputs"])
            loss, logs = loss_fn(output, batch["targets"])
            loss.backward()
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if global_step % train_cfg.get("log_interval", 20) == 0:
                print(f"epoch={epoch} step={global_step} {format_logs(logs)}")

        if epoch % train_cfg.get("val_interval", 1) == 0:
            val_mae = evaluate(model, loss_fn, val_loader, device, train_cfg.get("val_batches", 20))
            print(f"epoch={epoch} val_mae={val_mae:.6f}")
            if val_mae < best_val:
                best_val = val_mae
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": val_mae},
                    ckpt_dir / "best.pt",
                )
                print(f"saved best checkpoint: {ckpt_dir / 'best.pt'}")


def train_with_basicts_launcher(cfg: Dict) -> None:
    from basicts import BasicTSLauncher
    from basicts.configs import BasicTSForecastingConfig

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
        optimizer_params={"lr": train_cfg["learning_rate"]},
        ckpt_save_dir=train_cfg["ckpt_dir"] + "_basicts",
        eval_after_train=False,
    )
    BasicTSLauncher.launch_training(basicts_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="NUE-STG experiment entrypoint.")
    parser.add_argument("--config", required=True, help="Path to Python config file.")
    parser.add_argument("--debug_batch", action="store_true", help="Run one forward/loss/backward smoke test.")
    parser.add_argument("--runner", choices=["local", "basicts"], default="local")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug_batch:
        debug_batch(cfg)
        return
    if args.runner == "basicts":
        train_with_basicts_launcher(cfg)
    else:
        train_local(cfg)


if __name__ == "__main__":
    main()
