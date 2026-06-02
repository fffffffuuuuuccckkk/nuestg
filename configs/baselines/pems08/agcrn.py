from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "AGCRN-style",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "faithful_native_adapter",
            "notes": "Baseline-only faithful native AGCRN adapter checked against official AVWGCN/AGCRNCell/AGCRN; uses local scaler/splits.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "agcrn",
            "reference_status": "faithful_native_adapter",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {"name": "agcrn"},
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_agcrn"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
