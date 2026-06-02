from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STGCN",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "reference_native",
            "notes": "Reference-native STGCN adapted from hazdzz/STGCN; uses local scaler/splits.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "stgcn",
            "reference_status": "reference_native",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {"name": "stgcn"},
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stgcn"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
