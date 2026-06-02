from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "ST-Norm",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "stnorm_wavenet_adapter",
            "notes": "ST-Norm WaveNet adapter checked against official ST-Norm Wavenet.py; uses local scaler/splits.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "stnorm",
            "reference_status": "stnorm_wavenet_adapter",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "STNORM": {
                "channels": 16,
                "blocks": 4,
                "layers": 2,
                "snorm": True,
                "tnorm": True,
            },
            "backbone": {
                "name": "stnorm",
                "stnorm_wavenet": {
                    "channels": 16,
                    "blocks": 4,
                    "layers": 2,
                    "snorm_bool": True,
                    "tnorm_bool": True,
                },
            },
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stnorm"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
