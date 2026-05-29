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
            "reference_status": "faithful_native",
            "notes": "Native ST-Norm WaveNet adapted from official ST-Norm Wavenet.py; uses local scaler/splits.",
        },
        "MODEL": {"backbone_name": "stnorm", "backbone": {"name": "stnorm"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_stnorm"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
