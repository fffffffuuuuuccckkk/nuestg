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
            "reference_status": "faithful_native",
            "notes": "Native STGCN adapted from hazdzz/STGCN; uses local scaler/splits.",
        },
        "MODEL": {"backbone_name": "stgcn", "backbone": {"name": "stgcn"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_stgcn"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
