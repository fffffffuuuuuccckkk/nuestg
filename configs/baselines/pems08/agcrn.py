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
            "notes": "Baseline-only AGCRN-style backbone; not official line-by-line reproduction.",
        },
        "MODEL": {"backbone_name": "agcrn", "backbone": {"name": "agcrn"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_agcrn"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
