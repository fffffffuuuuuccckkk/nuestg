from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STID-like MLP",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "style_native",
            "notes": "Baseline-only lightweight STID-like MLP; not official STID.",
        },
        "MODEL": {"backbone_name": "stid_mlp", "backbone": {"name": "stid_mlp"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_stid_mlp"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
