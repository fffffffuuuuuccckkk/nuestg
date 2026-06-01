from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "CaST-fixed-node-adapter",
            "category": "st_ood",
            "setting": "fixed_node_forecasting",
            "status": "runnable",
            "reference_status": "simplified",
            "notes": "Pure-PyTorch fixed-node CaST adapter; not the full official PyG/ST-OOD data-object reproduction.",
        },
        "MODEL": {
            "backbone_name": "cast",
            "backbone": {
                "name": "cast",
                "cast": {
                    "representation_dim": 64,
                    "hid_dim": 16,
                    "node_embed_dim": 5,
                    "K": 2,
                    "depth": 4,
                    "dropout": 0.2,
                    "n_envs": 5,
                },
            },
        },
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_cast_adapter"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
