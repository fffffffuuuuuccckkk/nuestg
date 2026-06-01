from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STONE-fixed-node-adapter",
            "category": "st_ood",
            "setting": "fixed_node_forecasting",
            "status": "runnable",
            "reference_status": "simplified",
            "notes": "Pure-PyTorch fixed-node STONE adapter; PEMS lacks official coordinates/meta side information.",
        },
        "MODEL": {
            "backbone_name": "stone",
            "backbone": {
                "name": "stone",
                "stone": {
                    "representation_dim": 64,
                    "temporal_channels": 128,
                    "sem_dim": 64,
                    "hidden_dim": 64,
                    "gate_dim": 128,
                    "Kt": 3,
                    "dropout": 0.3,
                },
            },
        },
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/baseline_stone_adapter"},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
