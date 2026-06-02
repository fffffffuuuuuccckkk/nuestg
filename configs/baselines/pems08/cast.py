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
            "reference_status": "cast_fixed_node_simplified_adapter",
            "notes": "Pure-PyTorch fixed-node CaST adapter; not the full official PyG/ST-OOD data-object reproduction.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "cast",
            "reference_status": "cast_fixed_node_simplified_adapter",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
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
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_cast_adapter"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
