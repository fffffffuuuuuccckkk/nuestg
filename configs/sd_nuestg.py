from __future__ import annotations

import copy

from configs.base_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


DATASET_DIR = "/data/OuXiaoyu/mystg/datasets/SD"


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "DATASET": {
            "name": "SD",
            "data_file_path": DATASET_DIR,
            "adj_path": f"{DATASET_DIR}/adj_mx.pkl",
            "num_nodes": 716,
            "input_dim": 1,
            "output_dim": 1,
            "null_val": None,
        },
        "MODEL": {
            "num_nodes": 716,
            "input_dim": 1,
            "output_dim": 1,
            "adj_path": f"{DATASET_DIR}/adj_mx.pkl",
            "backbone_name": "stid_mlp",
            "backbone": {"name": "stid_mlp", "representation_dim": 64},
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/sd_nuestg",
        },
    },
)


def get_config():
    return CONFIG
