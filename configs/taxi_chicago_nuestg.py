from __future__ import annotations

import copy

from configs.base_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


DATASET_DIR = "/data/OuXiaoyu/mystg/datasets/Taxi_Chicago"


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "DATASET": {
            "name": "Taxi_Chicago",
            "data_file_path": DATASET_DIR,
            "adj_path": f"{DATASET_DIR}/adj.npy",
            "num_nodes": 77,
            "input_dim": 1,
            "output_dim": 1,
            "use_timestamps": True,
            "null_val": None,
        },
        "MODEL": {
            "num_nodes": 77,
            "input_dim": 1,
            "output_dim": 1,
            "adj_path": f"{DATASET_DIR}/adj.npy",
            "use_timestamp": True,
            "time_encoding_type": "stid",
            "time_emb_dim": 32,
            "num_time_in_day": 24,
            "num_day_in_week": 7,
            "use_time_of_day": True,
            "use_day_of_week": True,
            "required_timestamp": False,
            "backbone_name": "stid_mlp",
            "backbone": {"name": "stid_mlp", "representation_dim": 64},
        },
        "TRAIN": {
            "batch_size": 16,
            "num_workers": 0,
            "ckpt_dir": "./checkpoints/taxi_chicago_nuestg",
        },
    },
)


def get_config():
    return CONFIG
