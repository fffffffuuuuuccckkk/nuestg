from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


DATASET_DIR = "/data/OuXiaoyu/mystg/datasets/PEMS08-OOD"


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "DATASET": {
            "name": "PEMS08-OOD",
            "data_file_path": DATASET_DIR,
            "adj_path": f"{DATASET_DIR}/adj_mx.pkl",
            "num_nodes": 170,
            "input_dim": 1,
            "output_dim": 1,
            "use_timestamps": True,
            "null_val": None,
            "frequency_minutes": 5,
        },
        "MODEL": {
            "num_nodes": 170,
            "input_dim": 1,
            "output_dim": 1,
            "adj_path": f"{DATASET_DIR}/adj_mx.pkl",
            "use_timestamp": True,
            "time_encoding_type": "stid",
            "time_emb_dim": 32,
            "num_time_in_day": 288,
            "num_day_in_week": 7,
            "use_time_of_day": True,
            "use_day_of_week": True,
            "required_timestamp": False,
            "backbone_name": "stid_mlp",
            "backbone": {"name": "stid_mlp", "representation_dim": 64},
        },
        "RUN": {
            "method": "NUE-STG",
            "dataset_protocol": "PEMS08-OOD",
            "notes": "PEMS08 converted from ST-OOD: train/val use official ID splits, test uses official idx_shift OOD split.",
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/pems08_ood_nuestg",
        },
    },
)


def get_config():
    return CONFIG
