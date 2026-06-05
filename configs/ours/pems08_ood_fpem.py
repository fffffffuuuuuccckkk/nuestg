from __future__ import annotations

import copy

from configs.ours.pems08_fpem import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


DATASET_DIR = "/data/OuXiaoyu/mystg/datasets/PEMS08-OOD"


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "FPEM-STIDMLP-OOD",
            "dataset_protocol": "PEMS08-OOD",
            "notes": "FPEM on PEMS08-OOD: train/val use official ST-OOD ID splits, test uses official idx_shift OOD split.",
        },
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
            "num_time_in_day": 288,
            "num_day_in_week": 7,
            "use_timestamp": True,
            "required_timestamp": True,
            "backbone": {"name": "stid_mlp"},
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/pems08_ood_fpem",
        },
    },
)


def get_config():
    return CONFIG
