from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "D2STGNN",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "official_local_wrapper",
            "notes": "Official D2STGNN model wrapper adapted from local reference repo; uses local scaler/splits.",
        },
        "DATASET": {"use_timestamps": True},
        "MODEL": {
            "backbone_name": "d2stgnn",
            "reference_status": "official_local_wrapper",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {
                "name": "d2stgnn",
                "d2stgnn": {
                    "representation_dim": 64,
                    "repo_root": "/data/OuXiaoyu/mystg/baselines/D2STGNN",
                    "num_hidden": 32,
                    "node_hidden": 10,
                    "time_emb_dim": 10,
                    "dropout": 0.1,
                    "k_t": 3,
                    "k_s": 2,
                    "gap": 3,
                    "num_modalities": 2,
                    "num_time_in_day": 288,
                    "num_day_in_week": 7,
                },
            },
            "use_timestamp": True,
            "required_timestamp": True,
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_d2stgnn"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
