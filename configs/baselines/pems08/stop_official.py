from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


UNSUPPORTED_REASON = (
    "Full official STOP requires official SOOD/LargeST/KnowAir/TrafficStream "
    "protocol. Current PEMS08 config only supports STOP-adapter."
)

CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STOP-official",
            "display_name": "STOP-official",
            "category": "st_ood",
            "setting": "official_protocol",
            "status": "official_check",
            "reference_status": "official_local_wrapper",
            "main_table_safe": False,
            "is_official": False,
            "is_adapter": False,
            "unsupported_reason": UNSUPPORTED_REASON,
        },
        "DATASET": {"use_timestamps": True},
        "MODEL": {
            "backbone_name": "stop_official",
            "reference_status": "official_local_wrapper",
            "external_path": "/data/OuXiaoyu/mystg/baselines/STOP",
            "official_requires_special_data": True,
            "unsupported_reason": UNSUPPORTED_REASON,
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {"name": "stop_official", "stop_official": {"representation_dim": 64}},
            "use_timestamp": True,
            "required_timestamp": True,
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stop_official"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
