from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "CaST-official",
            "display_name": "CaST-official",
            "category": "st_ood",
            "setting": "official_protocol",
            "status": "official_check",
            "reference_status": "official_local_wrapper",
            "main_table_safe": False,
            "is_official": False,
            "is_adapter": False,
            "unsupported_reason": (
                "current PEMS08 BasicTS fixed-node config cannot provide the official "
                "PyG graph Data object, Hodge-Laplacian edge graph, CaST preprocessing, "
                "and VQ/MI loss protocol required by full CaST"
            ),
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "cast_official",
            "reference_status": "official_local_wrapper",
            "external_path": "/data/OuXiaoyu/mystg/baselines/CaST",
            "official_requires_special_data": True,
            "unsupported_reason": (
                "current PEMS08 BasicTS fixed-node config cannot provide the official "
                "PyG graph Data object, Hodge-Laplacian edge graph, CaST preprocessing, "
                "and VQ/MI loss protocol required by full CaST"
            ),
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {"name": "cast_official", "cast_official": {"representation_dim": 64}},
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_cast_official"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
