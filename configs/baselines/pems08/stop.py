from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STOP-faithful-architecture-adapter",
            "display_name": "STOP-faithful-architecture-adapter",
            "category": "st_ood",
            "setting": "fixed_node_forecasting",
            "status": "runnable",
            "reference_status": "stop_faithful_architecture_adapter_without_sood_protocol",
            "is_adapter": True,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 fixed-node batches lack STOP official SOOD/OOD split and dataset protocol.",
            "notes": "Faithful STOP architecture adapter using official MLP decomposition/prompt encoder and Core_Adaptive backcast; special SOOD protocol is not reproduced.",
        },
        "DATASET": {"use_timestamps": True},
        "MODEL": {
            "backbone_name": "stop",
            "reference_status": "stop_faithful_architecture_adapter_without_sood_protocol",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {
                "name": "stop",
                "stop": {
                    "representation_dim": 64,
                    "num_layer": 3,
                    "model_dim": 64,
                    "prompt_dim": 32,
                    "kernel_size": 3,
                    "hid_dim": 256,
                    "tod_size": 288,
                    "extra_type": True,
                    "same": False,
                    "core": 8,
                    "head": 4,
                    "core_dropout": 0.3,
                    "num_time_in_day": 288,
                    "num_day_in_week": 7,
                },
            },
            "use_timestamp": True,
            "required_timestamp": True,
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stop"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
