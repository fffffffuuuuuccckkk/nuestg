from __future__ import annotations

import copy

from configs.pems08_ood_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "GraphWaveNet-Full-OOD",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "official_graphwavenet_full_native_adapter",
            "dataset_protocol": "PEMS08-OOD",
            "notes": (
                "Pure Graph WaveNet on PEMS08-OOD. Uses the graphwavenet_full "
                "official prediction path with no NUE/FPEM environment branch; "
                "the second input channel is time-of-day from timestamps."
            ),
        },
        "DATASET": {
            "use_timestamps": True,
        },
        "MODEL": {
            "backbone_name": "graphwavenet_full",
            "reference_status": "official_graphwavenet_full_native_adapter",
            "use_timestamp": True,
            "required_timestamp": False,
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {
                "name": "graphwavenet_full",
                "graph_wavenet_full": {
                    "representation_dim": 64,
                    "dropout": 0.3,
                    "blocks": 4,
                    "layers": 2,
                    "kernel_size": 2,
                    "residual_channels": 32,
                    "dilation_channels": 32,
                    "skip_channels": 256,
                    "end_channels": 512,
                    "gcn_bool": True,
                    "addaptadj": True,
                    "in_dim": 2,
                    "supports_len": 2,
                    "use_static_adj": True,
                    "adjtype": "doubletransition",
                    "support_add_self_loop": False,
                    "randomadj": False,
                    "aptonly": False,
                    "engine_pad_input": True,
                    "use_time_of_day_channel": True,
                },
            },
        },
        "LOSS": {
            "loss_type": "mae",
            "null_val": None,
            "train_loss_scale": "original",
            "use_inv": False,
            "lambda_inv": 0.0,
        },
        "TRAIN": {
            "seed": 2026,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
            "val_batches": None,
            "ckpt_dir": "./checkpoints/pems08_ood/baseline_graphwavenet_full",
        },
        "EVAL": {
            "horizon_metrics": True,
            "save_test_diagnostics": True,
        },
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(
    CONFIG,
    {
        "MODEL": {
            "use_separated_z_for_y_inv": False,
        },
        "LOSS": {
            "train_loss_scale": "original",
            "use_inv": False,
            "lambda_inv": 0.0,
        },
        "SWAP": {
            "enabled": False,
        },
    },
)


def get_config():
    return CONFIG
