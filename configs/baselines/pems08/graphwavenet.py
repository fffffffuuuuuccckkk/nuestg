from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "GraphWaveNet-style",
            "category": "forecasting",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "graphwavenet_native_adapter",
            "notes": "Baseline-only Graph WaveNet native adapter checked against official model.py/util.py; uses local scaler/splits.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "graphwavenet",
            "reference_status": "graphwavenet_native_adapter",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "GWNET": {
                "residual_channels": 32,
                "dilation_channels": 32,
                "skip_channels": 256,
                "end_channels": 512,
                "blocks": 4,
                "layers": 2,
                "kernel_size": 2,
                "gcn_bool": True,
                "addaptadj": True,
                "adjtype": "doubletransition",
            },
            "backbone": {
                "name": "graphwavenet",
                "graph_wavenet": {
                    "residual_channels": 32,
                    "dilation_channels": 32,
                    "skip_channels": 256,
                    "end_channels": 512,
                    "blocks": 4,
                    "layers": 2,
                    "kernel_size": 2,
                    "gcn_bool": True,
                    "addaptadj": True,
                    "adjtype": "doubletransition",
                },
            },
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_graphwavenet"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
