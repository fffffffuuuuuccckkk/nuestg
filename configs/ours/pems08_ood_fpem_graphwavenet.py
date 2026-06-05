from __future__ import annotations

import copy

from configs.ours.pems08_ood_fpem import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "FPEM-GraphWaveNet-Full-OOD",
            "category": "plugin_ours",
            "setting": "forecasting",
            "status": "runnable",
            "reference_status": "fpem_with_official_graphwavenet_full_backbone",
            "dataset_protocol": "PEMS08-OOD",
            "notes": (
                "FPEM on PEMS08-OOD with Graph WaveNet as invariant backbone. "
                "The default backbone is graphwavenet_full; set MODEL.backbone_name=graphwavenet "
                "to use the older compact adapter."
            ),
        },
        "MODEL": {
            "backbone_name": "graphwavenet_full",
            "reference_status": "fpem_with_official_graphwavenet_full_backbone",
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
                "graph_wavenet": {
                    "dropout": 0.3,
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
        "TRAIN": {
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
            "ckpt_dir": "./checkpoints/pems08_ood/fpem_graphwavenet_full",
        },
    },
)


def get_config():
    return CONFIG
