from __future__ import annotations

import copy

from configs.ours.pems08.fpem_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "FPEM-AGCRN",
            "notes": "Future-Predictive Environment Masking with AGCRN backbone and latent FiLM fusion.",
        },
        "MODEL": {
            "backbone_name": "agcrn",
            "backbone": {"name": "agcrn"},
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/pems08/fpem_agcrn",
        },
    },
)


def get_config():
    return CONFIG
