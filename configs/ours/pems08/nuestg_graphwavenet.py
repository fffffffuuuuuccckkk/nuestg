from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "Ours-GraphWaveNet",
            "category": "plugin_ours",
            "setting": "forecasting",
            "status": "runnable",
            "notes": "Full NUE-STG with Graph WaveNet-style invariant backbone.",
        },
        "MODEL": {"backbone_name": "graphwavenet", "backbone": {"name": "graphwavenet"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ours_graphwavenet"},
    },
)


def get_config():
    return CONFIG
