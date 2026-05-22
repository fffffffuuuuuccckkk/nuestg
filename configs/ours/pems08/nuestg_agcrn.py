from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "Ours-AGCRN",
            "category": "plugin_ours",
            "setting": "forecasting",
            "status": "runnable",
            "notes": "Full NUE-STG with AGCRN-style invariant backbone.",
        },
        "MODEL": {"backbone_name": "agcrn", "backbone": {"name": "agcrn"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ours_agcrn"},
    },
)


def get_config():
    return CONFIG
