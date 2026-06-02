from __future__ import annotations

import copy

from configs.ours.pems08_fpem import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "FPEM-STIDMLP",
            "category": "plugin_ours",
            "setting": "forecasting",
            "status": "runnable",
            "notes": "Future-Predictive Environment Masking with STID-MLP backbone and latent FiLM fusion.",
        },
        "DATASET": {
            "use_timestamps": True,
        },
        "MODEL": {
            "backbone_name": "stid_mlp",
            "reference_status": "native_adapter",
            "method_variant": "fpem",
            "use_timestamp": True,
            "required_timestamp": True,
            "time_encoding_type": "stid",
            "env_token_mode": True,
            "fusion_type": "film",
            "backbone": {"name": "stid_mlp"},
        },
        "LOSS": {
            "use_future_mi": True,
            "future_mi_type": "ba_nll",
            "sep_mi_type": "cross_cov",
            "use_swap": True,
            "swap_weight_mode": "future_env_diff",
            "use_mask_sparse": True,
            "use_kl": True,
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/pems08/fpem_stid_mlp",
        },
        "EVAL": {
            "horizon_metrics": True,
        },
    },
)


def get_config():
    return CONFIG
