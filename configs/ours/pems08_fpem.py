from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update


CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {
            "method": "FPEM-STIDMLP",
            "category": "plugin_ours",
            "setting": "forecasting",
            "status": "runnable",
            "notes": "Future-Predictive Environment Masking with latent FiLM fusion.",
        },
        "MODEL": {
            "method_variant": "fpem",
            "env_token_mode": True,
            "mask_hidden_dim": 64,
            "mask_dropout": 0.1,
            "mask_init_bias": -1.0,
            "mask_temperature": 1.0,
            "mask_pooling": "masked_mean",
            "fusion_type": "film",
            "fusion_hidden_dim": 64,
            "fusion_dropout": 0.1,
            "fusion_zero_init": True,
            "env_transition_hidden_dim": 64,
            "env_transition_dropout": 0.1,
        },
        "LOSS": {
            "use_gate": False,
            "lambda_gate": 0.0,
            "use_residual_norm": False,
            "lambda_residual_norm": 0.0,
            "use_envpred": True,
            "lambda_envpred": 0.05,
            "envpred_loss_type": "mse",
            "use_future_mi": True,
            "lambda_future_mi": 0.02,
            "future_mi_tau": 0.2,
            "use_ind": True,
            "lambda_ind": 1e-3,
            "use_sparse": True,
            "use_mask_sparse": True,
            "lambda_sparse": 1e-3,
            "sparse_target": 0.3,
            "use_swap": True,
            "lambda_swap": 0.05,
            "swap_margin": 0.01,
            "swap_weight_mode": "future_env_diff",
            "use_rank": False,
            "lambda_rank": 0.0,
            "use_kl": True,
            "lambda_kl": 1e-4,
        },
        "TRAIN": {
            "ckpt_dir": "./checkpoints/pems08_fpem",
        },
    },
)


def get_config():
    return CONFIG
