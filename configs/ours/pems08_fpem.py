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
        "DATASET": {
            "use_timestamps": True,
        },
        "MODEL": {
            "method_variant": "fpem",
            "use_timestamp": True,
            "time_encoding_type": "stid",
            "time_emb_dim": 32,
            "tod_emb_dim": 16,
            "dow_emb_dim": 8,
            "num_time_in_day": 288,
            "num_day_in_week": 7,
            "timestamp_feature_dim": 0,
            "use_time_of_day": True,
            "use_day_of_week": True,
            "use_current_timestamp_for_z": True,
            "use_current_timestamp_for_env": True,
            "required_timestamp": False,
            "env_token_mode": True,
            "mask_hidden_dim": 64,
            "mask_dropout": 0.1,
            "mask_init_bias": -1.0,
            "mask_temperature": 1.0,
            "mask_pooling": "masked_mean",
            "mask_use_time": True,
            "fusion_type": "film",
            "fusion_hidden_dim": 64,
            "fusion_dropout": 0.1,
            "fusion_zero_init": True,
            "future_decoder_hidden_dim": 64,
            "future_decoder_dropout": 0.1,
            "future_decoder_use_time": True,
            "future_decoder_logvar_min": -8.0,
            "future_decoder_logvar_max": 4.0,
            "persistence": {
                "enabled": False,
            },
        },
        "LOSS": {
            "use_gate": False,
            "lambda_gate": 0.0,
            "use_residual_norm": False,
            "lambda_residual_norm": 0.0,
            "use_persistence_mi": False,
            "use_envpred": True,
            "lambda_envpred": 0.05,
            "envpred_loss_type": "mse",
            "use_future_mi": True,
            "lambda_future_mi": 0.05,
            "future_mi_type": "ba_nll",
            "future_mi_detach_target": True,
            "future_mi_infonce_tau": 0.2,
            "infonce_granularity": "token",
            "use_ind": True,
            "lambda_ind": 1e-3,
            "sep_mi_type": "cross_cov",
            "sep_use_full_env": True,
            "use_club": False,
            "lambda_club": 1e-3,
            "club_separate_update": False,
            "club_negative_mode": "shuffle",
            "use_sparse": True,
            "use_mask_sparse": True,
            "lambda_sparse": 1e-3,
            "lambda_mask_sparse": 1e-3,
            "sparse_target": 0.3,
            "use_swap": True,
            "lambda_swap": 0.05,
            "swap_margin": 0.01,
            "swap_weight_mode": "future_env_diff",
            "swap_detach_env": True,
            "use_rank": False,
            "lambda_rank": 0.0,
            "rank_margin": 0.01,
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
