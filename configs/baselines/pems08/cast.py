from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "CaST-faithful-pytorch-adapter",
            "display_name": "CaST-faithful-pytorch-adapter",
            "category": "st_ood",
            "setting": "fixed_node_forecasting",
            "status": "runnable",
            "reference_status": "cast_faithful_pytorch_adapter_with_official_aux_loss",
            "is_adapter": True,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 BasicTS fixed-node batches do not expose CaST official PyG PairData graph samples.",
            "notes": "Pure-PyTorch faithful CaST adapter with official temporal/entity/environment, Hodge edge, causal GCN, and VQ/commitment/MI losses.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "cast",
            "reference_status": "cast_faithful_pytorch_adapter_with_official_aux_loss",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {
                "name": "cast",
                "cast": {
                    "representation_dim": 64,
                    "hid_dim": 16,
                    "node_embed_dim": 5,
                    "K": 2,
                    "depth": 10,
                    "dropout": 0.2,
                    "n_envs": 5,
                    "time_delay_scaler": 6,
                    "beta1": 1.0,
                    "beta2": 1.0,
                },
            },
        },
        "LOSS": {"loss_type": "mae", "null_val": None, "use_backbone_aux": True, "lambda_backbone_aux": 1.0},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_cast_adapter"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
