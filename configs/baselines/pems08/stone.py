from __future__ import annotations

import copy

from configs.pems08_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update


CONFIG = copy.deepcopy(BASE_CONFIG)
deep_update(
    CONFIG,
    {
        "RUN": {
            "method": "STONE-faithful-pytorch-adapter",
            "display_name": "STONE-faithful-pytorch-adapter",
            "category": "st_ood",
            "setting": "fixed_node_forecasting",
            "status": "runnable",
            "reference_status": "stone_faithful_pytorch_adapter_without_spatial_side_info",
            "is_adapter": True,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 fixed-node batches lack STONE official coordinate/Frechet/spatial-shift side information.",
            "notes": "Faithful STONE architecture adapter with official STBlock, STAggBlock, and GatedFusionBlock; semantic side info falls back to learnable node embeddings.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "stone",
            "reference_status": "stone_faithful_pytorch_adapter_without_spatial_side_info",
            "input_dim": 1,
            "output_dim": 1,
            "num_nodes": 170,
            "backbone": {
                "name": "stone",
                "stone": {
                    "representation_dim": 64,
                    "SBlocks_num": 2,
                    "TBlocks_num": 5,
                    "temporal_channels": 128,
                    "sem_dim": 64,
                    "hidden_dim": 64,
                    "x_output_dim": 128,
                    "sem_output_dim": 64,
                    "gate_output_dim": 128,
                    "Kt": 3,
                    "Ks_s": 1,
                    "Ks_t": 1,
                    "adp_s_dim": 20,
                    "adp_t_dim": 20,
                    "dropout": 0.3,
                },
            },
        },
        "LOSS": {"loss_type": "mae", "null_val": None},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stone_adapter"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
