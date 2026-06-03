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
            "reference_repo": "/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024",
            "reference_level": "method_level_reproduction",
            "protocol": "BasicTS_fixed_node_PEMS08",
            "spatial_info": "fallback_from_adjacency_or_anchor_if_no_coordinates",
            "reference_status": "stone_method_level_pytorch_reproduction_with_adjacency_frechet_side_info",
            "is_adapter": True,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 fixed-node batches lack STONE official road-coordinate metadata and official observed/unobserved OOD split.",
            "fallback": "Frechet side information from adjacency shortest-path anchor distances; learnable anchor-distance fallback only if adjacency is unavailable; all nodes default observed.",
            "notes": "Method-level STONE reproduction under BasicTS fixed-node PEMS08: spatial side information, Frechet anchor embedding, temporal semantic graph, spatial semantic graph, graph-editor perturbation branch, observed/unobserved mask interface, STBlock, STAggBlock, and GatedFusionBlock. This is not the official STONE OOD protocol.",
        },
        "DATASET": {"use_timestamps": False},
        "MODEL": {
            "backbone_name": "stone",
            "reference_status": "stone_method_level_pytorch_reproduction_with_adjacency_frechet_side_info",
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
                    "anchor_repeats": 4,
                    "side_info_seed": 2026,
                    "graph_prior_weight": 0.2,
                    "use_graph_perturbation": True,
                    "graph_perturb_samples": 3,
                    "graph_perturb_ratio": 0.2,
                    "lambda_graph_perturb": 0.0,
                },
            },
        },
        "LOSS": {"loss_type": "mae", "null_val": None, "use_backbone_aux": True, "lambda_backbone_aux": 1.0},
        "TRAIN": {"seed": 2026, "batch_size": 32, "val_batches": None, "ckpt_dir": "./checkpoints/pems08/baseline_stone_adapter"},
        "EVAL": {"horizon_metrics": True},
    },
)
apply_ablation(CONFIG, "no_env")
deep_update(CONFIG, {"MODEL": {"use_separated_z_for_y_inv": False}})


def get_config():
    return CONFIG
