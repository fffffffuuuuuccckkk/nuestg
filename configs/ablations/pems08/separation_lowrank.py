import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update

CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {"method": "Ablation-separation_lowrank", "category": "ablation", "ablation": "separation_lowrank"},
        "MODEL": {"separation": {"enabled": True, "mode": "lowrank_residual", "lowrank": {"target": "hidden"}}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_separation_lowrank"},
    },
)

def get_config():
    return CONFIG
