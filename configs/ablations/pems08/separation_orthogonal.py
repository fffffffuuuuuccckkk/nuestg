import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update

CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {"method": "Ablation-separation_orthogonal", "category": "ablation", "ablation": "separation_orthogonal"},
        "MODEL": {"separation": {"enabled": True, "mode": "orthogonal_projection"}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_separation_orthogonal"},
    },
)

def get_config():
    return CONFIG
