import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update

CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {"method": "Ablation-persistence_no_gate_effect", "category": "ablation", "ablation": "persistence_no_gate_effect"},
        "LOSS": {"use_persistence_mi": True, "persistence_affects_gate": False},
        "MODEL": {"persistence": {"enabled": True}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_persistence_no_gate_effect"},
    },
)

def get_config():
    return CONFIG
