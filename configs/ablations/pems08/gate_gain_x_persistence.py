import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update

CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {"method": "Ablation-gate_gain_x_persistence", "category": "ablation", "ablation": "gate_gain_x_persistence"},
        "LOSS": {"use_persistence_mi": True, "persistence_affects_gate": True},
        "MODEL": {"persistence": {"enabled": True}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_gate_gain_x_persistence"},
    },
)

def get_config():
    return CONFIG
