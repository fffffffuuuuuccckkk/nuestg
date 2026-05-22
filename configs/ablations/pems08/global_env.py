import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update

CONFIG = copy.deepcopy(BASE_CONFIG)
apply_ablation(CONFIG, "global_env")
deep_update(CONFIG, {"RUN": {"method": "Ablation-global_env", "category": "ablation", "ablation": "global_env"}, "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_global_env"}})

def get_config():
    return CONFIG
