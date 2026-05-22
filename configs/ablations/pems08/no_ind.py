import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import apply_ablation, deep_update

CONFIG = copy.deepcopy(BASE_CONFIG)
apply_ablation(CONFIG, "no_ind")
deep_update(CONFIG, {"RUN": {"method": "Ablation-no_ind", "category": "ablation", "ablation": "no_ind"}, "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_no_ind"}})

def get_config():
    return CONFIG
