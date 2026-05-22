import copy
from configs.metr_la_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update
CONFIG = deep_update(copy.deepcopy(BASE_CONFIG), {"RUN": {"method": "Ours-STIDMLP", "category": "plugin_ours", "setting": "forecasting", "status": "runnable"}, "MODEL": {"backbone_name": "stid_mlp", "backbone": {"name": "stid_mlp"}}, "TRAIN": {"ckpt_dir": "./checkpoints/metr_la/ours_stid_mlp"}})
def get_config():
    return CONFIG
