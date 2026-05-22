import copy
from configs.metr_la_nuestg import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update
CONFIG = deep_update(copy.deepcopy(BASE_CONFIG), {"RUN": {"method": "Ours-GraphWaveNet", "category": "plugin_ours", "setting": "forecasting", "status": "runnable"}, "MODEL": {"backbone_name": "graphwavenet", "backbone": {"name": "graphwavenet"}}, "TRAIN": {"ckpt_dir": "./checkpoints/metr_la/ours_graphwavenet"}})
def get_config():
    return CONFIG
