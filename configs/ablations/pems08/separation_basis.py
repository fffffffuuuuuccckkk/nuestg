import copy

from configs.ours.pems08.nuestg_stid_mlp import CONFIG as BASE_CONFIG
from utils.config_utils import deep_update

CONFIG = deep_update(
    copy.deepcopy(BASE_CONFIG),
    {
        "RUN": {"method": "Ablation-separation_basis_batch", "category": "ablation", "ablation": "separation_basis_batch"},
        "MODEL": {"separation": {"enabled": True, "mode": "basis_projection", "basis": {"source": "batch_env"}}},
        "TRAIN": {"ckpt_dir": "./checkpoints/pems08/ablation_separation_basis_batch"},
    },
)

def get_config():
    return CONFIG
