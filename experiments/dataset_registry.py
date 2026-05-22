from __future__ import annotations

from typing import Dict


DATASET_REGISTRY: Dict[str, Dict] = {
    "pems08": {
        "name": "PEMS08",
        "default_config": "configs/pems08_nuestg.py",
        "input_len": 12,
        "output_len": 12,
        "num_nodes": 170,
        "setting": "traffic_forecasting",
        "notes": "Primary runnable dataset used by the bundled debug scripts.",
    },
    "metr_la": {
        "name": "METR-LA",
        "default_config": "configs/metr_la_nuestg.py",
        "input_len": 12,
        "output_len": 12,
        "setting": "traffic_forecasting",
        "notes": "Config scaffold is provided; runnability depends on local dataset availability.",
    },
}


def get_dataset(name: str) -> Dict:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key]
