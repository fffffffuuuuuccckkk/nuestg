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
    "newbike_chicago": {
        "name": "NewBike_Chicago",
        "default_config": "configs/newbike_chicago_nuestg.py",
        "input_len": 12,
        "output_len": 12,
        "num_nodes": 609,
        "setting": "st_ood_bike_demand",
        "notes": "Converted from ST-OOD NewBike_Chicago/2019 with official train/val/test/shift idx arrays.",
    },
    "taxi_chicago": {
        "name": "Taxi_Chicago",
        "default_config": "configs/taxi_chicago_nuestg.py",
        "input_len": 12,
        "output_len": 12,
        "num_nodes": 77,
        "setting": "st_ood_ride_hailing",
        "notes": "Converted from ST-OOD Taxi_Chicago/2013 with official train/val/test/shift idx arrays.",
    },
    "speed_nyc": {
        "name": "Speed_NYC",
        "default_config": "configs/speed_nyc_nuestg.py",
        "input_len": 12,
        "output_len": 12,
        "num_nodes": 139,
        "setting": "st_ood_traffic_speed",
        "notes": "Converted from ST-OOD Speed_NYC/2019 with official train/val/test/shift idx arrays.",
    },
}


def get_dataset(name: str) -> Dict:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key]
