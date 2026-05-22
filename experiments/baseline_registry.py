from __future__ import annotations

from typing import Dict, Iterable, List


def _train_command(config: str) -> str:
    return f"python train.py --config {config}"


def _external_entry(dataset: str, name: str, category: str, note: str) -> Dict:
    key = name.lower().replace("-", "").replace(" ", "_")
    config = f"configs/baselines/{dataset}/{key}_external.py"
    return {
        "name": name,
        "category": category,
        "status": "external_required",
        "config": config,
        "command": "import results via results/external_import_templates/*.csv",
        "paper_note": note,
        "expected_outputs": ["MAE", "RMSE", "MAPE"],
        "fairness_note": (
            "Use the same dataset split, input/output horizon, scaler, metrics, and seeds when importing external results. "
            "Do not claim this repository implements the method unless a runnable adapter is added."
        ),
    }


def _pems08_entries() -> List[Dict]:
    runnable = [
        (
            "STID-like MLP",
            "forecasting",
            "configs/baselines/pems08/stid_mlp.py",
            "Lightweight temporal MLP plus node embedding; not official STID.",
            "stid_mlp",
        ),
        (
            "GraphWaveNet-style",
            "forecasting",
            "configs/baselines/pems08/graphwavenet.py",
            "Graph WaveNet-style backbone from this repository; not official line-by-line reproduction.",
            "graphwavenet",
        ),
        (
            "AGCRN-style",
            "forecasting",
            "configs/baselines/pems08/agcrn.py",
            "AGCRN-style adaptive recurrent backbone from this repository; not official line-by-line reproduction.",
            "agcrn",
        ),
        (
            "Ours-STIDMLP",
            "plugin_ours",
            "configs/ours/pems08/nuestg_stid_mlp.py",
            "Full NUE-STG with the lightweight STID-like invariant backbone.",
            "stid_mlp",
        ),
        (
            "Ours-GraphWaveNet",
            "plugin_ours",
            "configs/ours/pems08/nuestg_graphwavenet.py",
            "Full NUE-STG with the Graph WaveNet-style invariant backbone.",
            "graphwavenet",
        ),
        (
            "Ours-AGCRN",
            "plugin_ours",
            "configs/ours/pems08/nuestg_agcrn.py",
            "Full NUE-STG with the AGCRN-style invariant backbone.",
            "agcrn",
        ),
    ]
    entries = [
        {
            "name": name,
            "category": category,
            "status": "runnable",
            "config": config,
            "command": _train_command(config),
            "paper_note": note,
            "expected_outputs": ["MAE", "RMSE", "MAPE"],
            "fairness_note": (
                "Runs through the same local train.py, BasicTS dataset split, input/output length, scaler, and metrics. "
                "For baseline-only configs, NUE-STG environment losses are disabled and prediction equals y_inv."
            ),
            "backbone": backbone,
        }
        for name, category, config, note, backbone in runnable
    ]
    external_forecasting = [
        ("DGCRN", "Requires official or independently verified implementation."),
        ("D2STGNN", "Requires official or independently verified implementation."),
        ("STAEformer", "Requires official or independently verified implementation."),
    ]
    external_ood = [
        ("CauSTG", "ST-OOD baseline requiring external implementation/results."),
        ("CaST", "ST-OOD baseline requiring external implementation/results."),
        ("STONE", "ST-OOD baseline requiring external implementation/results."),
        ("Samen", "Concept-shift pair-mining style baseline requiring external implementation/results."),
        ("CAN-ST", "ST-OOD baseline requiring external implementation/results."),
        ("STOP", "ST-OOD baseline requiring external implementation/results."),
        ("DIDA", "Optional ST-OOD baseline requiring external implementation/results."),
        ("I-DIDA", "Optional ST-OOD baseline requiring external implementation/results."),
        ("EAGLE", "Optional ST-OOD baseline requiring external implementation/results."),
    ]
    entries.extend(_external_entry("pems08", name, "forecasting", note) for name, note in external_forecasting)
    entries.extend(_external_entry("pems08", name, "st_ood", note) for name, note in external_ood)
    return entries


BASELINE_REGISTRY: Dict[str, List[Dict]] = {
    "pems08": _pems08_entries(),
}


def iter_baselines(dataset: str = "pems08", status: str | None = None, category: str | None = None) -> Iterable[Dict]:
    entries = BASELINE_REGISTRY.get(dataset.lower(), [])
    for entry in entries:
        if status is not None and entry["status"] != status:
            continue
        if category is not None and entry["category"] != category:
            continue
        yield entry


def get_baseline(name: str, dataset: str = "pems08") -> Dict:
    normalized = name.lower()
    for entry in BASELINE_REGISTRY.get(dataset.lower(), []):
        if entry["name"].lower() == normalized:
            return entry
    raise KeyError(f"Unknown baseline {name!r} for dataset {dataset!r}")
