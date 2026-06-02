from __future__ import annotations

from typing import Dict, Iterable, List


LOCAL_REPO_PATHS = {
    "STID": "/data/OuXiaoyu/mystg/baselines/STID",
    "STID-like MLP": "/data/OuXiaoyu/mystg/baselines/STID",
    "Graph WaveNet": "/data/OuXiaoyu/mystg/baselines/Graph-WaveNet",
    "AGCRN": "/data/OuXiaoyu/mystg/baselines/AGCRN",
    "STGCN": "/data/OuXiaoyu/mystg/baselines/stgcn",
    "ST-Norm": "/data/OuXiaoyu/mystg/baselines/ST-Norm",
    "D2STGNN": "/data/OuXiaoyu/mystg/baselines/D2STGNN",
    "CaST-faithful-pytorch-adapter": "/data/OuXiaoyu/mystg/baselines/CaST",
    "CaST-official": "/data/OuXiaoyu/mystg/baselines/CaST",
    "STONE-faithful-pytorch-adapter": "/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024",
    "STONE-official": "/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024",
    "STOP-faithful-architecture-adapter": "/data/OuXiaoyu/mystg/baselines/STOP",
    "STOP-official": "/data/OuXiaoyu/mystg/baselines/STOP",
}


def _train_command(config: str) -> str:
    return f"python train.py --config {config}"


def _external_entry(dataset: str, name: str, category: str, note: str) -> Dict:
    key = name.lower().replace("-", "").replace(" ", "_")
    config = f"configs/baselines/{dataset}/{key}_external.py"
    external_refs = {
        "D2STGNN": (
            "https://github.com/GestaltCogTeam/D2STGNN",
            [
                "models/model.py",
                "models/diffusion_block/dif_block.py",
                "models/inherent_block/inh_block.py",
                "models/dynamic_graph_conv/dy_graph_conv.py",
                "models/decouple/estimation_gate.py",
                "configs/PEMS08.yaml",
                "main.py",
            ],
        ),
        "CaST": (
            "https://github.com/yutong-xia/CaST",
            ["src/models/cast.py", "src/layers/cast_cell.py", "experiments/cast/main.py"],
        ),
        "STONE": (
            "https://github.com/PoorOtterBob/STONE-KDD-2024",
            ["Knowair/frechet.py", "Knowair/graph.py", "Knowair/spatial_side_information.py"],
        ),
        "STOP": (
            "https://github.com/PoorOtterBob/STOP",
            ["LargeST/src/models/stop.py", "TrafficStream/src/models/stop.py", "KnowAir/src/models/stop.py"],
        ),
    }
    official_repo, referenced_files = external_refs.get(name, ("", []))
    return {
        "name": name,
        "category": category,
        "status": "external_required",
        "config": config,
        "command": "import results via results/external_import_templates/*.csv",
        "paper_note": note,
        "expected_outputs": ["MAE", "RMSE", "MAPE"],
        "reference_status": "skipped_local_repo_missing",
        "implementation_type": "external_required",
        "official_repo": official_repo,
        "referenced_files": referenced_files,
        "architecture_check": {
            "expected": note,
            "implemented": "No in-repository train.py implementation.",
            "missing_or_different": "Import external results with source notes, or add an official wrapper later.",
        },
        "fairness_note": (
            "Use the same dataset split, input/output horizon, scaler, metrics, and seeds when importing external results. "
            "Do not claim this repository implements the method unless a runnable adapter is added."
        ),
    }


def _pems08_entries() -> List[Dict]:
    runnable = [
        (
            "STID",
            "forecasting",
            "configs/baselines/pems08/stid.py",
            "Faithful native STID adapted from official BasicTS/STID architecture.",
            "stid",
            "faithful_native_adapter",
            "https://github.com/GestaltCogTeam/STID",
            [
                "stid/arch/stid_arch.py::STID",
                "stid/arch/mlp.py::MultiLayerPerceptron",
                "stid/PEMS08.py::MODEL_PARAM",
            ],
        ),
        (
            "STID-like MLP",
            "forecasting",
            "configs/baselines/pems08/stid_mlp.py",
            "Lightweight temporal MLP plus node embedding; simplified debug baseline, not official STID.",
            "stid_mlp",
            "style_native",
            "https://github.com/GestaltCogTeam/STID",
            ["stid/arch/stid_arch.py", "stid/arch/mlp.py"],
        ),
        (
            "Graph WaveNet",
            "forecasting",
            "configs/baselines/pems08/graphwavenet.py",
            "Faithful native Graph WaveNet adapted from official model.py/util.py.",
            "graphwavenet",
            "graphwavenet_native_adapter",
            "https://github.com/nnzhan/Graph-WaveNet",
            ["model.py::gwnet/gcn/nconv", "util.py::load_adj", "train.py"],
        ),
        (
            "AGCRN",
            "forecasting",
            "configs/baselines/pems08/agcrn.py",
            "Faithful native AGCRN adapted from official AGCRN.py/AGCN.py/AGCRNCell.py.",
            "agcrn",
            "faithful_native_adapter",
            "https://github.com/LeiBAI/AGCRN",
            ["model/AGCRN.py::AGCRN/AVWDCRNN", "model/AGCN.py::AVWGCN", "model/AGCRNCell.py::AGCRNCell"],
        ),
        (
            "STGCN",
            "forecasting",
            "configs/baselines/pems08/stgcn.py",
            "Faithful native STGCN adapted from hazdzz/STGCN.",
            "stgcn",
            "reference_native",
            "https://github.com/hazdzz/STGCN",
            ["model/models.py::STGCNChebGraphConv", "model/layers.py::STConvBlock/TemporalConvLayer/ChebGraphConv"],
        ),
        (
            "ST-Norm",
            "forecasting",
            "configs/baselines/pems08/stnorm.py",
            "Faithful native ST-Norm WaveNet with internal spatial/temporal normalization.",
            "stnorm",
            "stnorm_wavenet_adapter",
            "https://github.com/JLDeng/ST-Norm",
            ["models/Wavenet.py::SNorm/TNorm/Wavenet", "main.py"],
        ),
        (
            "D2STGNN",
            "forecasting",
            "configs/baselines/pems08/d2stgnn.py",
            "Official D2STGNN model wrapper with local scaler/splits/metrics.",
            "d2stgnn",
            "official_local_wrapper",
            "https://github.com/GestaltCogTeam/D2STGNN",
            [
                "models/model.py::D2STGNN/DecoupleLayer",
                "models/diffusion_block/dif_block.py::DifBlock",
                "models/inherent_block/inh_block.py::InhBlock",
                "models/dynamic_graph_conv/dy_graph_conv.py::DynamicGraphConstructor",
                "configs/PEMS08.yaml",
                "main.py",
            ],
        ),
        (
            "CaST-faithful-pytorch-adapter",
            "st_ood",
            "configs/baselines/pems08/cast.py",
            "Faithful PyTorch CaST adapter with official temporal/entity/environment, Hodge edge, causal GCN, and VQ/commitment/MI losses; fixed-node because PyG PairData is unavailable in the local BasicTS batch.",
            "cast",
            "cast_faithful_pytorch_adapter_with_official_aux_loss",
            "https://github.com/yutong-xia/CaST",
            ["src/models/cast.py", "src/layers/cast_cell.py", "src/trainers/cast_trainer.py", "src/utils/dataset.py", "experiments/cast/main.py"],
        ),
        (
            "STONE-faithful-pytorch-adapter",
            "st_ood",
            "configs/baselines/pems08/stone.py",
            "Faithful STONE architecture adapter with official STBlock, STAggBlock, and GatedFusionBlock; PEMS semantic side information falls back to learnable node embeddings.",
            "stone",
            "stone_faithful_pytorch_adapter_without_spatial_side_info",
            "https://github.com/PoorOtterBob/STONE-KDD-2024",
            ["src/base/stone.py", "Knowair/model/STONE.py", "src/utils/spatial_side_information.py", "experiments/stone/main.py"],
        ),
        (
            "STOP-faithful-architecture-adapter",
            "st_ood",
            "configs/baselines/pems08/stop.py",
            "Faithful STOP LargeST architecture adapter with MLP decomposition/prompt encoder and Core_Adaptive backcast; local scaler/splits/metrics.",
            "stop",
            "stop_faithful_architecture_adapter_without_sood_protocol",
            "https://github.com/PoorOtterBob/STOP",
            ["LargeST/src/models/stop.py::STOP/MLP/Core_Adaptive", "LargeST/src/engines/stop_engine.py", "LargeST/experiments/stop/main.py"],
        ),
        (
            "Ours-STIDMLP",
            "plugin_ours",
            "configs/ours/pems08/nuestg_stid_mlp.py",
            "Full NUE-STG with the lightweight STID-like invariant backbone.",
            "stid_mlp",
            "plugin_ours",
            "",
            [],
        ),
        (
            "Ours-GraphWaveNet",
            "plugin_ours",
            "configs/ours/pems08/nuestg_graphwavenet.py",
            "Full NUE-STG with the Graph WaveNet-style invariant backbone.",
            "graphwavenet",
            "plugin_ours",
            "",
            [],
        ),
        (
            "Ours-AGCRN",
            "plugin_ours",
            "configs/ours/pems08/nuestg_agcrn.py",
            "Full NUE-STG with the AGCRN-style invariant backbone.",
            "agcrn",
            "plugin_ours",
            "",
            [],
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
            "reference_status": reference_status,
            "implementation_type": reference_status,
            "official_repo": official_repo,
            "local_repo_path": LOCAL_REPO_PATHS.get(name, ""),
            "referenced_files": referenced_files,
            "display_name": name,
            "is_adapter": reference_status
            in {
                "cast_fixed_node_simplified_adapter",
                "stone_fixed_node_simplified_adapter",
                "stop_architecture_adapter_without_sood_protocol",
                "cast_faithful_pytorch_adapter_with_official_aux_loss",
                "stone_faithful_pytorch_adapter_without_spatial_side_info",
                "stop_faithful_architecture_adapter_without_sood_protocol",
            },
            "is_official": reference_status == "official_local_wrapper",
            "main_table_safe": reference_status
            in {
                "reference_native",
                "graphwavenet_native_adapter",
                "faithful_native_adapter",
                "stnorm_wavenet_adapter",
                "official_local_wrapper",
            }
            and name not in {"CaST-faithful-pytorch-adapter", "STONE-faithful-pytorch-adapter", "STOP-faithful-architecture-adapter"},
            "architecture_check": {
                "expected": note,
                "implemented": f"Backbone={backbone}, shared local scaler/splits/metrics.",
                "missing_or_different": (
                    "Adapted to BaseBackbone input [B,L,N,C] and output [B,H,N,C]; "
                    "baseline-only configs disable NUE-STG environment modules."
                ),
            },
            "fairness_note": (
                "Runs through the same local train.py, BasicTS dataset split, input/output length, scaler, and metrics. "
                "For baseline-only configs, NUE-STG environment losses are disabled and prediction equals y_inv."
            ),
            "backbone": backbone,
        }
        for name, category, config, note, backbone, reference_status, official_repo, referenced_files in runnable
    ]
    official_checks = [
        {
            "name": "CaST-official",
            "display_name": "CaST-official",
            "category": "st_ood",
            "status": "official_check",
            "config": "configs/baselines/pems08/cast_official.py",
            "command": _train_command("configs/baselines/pems08/cast_official.py"),
            "paper_note": "Full official CaST check; skips on PEMS08 because PyG graph Data, Hodge edge graph, preprocessing, and VQ/MI loss protocol are not available in the unified fixed-node batch.",
            "expected_outputs": ["SKIP or official metrics"],
            "reference_status": "official_local_wrapper",
            "implementation_type": "official_local_wrapper",
            "official_repo": "https://github.com/yutong-xia/CaST",
            "local_repo_path": LOCAL_REPO_PATHS["CaST-official"],
            "referenced_files": ["src/models/cast.py", "src/layers/cast_cell.py", "src/utils/dataset.py", "src/trainers/cast_trainer.py"],
            "is_adapter": False,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 BasicTS fixed-node config cannot provide the full official CaST graph/loss/preprocessing protocol.",
            "backbone": "cast_official",
        },
        {
            "name": "STONE-official",
            "display_name": "STONE-official",
            "category": "st_ood",
            "status": "official_check",
            "config": "configs/baselines/pems08/stone_official.py",
            "command": _train_command("configs/baselines/pems08/stone_official.py"),
            "paper_note": "Full official STONE check; skips on PEMS08 because spatial side information and structural-shift metadata are required.",
            "expected_outputs": ["SKIP or official metrics"],
            "reference_status": "official_local_wrapper",
            "implementation_type": "official_local_wrapper",
            "official_repo": "https://github.com/PoorOtterBob/STONE-KDD-2024",
            "local_repo_path": LOCAL_REPO_PATHS["STONE-official"],
            "referenced_files": ["Knowair/model/STONE.py", "Knowair/frechet.py", "Knowair/spatial_side_information.py", "src/base/stone_engine.py"],
            "is_adapter": False,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 fixed-node config lacks STONE coordinate/meta/spatial-shift side information.",
            "backbone": "stone_official",
        },
        {
            "name": "STOP-official",
            "display_name": "STOP-official",
            "category": "st_ood",
            "status": "official_check",
            "config": "configs/baselines/pems08/stop_official.py",
            "command": _train_command("configs/baselines/pems08/stop_official.py"),
            "paper_note": "Full official STOP check; skips on PEMS08 because the SOOD/LargeST/KnowAir/TrafficStream protocol is required.",
            "expected_outputs": ["SKIP or official metrics"],
            "reference_status": "official_local_wrapper",
            "implementation_type": "official_local_wrapper",
            "official_repo": "https://github.com/PoorOtterBob/STOP",
            "local_repo_path": LOCAL_REPO_PATHS["STOP-official"],
            "referenced_files": ["LargeST/src/models/stop.py", "LargeST/src/engines/stop_engine.py", "KnowAir/src/models/stop.py", "TrafficStream/src/models/stop.py"],
            "is_adapter": False,
            "is_official": False,
            "main_table_safe": False,
            "unsupported_reason": "PEMS08 fixed-node config lacks the official STOP SOOD/OOD protocol.",
            "backbone": "stop_official",
        },
    ]
    entries.extend(official_checks)
    external_forecasting = [
        ("DGCRN", "Requires official or independently verified implementation."),
        ("STAEformer", "Requires official or independently verified implementation."),
    ]
    external_ood = [
        ("CauSTG", "ST-OOD baseline requiring external implementation/results."),
        ("Samen", "Concept-shift pair-mining style baseline requiring external implementation/results."),
        ("CAN-ST", "ST-OOD baseline requiring external implementation/results."),
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
