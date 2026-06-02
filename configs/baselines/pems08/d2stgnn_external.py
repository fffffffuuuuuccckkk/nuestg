CONFIG = {
    "BASELINE": {
        "name": "D2STGNN",
        "dataset": "PEMS08",
        "status": "external_required",
        "reference_status": "skipped_local_repo_missing",
        "category": "forecasting",
        "official_repo": "https://github.com/GestaltCogTeam/D2STGNN",
        "local_reference_path": "/data/OuXiaoyu/mystg/baselines/D2STGNN",
        "referenced_files": [
            "models/model.py",
            "models/diffusion_block/dif_block.py",
            "models/inherent_block/inh_block.py",
            "models/dynamic_graph_conv/dy_graph_conv.py",
            "models/decouple/estimation_gate.py",
            "configs/PEMS08.yaml",
            "main.py",
        ],
        "result_template": "results/external_import_templates/d2stgnn_pems08.csv",
        "notes": "Official code is present locally, but no unified train.py wrapper is implemented yet.",
    }
}
