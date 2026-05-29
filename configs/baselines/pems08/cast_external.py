CONFIG = {
    "BASELINE": {
        "name": "CaST",
        "dataset": "PEMS08",
        "status": "external_required",
        "reference_status": "skipped_external_missing",
        "category": "st_ood",
        "official_repo": "https://github.com/yutong-xia/CaST",
        "local_reference_path": "/data/OuXiaoyu/mystg/baselines/CaST",
        "referenced_files": [
            "src/models/cast.py",
            "src/layers/cast_cell.py",
            "experiments/cast/main.py",
            "README.md",
        ],
        "result_template": "results/external_import_templates/cast_pems08.csv",
        "notes": "Official code is present locally, but CaST requires graph/edge-feature dataset objects; no unified train.py wrapper is implemented yet.",
    }
}
