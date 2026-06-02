CONFIG = {
    "BASELINE": {
        "name": "STOP",
        "dataset": "PEMS08",
        "status": "external_required",
        "reference_status": "skipped_local_repo_missing",
        "category": "st_ood",
        "official_repo": "https://github.com/PoorOtterBob/STOP",
        "local_reference_path": "/data/OuXiaoyu/mystg/baselines/STOP",
        "referenced_files": [
            "README.md",
            "LargeST/src/models/stop.py",
            "LargeST/src/engines/stop_engine.py",
            "TrafficStream/src/models/stop.py",
            "KnowAir/src/models/stop.py",
        ],
        "result_template": "results/external_import_templates/stop_pems08.csv",
        "notes": "Official code is present locally, but STOP depends on special OOD dataset objects; no unified train.py wrapper is implemented yet.",
    }
}
