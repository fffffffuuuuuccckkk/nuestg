CONFIG = {
    "BASELINE": {
        "name": "STONE",
        "dataset": "PEMS08",
        "status": "external_required",
        "reference_status": "skipped_external_missing",
        "category": "st_ood",
        "official_repo": "https://github.com/PoorOtterBob/STONE-KDD-2024",
        "local_reference_path": "/data/OuXiaoyu/mystg/baselines/STONE-KDD-2024",
        "referenced_files": [
            "README.md",
            "Knowair/frechet.py",
            "Knowair/graph.py",
            "Knowair/spatial_side_information.py",
            "Knowair/train.py",
        ],
        "result_template": "results/external_import_templates/stone_pems08.csv",
        "notes": "Official code is present locally, but STONE requires spatial/structural-shift side information such as coordinates/meta graphs; no unified train.py wrapper is implemented yet.",
    }
}
