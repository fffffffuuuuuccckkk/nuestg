from pathlib import Path


MYSTG_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = MYSTG_ROOT / "datasets" / "PEMS08"


CONFIG = {
    "DATASET": {
        "name": "PEMS08",
        "data_file_path": str(DATASET_DIR),
        "input_len": 12,
        "output_len": 12,
        "use_timestamps": False,
        "memmap": True,
        "null_val": 0.0,
    },
    "MODEL": {
        "input_len": 12,
        "output_len": 12,
        "num_nodes": 170,
        "input_dim": 1,
        "output_dim": 1,
        "hidden_dim": 64,
        "env_dim": 32,
        "node_emb_dim": 32,
        "dropout": 0.1,
        "use_adj": True,
        "adj_path": str(DATASET_DIR / "adj_mx.pkl"),
        "deterministic_env_eval": True,
        "use_node_embedding": True,
        "enable_swap": True,
    },
    "LOSS": {
        "lambda_inv": 0.2,
        "lambda_gate": 0.1,
        "lambda_swap": 0.1,
        "lambda_swap_same": 0.05,
        "lambda_kl": 1e-4,
        "lambda_ind": 1e-3,
        "lambda_sparse": 1e-3,
        "gate_eta": 0.0,
        "gate_tau": 0.1,
        "swap_margin": 0.01,
        "null_val": 0.0,
    },
    "TRAIN": {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 20,
        "device": "cuda:0",
        "seed": 42,
        "num_workers": 0,
        "log_interval": 20,
        "val_interval": 1,
        "val_batches": 20,
        "grad_clip": 5.0,
        "ckpt_dir": str(MYSTG_ROOT / "outputs" / "pems08_nuestg"),
    },
}


def get_config():
    return CONFIG
