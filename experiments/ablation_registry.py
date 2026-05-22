from __future__ import annotations

from typing import Dict, Iterable


def _cmd(name: str) -> str:
    if name == "full":
        return "python train.py --config configs/ours/pems08/nuestg_stid_mlp.py"
    return f"python train.py --config configs/ours/pems08/nuestg_stid_mlp.py --ablation {name}"


ABLATION_REGISTRY: Dict[str, Dict] = {
    "full": {
        "name": "full",
        "description": "Complete NUE-STG with node-wise env, gate, swap, separation, persistence MI, KL, ind, and sparse regularization.",
        "config_overrides": {},
        "expected_effect": "Reference method.",
        "command": _cmd("full"),
    },
    "no_env": {
        "name": "no_env",
        "description": "Invariant-only baseline: rho=0 and environment-related losses/mechanisms disabled.",
        "config_overrides": {"MODEL.force_gate_value": 0.0, "LOSS.use_gate": False, "LOSS.use_swap": False},
        "expected_effect": "Tests whether node-wise environment utility improves over invariant backbone alone.",
        "command": _cmd("no_env"),
    },
    "no_gate": {
        "name": "no_gate",
        "description": "Fix rho=1 and use all environment residuals without utility selection.",
        "config_overrides": {"MODEL.force_gate_value": 1.0, "LOSS.use_gate": False},
        "expected_effect": "Tests whether conditional utility selection is better than always using environment residuals.",
        "command": _cmd("no_gate"),
    },
    "no_swap": {
        "name": "no_swap",
        "description": "Disable random batch-node environment swapping.",
        "config_overrides": {"LOSS.use_swap": False, "SWAP.enabled": False},
        "expected_effect": "Tests the contribution of the first-version counterfactual swap regularizer.",
        "command": _cmd("no_swap"),
    },
    "no_persistence": {
        "name": "no_persistence",
        "description": "Disable FutureEnvEncoder, persistence InfoNCE, and persistence influence on gate labels.",
        "config_overrides": {"MODEL.persistence.enabled": False, "LOSS.use_persistence_mi": False},
        "expected_effect": "Tests whether historical-future environment persistence helps.",
        "command": _cmd("no_persistence"),
    },
    "persistence_no_gate_effect": {
        "name": "persistence_no_gate_effect",
        "description": "Keep persistence MI but do not multiply gate targets by s_persist.",
        "config_overrides": {"LOSS.persistence_affects_gate": False},
        "expected_effect": "Separates representation learning from gate-target shaping.",
        "command": "python train.py --config configs/ablations/pems08/persistence_no_gate_effect.py",
    },
    "no_separation": {
        "name": "no_separation",
        "description": "Disable computation-level Z/E separation.",
        "config_overrides": {"MODEL.separation.enabled": False},
        "expected_effect": "Tests whether hard computation-level separation helps beyond soft ind_loss.",
        "command": _cmd("no_separation"),
    },
    "separation_orthogonal": {
        "name": "separation_orthogonal",
        "description": "Use per-node orthogonal projection of z_raw against env direction.",
        "config_overrides": {"MODEL.separation.mode": "orthogonal_projection"},
        "expected_effect": "Tests directional hard removal of env-aligned components from z.",
        "command": "python train.py --config configs/ablations/pems08/separation_orthogonal.py",
    },
    "separation_basis_batch": {
        "name": "separation_basis_batch",
        "description": "Use batch env SVD basis projection.",
        "config_overrides": {"MODEL.separation.mode": "basis_projection", "MODEL.separation.basis.source": "batch_env"},
        "expected_effect": "Tests shared batch-level environment subspace removal.",
        "command": "python train.py --config configs/ablations/pems08/separation_basis.py",
    },
    "separation_basis_learnable": {
        "name": "separation_basis_learnable",
        "description": "Use learnable environment basis projection.",
        "config_overrides": {"MODEL.separation.mode": "basis_projection", "MODEL.separation.basis.source": "learnable"},
        "expected_effect": "Tests whether a learned environment subspace is useful.",
        "command": "python train.py --config configs/ablations/pems08/separation_basis_learnable.py",
    },
    "separation_lowrank": {
        "name": "separation_lowrank",
        "description": "Use hidden low-rank/residual decomposition.",
        "config_overrides": {"MODEL.separation.mode": "lowrank_residual", "MODEL.separation.lowrank.target": "hidden"},
        "expected_effect": "Tests stable low-rank z plus residual env injection.",
        "command": "python train.py --config configs/ablations/pems08/separation_lowrank.py",
    },
    "global_env": {
        "name": "global_env",
        "description": "Broadcast graph-level environment to all nodes.",
        "config_overrides": {"MODEL.env_global_mode": True},
        "expected_effect": "Tests whether node-wise environment is necessary.",
        "command": _cmd("global_env"),
    },
    "shuffled_env": {
        "name": "shuffled_env",
        "description": "Use randomly mismatched environments for main prediction and disable swap.",
        "config_overrides": {"MODEL.use_shuffled_env_train": True, "MODEL.use_shuffled_env_eval": True, "LOSS.use_swap": False},
        "expected_effect": "Tests whether environment-node correspondence matters.",
        "command": _cmd("shuffled_env"),
    },
    "no_kl": {
        "name": "no_kl",
        "description": "Disable environment KL bottleneck.",
        "config_overrides": {"LOSS.use_kl": False},
        "expected_effect": "Tests whether limiting I(E;X) matters.",
        "command": _cmd("no_kl"),
    },
    "no_ind": {
        "name": "no_ind",
        "description": "Disable Z/E cross-covariance independence penalty.",
        "config_overrides": {"LOSS.use_ind": False},
        "expected_effect": "Tests whether soft redundancy reduction matters.",
        "command": _cmd("no_ind"),
    },
    "no_sparse": {
        "name": "no_sparse",
        "description": "Disable rho sparsity penalty.",
        "config_overrides": {"LOSS.use_sparse": False},
        "expected_effect": "Tests whether the gate tends to over-open without sparsity.",
        "command": _cmd("no_sparse"),
    },
    "no_potential_loss": {
        "name": "no_potential_loss",
        "description": "Placeholder only; this repository has no separate potential prediction loss term.",
        "config_overrides": {},
        "expected_effect": "Not applicable until such a loss is implemented.",
        "command": "status=not_implemented",
        "status": "not_implemented",
    },
    "gate_gain_only": {
        "name": "gate_gain_only",
        "description": "Gate target uses potential gain only; persistence MI may still be trained.",
        "config_overrides": {"LOSS.persistence_affects_gate": False},
        "expected_effect": "Compares pure potential-gain gate labels with gain times persistence.",
        "command": "python train.py --config configs/ablations/pems08/gate_gain_only.py",
    },
    "gate_gain_x_persistence": {
        "name": "gate_gain_x_persistence",
        "description": "Gate target uses s_gain * s_persist.",
        "config_overrides": {"LOSS.persistence_affects_gate": True},
        "expected_effect": "Default persistence-aware utility gate.",
        "command": "python train.py --config configs/ablations/pems08/gate_gain_x_persistence.py",
    },
}


def iter_ablations() -> Iterable[Dict]:
    return ABLATION_REGISTRY.values()


def get_ablation(name: str) -> Dict:
    if name not in ABLATION_REGISTRY:
        raise KeyError(f"Unknown ablation {name!r}; expected one of {sorted(ABLATION_REGISTRY)}")
    return ABLATION_REGISTRY[name]
