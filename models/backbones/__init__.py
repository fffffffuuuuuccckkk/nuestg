from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from models.backbones.agcrn import AGCRNBackbone
from models.backbones.base import BaseBackbone
from models.backbones.cast import CaSTBackbone
from models.backbones.d2stgnn import D2STGNNBackbone
from models.backbones.graph_wavenet import GraphWaveNetBackbone
from models.backbones.stgcn import STGCNBackbone
from models.backbones.stid import STIDBackbone
from models.backbones.stid_mlp import STIDMLPBackbone
from models.backbones.stnorm_wavenet import STNormWaveNetBackbone
from models.backbones.stone import STONEBackbone
from models.backbones.stop import STOPBackbone


def _to_model_dict(cfg: Any) -> Dict:
    if isinstance(cfg, dict):
        return cfg["MODEL"] if "MODEL" in cfg else cfg
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))


def build_backbone(cfg: Any) -> BaseBackbone:
    model_cfg = _to_model_dict(cfg)
    backbone_cfg = dict(model_cfg.get("backbone", {}) or {})
    name = str(model_cfg.get("backbone_name") or backbone_cfg.get("name") or "stid_mlp").lower()
    representation_dim = int(backbone_cfg.get("representation_dim", model_cfg.get("hidden_dim", 64)))
    common = {
        "input_len": int(model_cfg["input_len"]),
        "output_len": int(model_cfg["output_len"]),
        "num_nodes": int(model_cfg["num_nodes"]),
        "input_dim": int(model_cfg["input_dim"]),
        "output_dim": int(model_cfg["output_dim"]),
        "representation_dim": representation_dim,
    }

    if name in {"stid_mlp", "mlp", "stid_like"}:
        stid_cfg = dict(backbone_cfg.get("stid_mlp", {}) or {})
        if "representation_dim" in stid_cfg:
            common["representation_dim"] = int(stid_cfg.pop("representation_dim"))
        return STIDMLPBackbone(
            **common,
            hidden_dim=int(stid_cfg.get("hidden_dim", model_cfg.get("hidden_dim", representation_dim))),
            node_emb_dim=int(stid_cfg.get("node_emb_dim", model_cfg.get("node_emb_dim", 32))),
            dropout=float(stid_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
            use_node_embedding=bool(stid_cfg.get("use_node_embedding", model_cfg.get("use_node_embedding", True))),
        )

    if name in {"stid", "official_stid"}:
        stid_cfg = dict(backbone_cfg.get("stid", {}) or {})
        if "representation_dim" in stid_cfg:
            common["representation_dim"] = int(stid_cfg.pop("representation_dim"))
        return STIDBackbone(**common, **stid_cfg)

    if name in {"graphwavenet", "gwnet", "graph_wavenet"}:
        gw_cfg = dict(backbone_cfg.get("graph_wavenet", {}) or {})
        if "representation_dim" in gw_cfg:
            common["representation_dim"] = int(gw_cfg.pop("representation_dim"))
        return GraphWaveNetBackbone(**common, **gw_cfg)

    if name == "agcrn":
        agcrn_cfg = dict(backbone_cfg.get("agcrn", {}) or {})
        if "representation_dim" in agcrn_cfg:
            common["representation_dim"] = int(agcrn_cfg.pop("representation_dim"))
        return AGCRNBackbone(**common, **agcrn_cfg)

    if name == "stgcn":
        stgcn_cfg = dict(backbone_cfg.get("stgcn", {}) or {})
        if "representation_dim" in stgcn_cfg:
            common["representation_dim"] = int(stgcn_cfg.pop("representation_dim"))
        return STGCNBackbone(**common, **stgcn_cfg)

    if name in {"stnorm", "st_norm", "stnorm_wavenet"}:
        stnorm_cfg = dict(backbone_cfg.get("stnorm_wavenet", {}) or {})
        if "representation_dim" in stnorm_cfg:
            common["representation_dim"] = int(stnorm_cfg.pop("representation_dim"))
        return STNormWaveNetBackbone(**common, **stnorm_cfg)

    if name == "d2stgnn":
        d2_cfg = dict(backbone_cfg.get("d2stgnn", {}) or {})
        if "representation_dim" in d2_cfg:
            common["representation_dim"] = int(d2_cfg.pop("representation_dim"))
        return D2STGNNBackbone(**common, **d2_cfg)

    if name == "cast":
        cast_cfg = dict(backbone_cfg.get("cast", {}) or {})
        if "representation_dim" in cast_cfg:
            common["representation_dim"] = int(cast_cfg.pop("representation_dim"))
        return CaSTBackbone(**common, **cast_cfg)

    if name == "stone":
        stone_cfg = dict(backbone_cfg.get("stone", {}) or {})
        if "representation_dim" in stone_cfg:
            common["representation_dim"] = int(stone_cfg.pop("representation_dim"))
        return STONEBackbone(**common, **stone_cfg)

    if name == "stop":
        stop_cfg = dict(backbone_cfg.get("stop", {}) or {})
        if "representation_dim" in stop_cfg:
            common["representation_dim"] = int(stop_cfg.pop("representation_dim"))
        return STOPBackbone(**common, **stop_cfg)

    raise ValueError(
        f"Unsupported MODEL.backbone_name={name!r}; "
        "expected one of stid, stid_mlp, graphwavenet, graph_wavenet, gwnet, agcrn, "
        "stgcn, stnorm, d2stgnn, cast, stone, stop"
    )


__all__ = [
    "AGCRNBackbone",
    "BaseBackbone",
    "CaSTBackbone",
    "D2STGNNBackbone",
    "GraphWaveNetBackbone",
    "STGCNBackbone",
    "STIDBackbone",
    "STIDMLPBackbone",
    "STNormWaveNetBackbone",
    "STONEBackbone",
    "STOPBackbone",
    "build_backbone",
]
