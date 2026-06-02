from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from models.backbones._time_utils import d2stgnn_time_features
from models.backbones.base import BaseBackbone
from models.backbones.official_utils import ensure_repo_exists_or_skip


def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.clamp_min(0)
    denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / denom


def _load_official_d2stgnn(repo_root: str):
    """Load official D2STGNN without permanently replacing this repo's models package.

    Reference:
      /data/OuXiaoyu/mystg/baselines/D2STGNN/models/model.py
      /data/OuXiaoyu/mystg/baselines/D2STGNN/main.py
    """

    root = str(Path(repo_root).resolve())
    project_packages = {name: sys.modules.get(name) for name in ("models", "utils")}
    old_path = list(sys.path)
    for key in list(sys.modules):
        if key in {"models", "utils"} or key.startswith("models.") or key.startswith("utils."):
            # Keep already imported project modules alive through local globals;
            # official D2STGNN imports its own absolute `models.*` and `utils.*` modules.
            sys.modules.pop(key, None)
    sys.path.insert(0, root)
    try:
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [str(Path(root) / "models")]
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(Path(root) / "utils")]
        sys.modules["models"] = models_pkg
        sys.modules["utils"] = utils_pkg
        spec = importlib.util.spec_from_file_location("models.model", str(Path(root) / "models" / "model.py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official D2STGNN model.py from {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["models.model"] = module
        spec.loader.exec_module(module)
        return module.D2STGNN
    finally:
        for key in list(sys.modules):
            if key in {"models", "utils"} or key.startswith("models.") or key.startswith("utils."):
                sys.modules.pop(key, None)
        for name, module in project_packages.items():
            if module is not None:
                sys.modules[name] = module
        sys.path[:] = old_path


class D2STGNNBackbone(BaseBackbone):
    """Official-wrapper D2STGNN backbone adapted to [B,L,N,C] -> [B,H,N,C].

    The official code expects value + time-of-day + day-of-week channels and
    returns [B,N,H]. This wrapper builds those channels from local timestamp
    arrays while keeping local scaler/splits/metrics.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        repo_root: str = "",
        num_hidden: int = 32,
        node_hidden: int = 10,
        time_emb_dim: int = 10,
        dropout: float = 0.1,
        k_t: int = 3,
        k_s: int = 2,
        gap: int = 3,
        num_modalities: int = 2,
        num_time_in_day: int = 288,
        num_day_in_week: int = 7,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        if output_dim != 1:
            raise ValueError("Official D2STGNN wrapper currently supports output_dim=1.")
        if output_len % gap != 0:
            raise ValueError(f"D2STGNN requires output_len divisible by gap, got {output_len=} {gap=}.")
        repo_path = ensure_repo_exists_or_skip(
            "D2STGNN",
            ("D2STGNN", "GestaltCogTeam-D2STGNN"),
            explicit_path=repo_root or None,
        )
        self.repo_root = str(repo_path)
        official_cls = _load_official_d2stgnn(self.repo_root)
        self.num_time_in_day = int(num_time_in_day)
        self.num_day_in_week = int(num_day_in_week)
        self.model_args = {
            "num_feat": input_dim,
            "num_hidden": int(num_hidden),
            "node_hidden": int(node_hidden),
            "time_emb_dim": int(time_emb_dim),
            "dropout": float(dropout),
            "seq_length": int(output_len),
            "k_t": int(k_t),
            "k_s": int(k_s),
            "gap": int(gap),
            "num_modalities": int(num_modalities),
            "num_nodes": int(num_nodes),
            "adjs": [torch.eye(num_nodes), torch.eye(num_nodes)],
            "adjs_ori": torch.eye(num_nodes),
            "dataset": "PEMS08",
            "device": torch.device("cpu"),
        }
        self.official_model = official_cls(**self.model_args)
        self.representation_proj = nn.Sequential(
            nn.Linear(input_len * input_dim, max(representation_dim, num_hidden)),
            nn.GELU(),
            nn.Linear(max(representation_dim, num_hidden), representation_dim),
        )
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def _supports(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        if adj is None:
            base = torch.eye(self.num_nodes, device=device, dtype=dtype)
        else:
            base = adj[0] if adj.dim() == 3 else adj
            base = base.to(device=device, dtype=dtype)
        if base.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f"D2STGNN adj must be [N,N], got {tuple(base.shape)}")
        return [_row_normalize(base), _row_normalize(base.t())]

    def _sync_supports(self, adj: Optional[torch.Tensor], x: torch.Tensor) -> None:
        supports = self._supports(adj, x.device, x.dtype)
        args = self.official_model._model_args
        args["adjs"] = supports
        args["adjs_ori"] = supports[0]
        for layer in self.official_model.layers:
            layer.dif_layer.pre_defined_graph = supports
            layer.dif_layer.localized_st_conv.pre_defined_graph = layer.dif_layer.localized_st_conv.get_graph(supports)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        seq_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size, input_len, num_nodes, input_dim = x.shape
        self._sync_supports(adj, x)
        tod, dow = d2stgnn_time_features(
            seq_time,
            batch_size,
            input_len,
            num_nodes,
            x.device,
            x.dtype,
            num_time_in_day=self.num_time_in_day,
            num_day_in_week=self.num_day_in_week,
        )
        if tod.numel() and (float(tod.detach().min().cpu()) < -1e-6 or float(tod.detach().max().cpu()) > 1.0 + 1e-6):
            raise ValueError("D2STGNN official wrapper expects time_of_day normalized to [0,1]; official model multiplies by 288.")
        if dow.numel():
            dow_min = float(dow.detach().min().cpu())
            dow_max = float(dow.detach().max().cpu())
            if dow_min < -1e-6 or dow_max > self.num_day_in_week - 1 + 1e-6:
                raise ValueError("D2STGNN official wrapper expects day_of_week integer indices in [0,6].")
            if not torch.allclose(dow, dow.round(), atol=1e-5):
                raise ValueError("D2STGNN official wrapper expects day_of_week integer indices, not normalized floats.")
        history = torch.cat([x, tod.unsqueeze(-1), dow.unsqueeze(-1)], dim=-1)
        forecast = self.official_model(history)
        if forecast.dim() != 3:
            raise AssertionError(f"Official D2STGNN output must be [B,N,H], got {tuple(forecast.shape)}")
        y_inv = forecast.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        z_in = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, input_len * input_dim)
        z_inv = self.representation_proj(z_in)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
