from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


class STONEBackbone(BaseBackbone):
    """Fixed-node STONE native adapter.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/STONE-KDD-2024/src/base/stone.py
      - /data/OuXiaoyu/mystg/baselines/STONE-KDD-2024/Knowair/model/STONE.py
      - /data/OuXiaoyu/mystg/baselines/STONE-KDD-2024/src/utils/spatial_side_information.py

    Full STONE uses spatial/structural-shift side information and Fréchet
    embeddings. For fixed-node PEMS, this adapter keeps the STONE style:
    temporal gated convolutions, semantic node stream, adaptive interaction,
    graph aggregation, and gated fusion. Semantic features fall back to
    learnable node embeddings because PEMS lacks the official coordinates/meta.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        temporal_channels: int = 128,
        sem_dim: int = 64,
        hidden_dim: int = 64,
        gate_dim: int = 128,
        Kt: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.Kt = int(Kt)
        self.temporal_channels = int(temporal_channels)
        self.input_proj = nn.Conv2d(input_dim, temporal_channels, kernel_size=(1, 1))
        self.temporal_filter = nn.Conv2d(temporal_channels, temporal_channels, kernel_size=(Kt, 1))
        self.temporal_gate = nn.Conv2d(temporal_channels, temporal_channels, kernel_size=(Kt, 1))
        reduced_len = input_len - Kt + 1
        if reduced_len < 1:
            raise ValueError("STONE adapter requires input_len >= Kt")
        self.temporal_reduce = nn.Linear(reduced_len, hidden_dim)
        self.semantic = nn.Parameter(torch.randn(num_nodes, sem_dim))
        self.sem_proj = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.x_adj_out = nn.Linear(hidden_dim, hidden_dim)
        self.x_adj_in = nn.Linear(hidden_dim, hidden_dim)
        self.sem_adj_out = nn.Linear(hidden_dim, hidden_dim)
        self.sem_adj_in = nn.Linear(hidden_dim, hidden_dim)
        self.x_graph = nn.Linear(hidden_dim, hidden_dim)
        self.sem_graph = nn.Linear(hidden_dim, hidden_dim)
        self.gate_sem = nn.Linear(hidden_dim * 2, gate_dim)
        self.gate_x = nn.Linear(hidden_dim, gate_dim)
        self.out1 = nn.Linear(gate_dim, gate_dim)
        self.out2 = nn.Linear(gate_dim, output_len * output_dim)
        self.dropout = nn.Dropout(dropout)
        self.representation_proj = nn.Linear(gate_dim, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    @staticmethod
    def _adaptive_adj(out_layer: nn.Linear, in_layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        out = out_layer(x)
        inn = in_layer(x)
        adj = torch.einsum("bid,bjd->bij", out, inn) / max(1.0, float(out.shape[-1]) ** 0.5)
        return torch.softmax(F.relu(adj), dim=-1)

    def _static_adj(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if adj is None:
            base = torch.eye(self.num_nodes, device=device, dtype=dtype)
        else:
            base = adj[0] if adj.dim() == 3 else adj
            base = base.to(device=device, dtype=dtype).clamp_min(0)
        base = base + torch.eye(self.num_nodes, device=device, dtype=dtype)
        return base / base.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        h = self.input_proj(x.permute(0, 3, 1, 2))
        filt = self.temporal_filter(h)
        gate = torch.sigmoid(self.temporal_gate(h))
        h = filt * gate
        h = self.temporal_reduce(h.permute(0, 3, 1, 2)).mean(dim=2)
        sem = self.sem_proj(self.semantic).unsqueeze(0).expand(batch_size, -1, -1)
        x_adj = self._adaptive_adj(self.x_adj_out, self.x_adj_in, sem)
        sem_adj = self._adaptive_adj(self.sem_adj_out, self.sem_adj_in, h)
        static_adj = self._static_adj(adj, x.device, x.dtype).unsqueeze(0)
        h = self.x_graph(torch.einsum("bij,bjd->bid", 0.5 * x_adj + 0.5 * static_adj, h))
        sem = self.sem_graph(torch.einsum("bij,bjd->bid", sem_adj, sem))
        h = self.dropout(F.relu(h))
        sem = self.dropout(F.relu(sem))
        gate_sem = torch.sigmoid(self.gate_sem(torch.cat([sem, h], dim=-1)))
        hidden = gate_sem * self.gate_x(h)
        hidden = F.relu(self.out1(hidden))
        pred = self.out2(hidden)
        y_inv = pred.view(batch_size, self.num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        z_inv = self.representation_proj(hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
