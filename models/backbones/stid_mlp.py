from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from models.backbones.base import BaseBackbone


class STIDMLPBackbone(BaseBackbone):
    """Lightweight STID-like MLP backbone used as the default invariant branch.

    This is the same temporal-MLP plus optional node embedding design previously
    embedded directly in NUESTG. It is not the full official STID implementation.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        hidden_dim: int = 64,
        node_emb_dim: int = 32,
        dropout: float = 0.1,
        use_node_embedding: bool = True,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if use_node_embedding:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
            projector_in_dim = hidden_dim + node_emb_dim
        else:
            self.node_emb = None
            projector_in_dim = hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(projector_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self._check_input(x)
        batch_size, input_len, num_nodes, input_dim = x.shape
        node_history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, input_len * input_dim)
        h = self.temporal_encoder(node_history)
        if self.node_emb is not None:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            h = torch.cat([h, node_emb], dim=-1)
        z_inv = self.projector(h)
        y_inv = self.forecast_from_representation(z_inv)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
