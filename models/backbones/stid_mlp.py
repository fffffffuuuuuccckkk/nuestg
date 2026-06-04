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
        use_time_of_day_embedding: bool = False,
        use_day_of_week_embedding: bool = False,
        tod_emb_dim: int = 16,
        dow_emb_dim: int = 8,
        num_time_in_day: int = 288,
        num_day_in_week: int = 7,
        require_time_features: bool = False,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.use_node_embedding = bool(use_node_embedding)
        self.use_time_of_day_embedding = bool(use_time_of_day_embedding)
        self.use_day_of_week_embedding = bool(use_day_of_week_embedding)
        self.tod_emb_dim = int(tod_emb_dim)
        self.dow_emb_dim = int(dow_emb_dim)
        self.num_time_in_day = int(num_time_in_day)
        self.num_day_in_week = int(num_day_in_week)
        self.require_time_features = bool(require_time_features)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if self.use_node_embedding:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
            projector_in_dim = hidden_dim + node_emb_dim
        else:
            self.node_emb = None
            projector_in_dim = hidden_dim
        if self.use_time_of_day_embedding:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.num_time_in_day, self.tod_emb_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            projector_in_dim += self.tod_emb_dim
        else:
            self.time_in_day_emb = None
        if self.use_day_of_week_embedding:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.num_day_in_week, self.dow_emb_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            projector_in_dim += self.dow_emb_dim
        else:
            self.day_in_week_emb = None

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

    @staticmethod
    def _time_index(values: torch.Tensor, size: int) -> torch.Tensor:
        values = torch.nan_to_num(values, nan=0.0)
        if values.numel() > 0 and float(values.detach().max().cpu()) <= 1.0 + 1e-6:
            values = values * size
        return values.long().clamp_(0, size - 1)

    def _last_time_feature(
        self,
        seq_time: Optional[torch.Tensor],
        cur_time: Optional[torch.Tensor],
        feature_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_time is not None:
            if seq_time.dim() == 3:
                return seq_time[:, -1, feature_idx].unsqueeze(-1).expand(-1, self.num_nodes)
            if seq_time.dim() == 4:
                return seq_time[:, -1, :, feature_idx]
            raise ValueError(f"seq_time must be [B,L,T] or [B,L,N,T], got {tuple(seq_time.shape)}")
        if cur_time is not None:
            if cur_time.dim() == 2:
                return cur_time[:, feature_idx].unsqueeze(-1).expand(-1, self.num_nodes)
            if cur_time.dim() == 3:
                return cur_time[:, :, feature_idx]
            raise ValueError(f"cur_time must be [B,T] or [B,N,T], got {tuple(cur_time.shape)}")
        if self.require_time_features:
            raise ValueError("STIDMLPBackbone requires seq_time/cur_time for TOD/DOW embeddings.")
        return torch.zeros(batch_size, self.num_nodes, device=device)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        seq_time: Optional[torch.Tensor] = None,
        cur_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del adj, kwargs
        x = self._check_input(x)
        batch_size, input_len, num_nodes, input_dim = x.shape
        node_history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, input_len * input_dim)
        h = self.temporal_encoder(node_history)
        if self.node_emb is not None:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            h = torch.cat([h, node_emb], dim=-1)
        if self.time_in_day_emb is not None:
            tod = self._last_time_feature(seq_time, cur_time, 0, batch_size, x.device).to(device=x.device)
            tod_emb = self.time_in_day_emb[self._time_index(tod, self.num_time_in_day)]
            h = torch.cat([h, tod_emb], dim=-1)
        if self.day_in_week_emb is not None:
            dow = self._last_time_feature(seq_time, cur_time, 1, batch_size, x.device).to(device=x.device)
            dow_emb = self.day_in_week_emb[self._time_index(dow, self.num_day_in_week)]
            h = torch.cat([h, dow_emb], dim=-1)
        z_inv = self.projector(h)
        y_inv = self.forecast_from_representation(z_inv)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
