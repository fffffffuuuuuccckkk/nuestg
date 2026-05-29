from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from models.backbones.base import BaseBackbone


class STIDResidualMLP(nn.Module):
    """Official STID residual 1x1 MLP block.

    Reference: /data/OuXiaoyu/mystg/baselines/STID/stid/arch/mlp.py
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x)))) + x


class STIDBackbone(BaseBackbone):
    """Faithful native STID backbone adapted to the local BaseBackbone API.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/STID/stid/arch/stid_arch.py
      - /data/OuXiaoyu/mystg/baselines/STID/stid/PEMS08.py

    Adaptations: the official BasicTS model receives value, time-of-day, and
    day-of-week as input channels. The local runner keeps values and timestamp
    arrays separate, so this wrapper builds the same embeddings from seq_time.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        embed_dim: int = 32,
        num_layer: int = 3,
        if_node: bool = True,
        node_dim: int = 32,
        if_T_i_D: bool = True,
        if_D_i_W: bool = True,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        dropout: float = 0.15,
        require_time_features: bool = True,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.embed_dim = int(embed_dim)
        self.num_layer = int(num_layer)
        self.if_spatial = bool(if_node)
        self.if_time_in_day = bool(if_T_i_D)
        self.if_day_in_week = bool(if_D_i_W)
        self.node_dim = int(node_dim)
        self.temp_dim_tid = int(temp_dim_tid)
        self.temp_dim_diw = int(temp_dim_diw)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        self.require_time_features = bool(require_time_features)

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        else:
            self.time_in_day_emb = None
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        else:
            self.day_in_week_emb = None

        self.time_series_emb_layer = nn.Conv2d(
            input_dim * input_len,
            self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[STIDResidualMLP(self.hidden_dim, dropout=dropout) for _ in range(self.num_layer)]
        )
        self.regression_layer = nn.Conv2d(
            self.hidden_dim,
            output_len * output_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.representation_proj = (
            nn.Identity()
            if self.hidden_dim == representation_dim
            else nn.Linear(self.hidden_dim, representation_dim)
        )
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    @staticmethod
    def _time_index(values: torch.Tensor, size: int) -> torch.Tensor:
        values = torch.nan_to_num(values, nan=0.0)
        if values.numel() > 0 and float(values.detach().max().cpu()) <= 1.0 + 1e-6:
            values = values * size
        return values.long().clamp_(0, size - 1)

    def _last_time_feature(self, seq_time: Optional[torch.Tensor], feature_idx: int, batch_size: int) -> torch.Tensor:
        if seq_time is None:
            if self.require_time_features:
                raise ValueError("STIDBackbone requires seq_time with time_of_day/day_of_week features.")
            return torch.zeros(batch_size, self.num_nodes, device=self.node_emb.device if self.node_emb is not None else None)
        if seq_time.dim() == 3:
            values = seq_time[:, -1, feature_idx].unsqueeze(-1).expand(-1, self.num_nodes)
        elif seq_time.dim() == 4:
            values = seq_time[:, -1, :, feature_idx]
        else:
            raise ValueError(f"seq_time must be [B,L,T] or [B,L,N,T], got {tuple(seq_time.shape)}")
        return values

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
        del adj, kwargs
        x = self._check_input(x)
        batch_size, input_len, num_nodes, input_dim = x.shape
        input_data = x.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, input_len * input_dim).transpose(1, 2).unsqueeze(-1)
        hidden_parts = [self.time_series_emb_layer(input_data)]

        if self.node_emb is not None:
            hidden_parts.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        if self.time_in_day_emb is not None:
            tod = self._last_time_feature(seq_time, 0, batch_size).to(device=x.device)
            hidden_parts.append(self.time_in_day_emb[self._time_index(tod, self.time_of_day_size)].transpose(1, 2).unsqueeze(-1))
        if self.day_in_week_emb is not None:
            dow = self._last_time_feature(seq_time, 1, batch_size).to(device=x.device)
            hidden_parts.append(self.day_in_week_emb[self._time_index(dow, self.day_of_week_size)].transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat(hidden_parts, dim=1)
        hidden = self.encoder(hidden)
        pred = self.regression_layer(hidden).squeeze(-1)
        y_inv = pred.view(batch_size, self.output_len, self.output_dim, num_nodes).permute(0, 1, 3, 2)
        node_hidden = hidden.squeeze(-1).transpose(1, 2)
        z_inv = self.representation_proj(node_hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
