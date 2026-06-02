from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from models.backbones._time_utils import append_normalized_time_features
from models.backbones.base import BaseBackbone


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        return self.avg(torch.cat([front, x, end], dim=1).permute(0, 2, 1)).permute(0, 2, 1)


class _FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv2d(4 * dim, dim, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


class _CoreAdaptive(nn.Module):
    """Official STOP `Core_Adaptive` module from LargeST `src/models/stop.py`."""

    def __init__(
        self,
        d_in: int,
        d_core: int,
        d_out: int,
        core_num: int,
        head: int = 4,
        nndropout: float = 0.3,
    ) -> None:
        super().__init__()
        if core_num <= 0:
            raise ValueError("core_num must be positive when Core_Adaptive is enabled")
        if head <= 0 or d_core % head != 0:
            raise ValueError(f"d_core={d_core} must be divisible by head={head}")
        self.head_dim = d_core // head
        self.cores = nn.Parameter(torch.randn((head, core_num, self.head_dim)))
        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4 * (d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(nndropout),
            nn.Conv2d(4 * (d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        self.norm = nn.BatchNorm2d(d_out)
        self.head = int(head)

    def forward(self, x: torch.Tensor, ssie: Optional[torch.Tensor] = None, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        del ssie, adj
        x_in = x.permute(0, 3, 1, 2)
        batch_size, channels, steps, nodes = x_in.shape
        q = self.value(x_in)
        q = torch.stack(torch.split(q, self.head_dim, dim=1), dim=1)
        affiliation = torch.einsum("hcd,bhdtn->bhctn", self.cores, q).transpose(-2, -3) / math.sqrt(self.head_dim)
        node_to_core = torch.softmax(affiliation, dim=-1)
        core_to_node = torch.softmax(affiliation, dim=-2)
        v = torch.stack(torch.split(x_in, self.head_dim, dim=1), dim=1)
        v = torch.einsum("bhftn,bhtcn->bhftc", v, node_to_core)
        v = torch.einsum("bhftc,bhtcn->bhftn", v, core_to_node)
        v = v.transpose(0, 1).reshape(batch_size, channels, steps, nodes)
        out = torch.cat([x_in - v, v], dim=1)
        out = self.ffn(out)
        out = self.norm(out + x_in)
        return out.permute(0, 2, 3, 1)


class _STOPBaseMLP(nn.Module):
    """Native copy of STOP LargeST `MLP` adapted to local timestamp arrays."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_dim: int,
        num_layer: int,
        model_dim: int,
        prompt_dim: int,
        tod_size: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.model_dim = int(model_dim)
        self.prompt_dim = int(prompt_dim)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod_size, prompt_dim))
        self.day_in_week_emb = nn.Parameter(torch.empty(7, prompt_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)
        self.decomposition = _MovingAverage(kernel_size)
        self.time_series_emb_layer1 = nn.Conv1d(input_len, model_dim, kernel_size=1)
        self.time_series_emb_layer2 = nn.Conv1d(input_len, model_dim, kernel_size=1)
        self.hidden_dim = model_dim + 2 * prompt_dim
        self.encoder = nn.Sequential(*[_FeedForward(self.hidden_dim) for _ in range(num_layer)])
        self.regression_layer = nn.Conv2d(self.hidden_dim, output_len * output_dim, kernel_size=(1, 1))

    @staticmethod
    def _idx(values: torch.Tensor, size: int) -> torch.Tensor:
        return torch.floor(torch.nan_to_num(values, nan=0.0) * float(size)).long().clamp(0, size - 1)

    def forward(self, history_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = history_data[..., 0]
        tod = history_data[..., 1]
        dow = history_data[..., 2]
        tid_emb = self.time_in_day_emb[self._idx(tod[:, -1, :], self.time_in_day_emb.shape[0])]
        diw_emb = self.day_in_week_emb[self._idx(dow[:, -1, :], 7)]
        trend = self.decomposition(value)
        seasonal = value - trend
        time_series_emb = (self.time_series_emb_layer1(seasonal) + self.time_series_emb_layer2(trend)).unsqueeze(-1)
        hidden = torch.cat(
            [time_series_emb, tid_emb.transpose(1, 2).unsqueeze(-1), diw_emb.transpose(1, 2).unsqueeze(-1)],
            dim=1,
        )
        h = hidden.transpose(1, -1)
        hidden = self.encoder(hidden)
        z = hidden.transpose(1, -1)
        prediction = self.regression_layer(hidden)
        return h, z, prediction

    def module(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(hidden.transpose(1, -1))
        return hidden.transpose(1, -1)


class STOPBackbone(BaseBackbone):
    """Native STOP adapter based on official LargeST `src/models/stop.py`.

    It keeps series decomposition, TOD/DOW prompt embeddings, residual MLP
    encoder, backcast residual branch, and decoder. The special spatial OOD
    train/eval protocol remains outside this fixed-node local train.py adapter.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        num_layer: int = 3,
        model_dim: int = 64,
        prompt_dim: int = 32,
        kernel_size: int = 3,
        hid_dim: int = 256,
        tod_size: int = 288,
        extra_type: bool = True,
        same: bool = False,
        core: int = 8,
        head: int = 4,
        core_dropout: float = 0.3,
        num_time_in_day: int = 288,
        num_day_in_week: int = 7,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.num_time_in_day = int(num_time_in_day)
        self.num_day_in_week = int(num_day_in_week)
        self.extra_type = bool(extra_type)
        self.same = bool(same)
        self.stmodel = _STOPBaseMLP(input_len, output_len, output_dim, num_layer, model_dim, prompt_dim, tod_size, kernel_size)
        hidden_dim = model_dim + 2 * prompt_dim
        self.stmodel_detach = _STOPBaseMLP(input_len, output_len, output_dim, num_layer, model_dim, prompt_dim, tod_size, kernel_size)
        self.use_core_adaptive = int(core) > 0
        if self.use_core_adaptive:
            self.backcast = _CoreAdaptive(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                core_num=int(core),
                head=int(head),
                nndropout=float(core_dropout),
            )
        else:
            self.backcast = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hid_dim), nn.GELU(), nn.Linear(hid_dim, output_len))
        self.representation_proj = nn.Linear(hidden_dim, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

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
        history = append_normalized_time_features(x, seq_time, self.num_time_in_day, self.num_day_in_week)
        h, z, y = self.stmodel(history)
        if self.extra_type:
            h_res = self.backcast(z, None) if self.use_core_adaptive else self.backcast(z)
            residual_encoder = self.stmodel if self.same else self.stmodel_detach
            z_res = residual_encoder.module(h - h_res)
            y = y + self.decoder(z_res).transpose(1, -1)
        y_inv = y.view(x.shape[0], self.output_len, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        z_inv = self.representation_proj(z.squeeze(1))
        self._assert_outputs(z_inv, y_inv, x.shape[0])
        return {"z_inv": z_inv, "y_inv": y_inv}
