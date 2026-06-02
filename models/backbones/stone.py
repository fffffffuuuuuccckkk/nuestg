from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


class _Align(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            return self.align_conv(x)
        if self.c_in < self.c_out:
            batch_size, _, timestep, node_num = x.shape
            pad = torch.zeros(batch_size, self.c_out - self.c_in, timestep, node_num, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x


class _CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, dilation: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=nn.modules.utils._pair(kernel_size),
            stride=1,
            padding=0,
            dilation=dilation,
            bias=True,
        )


class _DilationGatedTemporalConvLayer(nn.Module):
    """Official STONE gated temporal convolution layer."""

    def __init__(self, kt: int, c_in: int, c_out: int, dilation: int, node_num: int) -> None:
        super().__init__()
        del node_num
        self.kt = int(kt)
        self.c_out = int(c_out)
        self.dilation = int(dilation)
        self.align = _Align(c_in, c_out)
        self.causal_conv = _CausalConv2d(c_in, 2 * c_out, kernel_size=(kt, 1), dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)[:, :, (self.kt - 1) * self.dilation :, :]
        x_conv = self.causal_conv(x)
        x_p = x_conv[:, : self.c_out, :, :]
        x_q = x_conv[:, -self.c_out :, :, :]
        return (x_p + x_in) * torch.sigmoid(x_q)


class _AttentionMLP(nn.Module):
    """Official STONE semantic attention MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        sem_dim: int,
        has_shallow_encode: bool = True,
    ) -> None:
        super().__init__()
        del hidden_dim
        self.has_shallow_encode = bool(has_shallow_encode)
        if self.has_shallow_encode:
            self.shallow_encode = nn.Sequential(
                nn.Linear(sem_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
            )
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.output_dim = int(output_dim)

    def forward(self, sem: torch.Tensor) -> torch.Tensor:
        if self.has_shallow_encode:
            sem = self.shallow_encode(sem)
        gate = self.gate(sem)
        residual = sem
        q = self.q(sem)
        k = self.k(sem)
        v = self.v(sem)
        att = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(self.output_dim)
        att = torch.softmax(att, dim=-1)
        sem = torch.einsum("bjd,bij->bid", v, att)
        sem = self.bn(sem.transpose(1, 2)).transpose(1, 2)
        sem = F.relu(sem)
        return gate * residual + (1.0 - gate) * sem


class _AdaptiveInteraction(nn.Module):
    """Official STONE adaptive interaction graph constructor."""

    def __init__(self, input_dim: int, output_dim: int, node_num: int) -> None:
        super().__init__()
        del node_num
        self.e_out1 = nn.Linear(input_dim, output_dim)
        self.e_in1 = nn.Linear(input_dim, output_dim)
        self.e_out3 = nn.Linear(input_dim, output_dim)
        self.e_in3 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_out1 = self.e_out1(x)
        e_in1 = self.e_in1(x)
        e_out2 = self.e_out1(x)
        e_in2 = self.e_in1(x)
        e_out = torch.einsum("bid,bjd->bij", e_out1, e_out2) / math.sqrt(self.output_dim)
        e_in = torch.einsum("bid,bjd->bij", e_in1, e_in2) / math.sqrt(self.output_dim)
        e_out3 = torch.einsum("bij,bjd->bid", e_out, e_out2)
        e_in3 = torch.einsum("bij,bjd->bid", e_in, e_in2)
        e_out3 = self.bn1(e_out3.transpose(1, 2)).transpose(1, 2)
        e_in3 = self.bn2(e_in3.transpose(1, 2)).transpose(1, 2)
        adj = torch.einsum("bik,bjk->bij", e_out3, e_in3)
        return torch.softmax(F.relu(adj), dim=-1)


class _GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, ks: int) -> None:
        super().__init__()
        self.ks = int(ks)
        self.weight = nn.Parameter(torch.empty(ks + 1, c_in, c_out))
        self.bias = nn.Parameter(torch.empty(c_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_list = [x]
        for _ in range(self.ks):
            x_list.append(torch.einsum("bij,bjd->bid", adj, x_list[-1]))
        stacked = torch.stack(x_list, dim=0)
        out = torch.einsum("kbit,kts->bis", stacked, self.weight)
        return out + self.bias


class _GraphConvLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, ks: int) -> None:
        super().__init__()
        self.align = nn.Linear(c_in, c_out)
        self.graph_conv = _GraphConv(c_out, c_out, ks)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)
        return self.graph_conv(x_in, adj) + x_in


class _STBlock(nn.Module):
    def __init__(
        self,
        s_blocks: list[list[int]],
        t_blocks: list[list[int]],
        node_num: int,
        dropout: float,
        kt: int,
        sem_dim: int,
        has_shallow_encode: list[bool],
    ) -> None:
        super().__init__()
        self.s_mlp = nn.ModuleList(
            [
                _AttentionMLP(
                    input_dim=block[0],
                    output_dim=block[-1],
                    hidden_dim=block[1],
                    dropout=dropout,
                    sem_dim=sem_dim,
                    has_shallow_encode=has_shallow_encode[i],
                )
                for i, block in enumerate(s_blocks)
            ]
        )
        self.t_mlp = nn.ModuleList(
            [
                _DilationGatedTemporalConvLayer(
                    kt=kt,
                    c_in=block[0],
                    c_out=block[-1],
                    dilation=1,
                    node_num=node_num,
                )
                for block in t_blocks
            ]
        )
        self.t_mlp.append(
            _DilationGatedTemporalConvLayer(
                kt=2,
                c_in=t_blocks[-1][0],
                c_out=t_blocks[-1][-1],
                dilation=1,
                node_num=node_num,
            )
        )

    def forward(self, x: torch.Tensor, sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.s_mlp:
            sem = layer(sem)
        x_list = []
        for layer in self.t_mlp:
            x = layer(x)
            x_list.append(x)
        return torch.cat(x_list, dim=2), sem


class _STAggBlock(nn.Module):
    def __init__(
        self,
        x_input_dim: int,
        x_output_dim: int,
        sem_input_dim: int,
        sem_output_dim: int,
        node_num: int,
        dropout: float,
        ks_s: int,
        ks_t: int,
        adp_s_dim: int,
        adp_t_dim: int,
    ) -> None:
        super().__init__()
        self.t_diff = _GraphConvLayer(sem_input_dim, sem_output_dim, ks_t)
        self.s_conv = _GraphConvLayer(x_input_dim, x_output_dim, ks_s)
        self.t_adpadj = _AdaptiveInteraction(x_input_dim, adp_s_dim, node_num)
        self.s_adpadj = _AdaptiveInteraction(sem_input_dim, adp_t_dim, node_num)
        self.bn_sem = nn.BatchNorm1d(sem_output_dim)
        self.bn_x = nn.BatchNorm1d(x_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sem_adpadj = self.t_adpadj(x)
        x_adpadj = self.s_adpadj(sem)
        x = self.s_conv(x, x_adpadj)
        sem = self.t_diff(sem, sem_adpadj)
        x = self.dropout(self.bn_x(F.relu(x).transpose(1, 2)).transpose(1, 2))
        sem = self.dropout(self.bn_sem(F.relu(sem).transpose(1, 2)).transpose(1, 2))
        return x, sem


class _GatedFusionBlock(nn.Module):
    def __init__(self, sem_input_dim: int, x_input_dim: int, gate_output_dim: int, horizon: int, output_dim: int) -> None:
        super().__init__()
        self.gate_sem = nn.Linear(sem_input_dim + x_input_dim, gate_output_dim)
        self.gate_x = nn.Linear(x_input_dim, gate_output_dim)
        self.out1 = nn.Linear(gate_output_dim, gate_output_dim)
        self.out2 = nn.Linear(gate_output_dim, horizon * output_dim)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor, sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_sem(torch.cat((sem, x), dim=-1)))
        hidden = gate * self.gate_x(x)
        hidden = F.relu(self.out1(hidden))
        pred = self.out2(hidden)
        batch_size, num_nodes, _ = pred.shape
        pred = pred.view(batch_size, num_nodes, self.horizon, self.output_dim).permute(0, 2, 1, 3)
        return pred, hidden


class STONEBackbone(BaseBackbone):
    """Faithful STONE architecture adapter for the local fixed-node runner.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/STONE-KDD-2024/Knowair/model/STONE.py
      - /data/OuXiaoyu/mystg/baselines/STONE-KDD-2024/src/base/stone.py

    The module now follows the official STONE path: semantic attention
    `STBlock`, gated temporal convolutions, adaptive spatial/temporal graph
    aggregation `STAggBlock`, and `GatedFusionBlock`. PEMS08 still lacks the
    official coordinate/Frechet/structural-shift side information, so semantic
    inputs are learnable node embeddings under the fixed-node protocol.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        SBlocks_num: int = 2,
        TBlocks_num: int = 5,
        sem_dim: int = 64,
        hidden_dim: int = 64,
        temporal_channels: int = 128,
        x_output_dim: int = 128,
        sem_output_dim: int = 64,
        gate_output_dim: int = 128,
        Kt: int = 3,
        Ks_s: int = 1,
        Ks_t: int = 1,
        dropout: float = 0.3,
        adp_s_dim: int = 20,
        adp_t_dim: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        s_blocks = [[hidden_dim, 16, hidden_dim] for _ in range(int(SBlocks_num))]
        has_shallow_encode = [idx == 0 for idx in range(int(SBlocks_num))]
        t_blocks = [[input_dim, temporal_channels]]
        t_blocks.extend([[temporal_channels, temporal_channels] for _ in range(max(0, int(TBlocks_num) - 1))])
        temporal_total = self._temporal_total_length(input_len, len(t_blocks), int(Kt))
        self.semantic = nn.Parameter(torch.randn(num_nodes, sem_dim))
        self.stblock = _STBlock(
            s_blocks=s_blocks,
            t_blocks=t_blocks,
            node_num=num_nodes,
            dropout=dropout,
            kt=int(Kt),
            sem_dim=sem_dim,
            has_shallow_encode=has_shallow_encode,
        )
        self.x1 = nn.Linear(temporal_total, gate_output_dim)
        self.x2 = nn.Linear(gate_output_dim, 1)
        self.stagg = _STAggBlock(
            x_input_dim=t_blocks[-1][-1],
            x_output_dim=x_output_dim,
            sem_input_dim=s_blocks[-1][-1],
            sem_output_dim=sem_output_dim,
            node_num=num_nodes,
            dropout=dropout,
            ks_s=int(Ks_s),
            ks_t=int(Ks_t),
            adp_s_dim=int(adp_s_dim),
            adp_t_dim=int(adp_t_dim),
        )
        self.gatefusion = _GatedFusionBlock(sem_output_dim, x_output_dim, gate_output_dim, output_len, output_dim)
        self.representation_proj = nn.Linear(gate_output_dim, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    @staticmethod
    def _temporal_total_length(input_len: int, num_t_blocks: int, kt: int) -> int:
        length = int(input_len)
        total = 0
        for _ in range(num_t_blocks):
            length -= int(kt) - 1
            if length < 1:
                raise ValueError("STONE temporal blocks shrink input_len below 1; reduce TBlocks_num or Kt.")
            total += length
        length -= 1
        if length < 1:
            raise ValueError("STONE final temporal block shrinks input_len below 1; reduce TBlocks_num or Kt.")
        return total + length

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del adj, kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        sem = self.semantic.unsqueeze(0).expand(batch_size, -1, -1)
        x_temporal, sem = self.stblock(x.permute(0, 3, 1, 2), sem)
        x_temporal = self.x1(x_temporal.transpose(2, 3))
        x_temporal = F.relu(x_temporal)
        x_temporal = self.x2(x_temporal).transpose(2, 3)
        x_repr = x_temporal.permute(0, 3, 1, 2).squeeze(-1)
        x_repr, sem = self.stagg(x_repr, sem)
        y_inv, hidden = self.gatefusion(x_repr, sem)
        z_inv = self.representation_proj(hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
