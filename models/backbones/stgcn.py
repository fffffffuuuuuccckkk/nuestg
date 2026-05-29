from __future__ import annotations

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
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in > c_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            return self.proj(x)
        if self.c_in < self.c_out:
            pad = x.new_zeros(x.shape[0], self.c_out - self.c_in, x.shape[2], x.shape[3])
            return torch.cat([x, pad], dim=1)
        return x


class _TemporalConvLayer(nn.Module):
    """STGCN gated temporal convolution.

    Reference: /data/OuXiaoyu/mystg/baselines/stgcn/model/layers.py
    """

    def __init__(self, kt: int, c_in: int, c_out: int, act_func: str = "glu") -> None:
        super().__init__()
        self.kt = int(kt)
        self.c_out = int(c_out)
        self.align = _Align(c_in, c_out)
        out_channels = 2 * c_out if act_func in {"glu", "gtu"} else c_out
        self.causal_conv = nn.Conv2d(c_in, out_channels, kernel_size=(self.kt, 1), bias=True)
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)[:, :, self.kt - 1 :, :]
        conv = self.causal_conv(x)
        if self.act_func in {"glu", "gtu"}:
            x_p = conv[:, : self.c_out]
            x_q = conv[:, -self.c_out :]
            if self.act_func == "glu":
                return (x_p + x_in) * torch.sigmoid(x_q)
            return torch.tanh(x_p + x_in) * torch.sigmoid(x_q)
        if self.act_func == "relu":
            return F.relu(conv + x_in)
        if self.act_func == "silu":
            return F.silu(conv + x_in)
        raise ValueError(f"Unsupported STGCN temporal activation {self.act_func!r}")


class _GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type: str, c_in: int, c_out: int, ks: int, bias: bool = True) -> None:
        super().__init__()
        self.graph_conv_type = graph_conv_type
        self.align = _Align(c_in, c_out)
        self.ks = max(1, int(ks))
        if graph_conv_type == "cheb_graph_conv":
            self.weight = nn.Parameter(torch.empty(self.ks, c_out, c_out))
        elif graph_conv_type == "graph_conv":
            self.weight = nn.Parameter(torch.empty(c_out, c_out))
        else:
            raise ValueError("graph_conv_type must be cheb_graph_conv or graph_conv")
        self.bias = nn.Parameter(torch.empty(c_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _cheb_supports(gso: torch.Tensor, ks: int) -> torch.Tensor:
        supports = [torch.eye(gso.shape[0], dtype=gso.dtype, device=gso.device)]
        if ks > 1:
            supports.append(gso)
        for _ in range(2, ks):
            supports.append(2 * gso.matmul(supports[-1]) - supports[-2])
        return torch.stack(supports[:ks], dim=0)

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)
        x_perm = x_in.permute(0, 2, 3, 1)
        if self.graph_conv_type == "cheb_graph_conv":
            supports = self._cheb_supports(gso, self.ks)
            x_g = torch.einsum("knm,btmc->btknc", supports, x_perm)
            x_gc = torch.einsum("btkni,kio->btno", x_g, self.weight)
        else:
            x_gc = torch.einsum("nm,btmc->btnc", gso, x_perm)
            x_gc = torch.einsum("btni,io->btno", x_gc, self.weight)
        if self.bias is not None:
            x_gc = x_gc + self.bias
        return x_gc.permute(0, 3, 1, 2) + x_in


class _STConvBlock(nn.Module):
    def __init__(
        self,
        kt: int,
        ks: int,
        num_nodes: int,
        c_in: int,
        channels: tuple[int, int, int],
        act_func: str,
        graph_conv_type: str,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tmp_conv1 = _TemporalConvLayer(kt, c_in, channels[0], act_func)
        self.graph_conv = _GraphConvLayer(graph_conv_type, channels[0], channels[1], ks, bias)
        self.tmp_conv2 = _TemporalConvLayer(kt, channels[1], channels[2], act_func)
        self.layer_norm = nn.LayerNorm([num_nodes, channels[2]], eps=1e-12)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)
        x = F.relu(self.graph_conv(x, gso))
        x = self.tmp_conv2(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class _OutputBlock(nn.Module):
    """STGCN final TNFF output block."""

    def __init__(
        self,
        ko: int,
        c_in: int,
        hidden_channels: tuple[int, int],
        out_channels: int,
        num_nodes: int,
        act_func: str,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tmp_conv1 = _TemporalConvLayer(ko, c_in, hidden_channels[0], act_func)
        self.layer_norm = nn.LayerNorm([num_nodes, hidden_channels[0]], eps=1e-12)
        self.fc1 = nn.Linear(hidden_channels[0], hidden_channels[1], bias=bias)
        self.fc2 = nn.Linear(hidden_channels[1], out_channels, bias=bias)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tmp_conv1(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1))
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        out = self.fc2(hidden).permute(0, 3, 1, 2)
        return out, hidden.squeeze(1)


class STGCNBackbone(BaseBackbone):
    """Faithful native STGCN backbone adapted to the local BaseBackbone API.

    Reference implementation: https://github.com/hazdzz/stgcn
    Files read: model/models.py, model/layers.py, main.py.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        kt: int = 3,
        ks: int = 3,
        stblock_num: int = 2,
        channels: tuple[int, int, int] = (64, 16, 64),
        act_func: str = "glu",
        graph_conv_type: str = "cheb_graph_conv",
        dropout: float = 0.5,
        enable_bias: bool = True,
        output_hidden_channels: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.kt = int(kt)
        self.ks = int(ks)
        self.stblock_num = int(stblock_num)
        self.graph_conv_type = graph_conv_type
        blocks = []
        c_in = input_dim
        for _ in range(self.stblock_num):
            block = _STConvBlock(
                self.kt,
                self.ks,
                num_nodes,
                c_in,
                tuple(channels),
                act_func,
                graph_conv_type,
                enable_bias,
                dropout,
            )
            blocks.append(block)
            c_in = channels[-1]
        self.blocks = nn.ModuleList(blocks)
        ko = input_len - self.stblock_num * 2 * (self.kt - 1)
        if ko < 1:
            raise ValueError(
                f"STGCN requires input_len - stblock_num*2*(kt-1) >= 1, got ko={ko}."
            )
        self.output = _OutputBlock(
            ko,
            c_in,
            tuple(output_hidden_channels),
            output_len * output_dim,
            num_nodes,
            act_func,
            enable_bias,
            dropout,
        )
        self.representation_proj = nn.Linear(output_hidden_channels[1], representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def _gso(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if adj is None:
            return torch.eye(self.num_nodes, device=device, dtype=dtype)
        if adj.dim() == 3:
            adj = adj[0]
        return adj.to(device=device, dtype=dtype)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        h = x.permute(0, 3, 1, 2)
        gso = self._gso(adj, x.device, x.dtype)
        for block in self.blocks:
            h = block(h, gso)
        out, node_hidden = self.output(h)
        z_inv = self.representation_proj(node_hidden)
        y_raw = out.squeeze(2).transpose(1, 2)
        y_inv = y_raw.view(batch_size, self.num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
