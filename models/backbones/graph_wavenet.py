from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


class GraphConv(nn.Module):
    """Small Graph WaveNet-style graph convolution with a fixed support budget."""

    def __init__(self, input_channels: int, output_channels: int, max_supports: int, order: int, dropout: float) -> None:
        super().__init__()
        self.max_supports = max(0, int(max_supports))
        self.order = max(1, int(order))
        self.dropout = float(dropout)
        self.proj = nn.Conv2d(
            input_channels * (1 + self.max_supports * self.order),
            output_channels,
            kernel_size=(1, 1),
        )

    @staticmethod
    def nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, support)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for support in supports[: self.max_supports]:
            x1 = self.nconv(x, support)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, support)
                out.append(x1)
        while len(out) < 1 + self.max_supports * self.order:
            out.append(torch.zeros_like(x))
        h = torch.cat(out, dim=1)
        h = self.proj(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class GraphWaveNetBackbone(BaseBackbone):
    """Graph WaveNet-style invariant backbone for NUE-STG.

    This module follows the main Graph WaveNet ideas: dilated gated temporal
    convolutions, skip/residual paths, optional static graph supports, and an
    optional adaptive adjacency. It is intentionally a compact, stable
    Graph-WaveNet-style implementation rather than a line-by-line copy of the
    original repository.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        kernel_size: int = 2,
        dilation_exponential: int = 2,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 128,
        end_channels: int = 256,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit=None,
        supports_len: int = 1,
        use_static_adj: bool = True,
        adj_norm: str = "sym",
        adjtype: str = "doubletransition",
        support_add_self_loop: bool = False,
        adaptive_embed_dim: int = 10,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del hidden_dim, aptinit, adj_norm, support_add_self_loop
        self.dropout = float(dropout)
        self.blocks = int(blocks)
        self.layers = int(layers)
        self.kernel_size = int(kernel_size)
        self.dilation_exponential = int(dilation_exponential)
        self.gcn_bool = bool(gcn_bool)
        self.addaptadj = bool(addaptadj)
        self.use_static_adj = bool(use_static_adj)
        self.adjtype = str(adjtype or "doubletransition").lower()

        self.start_conv = nn.Conv2d(input_dim, residual_channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.norms = nn.ModuleList()

        static_supports_len = 0
        if self.use_static_adj:
            static_supports_len = 2 if self.adjtype in {"doubletransition", "dual_random_walk", "double_transition"} else 1
        max_supports = static_supports_len + (1 if self.addaptadj else 0)
        max_supports = max(max_supports, int(supports_len or 0))
        self.max_supports = max_supports

        for block in range(self.blocks):
            for layer in range(self.layers):
                dilation = self.dilation_exponential ** layer
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                if self.gcn_bool:
                    self.gconvs.append(GraphConv(dilation_channels, residual_channels, max_supports, order=2, dropout=dropout))
                else:
                    self.gconvs.append(nn.Identity())
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.norms.append(nn.BatchNorm2d(residual_channels))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, output_len * output_dim, kernel_size=(1, 1), bias=True)
        self.representation_proj = nn.Linear(end_channels, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

        if self.addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adaptive_embed_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(adaptive_embed_dim, num_nodes) * 0.1)
        else:
            self.nodevec1 = None
            self.nodevec2 = None

    @staticmethod
    def _random_walk(adj: torch.Tensor) -> torch.Tensor:
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / denom

    def _supports(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        supports: List[torch.Tensor] = []
        if self.use_static_adj and adj is not None:
            adj = adj.to(device=device, dtype=dtype)
            if adj.dim() == 3:
                supports.extend([adj_i for adj_i in adj])
            elif self.adjtype in {"doubletransition", "dual_random_walk", "double_transition"}:
                supports.append(self._random_walk(adj))
                supports.append(self._random_walk(adj.transpose(0, 1)))
            elif self.adjtype in {"transition", "random_walk", "row"}:
                supports.append(self._random_walk(adj))
            else:
                supports.append(adj)
        if self.addaptadj and self.nodevec1 is not None and self.nodevec2 is not None:
            adp = torch.softmax(torch.relu(self.nodevec1.matmul(self.nodevec2)), dim=1)
            supports.append(adp.to(device=device, dtype=dtype))
        return supports

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self._check_input(x)
        batch_size = x.shape[0]
        x_gw = x.permute(0, 3, 2, 1)
        h = self.start_conv(x_gw)
        supports = self._supports(adj, h.device, h.dtype)
        skip = None

        for idx in range(len(self.filter_convs)):
            residual = h
            dilation = self.filter_convs[idx].dilation[1]
            pad = dilation * (self.kernel_size - 1)
            h_pad = F.pad(h, (pad, 0, 0, 0))
            filtered = torch.tanh(self.filter_convs[idx](h_pad))
            gated = torch.sigmoid(self.gate_convs[idx](h_pad))
            h = filtered * gated
            s = self.skip_convs[idx](h)
            skip = s if skip is None else skip + s
            if self.gcn_bool and supports:
                h = self.gconvs[idx](h, supports)
            else:
                h = self.residual_convs[idx](h)
            h = self.norms[idx](h + residual[..., -h.size(-1) :])

        if skip is None:
            skip = h.new_zeros(batch_size, self.end_conv_1.in_channels, self.num_nodes, 1)
        rep = F.relu(self.end_conv_1(F.relu(skip)))
        out = self.end_conv_2(rep)[..., -1]
        y_inv = out.view(batch_size, self.output_len, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        node_hidden = rep[..., -1].transpose(1, 2)
        z_inv = self.representation_proj(node_hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
