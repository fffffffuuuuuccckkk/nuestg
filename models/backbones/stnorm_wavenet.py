from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


class SNorm(nn.Module):
    """Spatial normalization from ST-Norm.

    Reference: /data/OuXiaoyu/mystg/baselines/ST-Norm/models/Wavenet.py
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - x.mean(2, keepdim=True)) / (x.var(2, keepdim=True, unbiased=True) + 1e-5).sqrt()
        return x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class TNorm(nn.Module):
    """Temporal normalization from ST-Norm."""

    def __init__(self, num_nodes: int, channels: int, track_running_stats: bool = True, momentum: float = 0.1) -> None:
        super().__init__()
        self.track_running_stats = bool(track_running_stats)
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer("running_mean", torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_var", torch.ones(1, channels, num_nodes, 1))
        self.momentum = float(momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdim=True)
            var = x.var((0, 3), keepdim=True, unbiased=False)
            if self.training:
                n = max(x.shape[0] * x.shape[3], 2)
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1.0 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean(3, keepdim=True)
            var = x.var(3, keepdim=True, unbiased=True)
        x_norm = (x - mean) / (var + 1e-5).sqrt()
        return x_norm * self.gamma + self.beta


class STNormWaveNetBackbone(BaseBackbone):
    """Faithful native ST-Norm WaveNet backbone.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/ST-Norm/models/Wavenet.py
      - /data/OuXiaoyu/mystg/baselines/ST-Norm/main.py

    ST-Norm is model-internal spatial/temporal normalization, separate from
    train-split data scaling.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        tnorm_bool: bool = True,
        snorm_bool: bool = True,
        channels: int = 16,
        kernel_size: int = 2,
        blocks: int = 1,
        layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.blocks = int(blocks)
        self.layers = int(layers)
        self.tnorm_bool = bool(tnorm_bool)
        self.snorm_bool = bool(snorm_bool)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = nn.Dropout(float(dropout))

        self.start_conv = nn.Conv2d(input_dim, self.channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.tnorms = nn.ModuleList()
        self.snorms = nn.ModuleList()
        norm_inputs = 1 + int(self.tnorm_bool) + int(self.snorm_bool)
        for _ in range(self.blocks):
            dilation = 1
            for _ in range(self.layers):
                if self.tnorm_bool:
                    self.tnorms.append(TNorm(num_nodes, self.channels))
                if self.snorm_bool:
                    self.snorms.append(SNorm(self.channels))
                in_channels = norm_inputs * self.channels
                self.filter_convs.append(
                    nn.Conv2d(in_channels, self.channels, kernel_size=(1, self.kernel_size), dilation=dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(in_channels, self.channels, kernel_size=(1, self.kernel_size), dilation=dilation)
                )
                self.residual_convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1)))
                dilation *= 2
        self.end_conv_1 = nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(self.channels, output_len * output_dim, kernel_size=(1, 1), bias=True)
        self.representation_proj = nn.Linear(self.channels, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del adj, kwargs
        x = self._check_input(x).permute(0, 3, 2, 1)
        batch_size = x.shape[0]
        h = self.start_conv(x)
        skip = None
        t_idx = 0
        s_idx = 0
        for i in range(self.blocks * self.layers):
            residual = h
            pieces = [h]
            if self.tnorm_bool:
                pieces.append(self.tnorms[t_idx](h))
                t_idx += 1
            if self.snorm_bool:
                pieces.append(self.snorms[s_idx](h))
                s_idx += 1
            h_cat = torch.cat(pieces, dim=1)
            dilation = self.filter_convs[i].dilation[1]
            pad = dilation * (self.kernel_size - 1)
            h_pad = F.pad(h_cat, (pad, 0, 0, 0))
            filtered = torch.tanh(self.filter_convs[i](h_pad))
            gated = torch.sigmoid(self.gate_convs[i](h_pad))
            h = filtered * gated
            s = self.skip_convs[i](h)
            skip = s if skip is None else skip[..., -s.size(-1) :] + s
            h = self.residual_convs[i](h)
            h = h + residual[..., -h.size(-1) :]
        if skip is None:
            skip = h
        rep = F.relu(self.end_conv_1(F.relu(skip)))
        out = self.end_conv_2(rep)[..., -1]
        y_inv = out.view(batch_size, self.output_len, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        z_inv = self.representation_proj(rep[..., -1].transpose(1, 2))
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
