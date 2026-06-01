from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


class _SamePadConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        padding = receptive_field // 2
        self.remove = 1 if receptive_field % 2 == 0 else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : -self.remove] if self.remove > 0 else out


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, final: bool = False) -> None:
        super().__init__()
        self.conv1 = _SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = self.conv1(F.gelu(x))
        x = self.conv2(F.gelu(x))
        return x + residual


class _DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[
                _ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size,
                    dilation=2**i,
                    final=i == len(channels) - 1,
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class _SelfAttention(nn.Module):
    def __init__(self, num_heads: int, in_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hid_dim // self.num_heads
        self.query = nn.Linear(in_dim, hid_dim)
        self.key = nn.Linear(in_dim, hid_dim)
        self.value = nn.Linear(in_dim, hid_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hid_dim, hid_dim)
        self.norm = _LayerNorm(hid_dim)
        self.out_dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        return x.view(*shape).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query(x))
        k = self._split(self.key(x))
        v = self._split(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.shape[:-2], self.num_heads * self.head_dim)
        hidden = self.out_dropout(self.dense(context))
        return self.norm(hidden + x)


class _TempDisentangler(nn.Module):
    """CaST temporal entity/environment disentangler from cast_cell.py."""

    def __init__(self, dim: int, kernels: list[int], length: int, dropout: float) -> None:
        super().__init__()
        self.env_encoder = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k - 1) for k in kernels])
        self.entity_time = _SelfAttention(num_heads=4, in_dim=dim, hid_dim=dim, dropout=dropout)
        self.length = int(length)
        self.num_freqs = self.length // 2 + 1
        self.freq_weight = nn.Parameter(torch.empty(self.num_freqs, dim, dim, dtype=torch.cfloat))
        self.freq_bias = nn.Parameter(torch.empty(self.num_freqs, dim, dtype=torch.cfloat))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5))
        nn.init.zeros_(self.freq_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_rep = []
        for kernel, conv in zip([m.kernel_size[0] for m in self.env_encoder], self.env_encoder):
            out = conv(x)
            if kernel != 1:
                out = out[..., : -(kernel - 1)]
            env_rep.append(out.transpose(1, 2))
        environment = torch.stack(env_rep, dim=0).mean(dim=0)
        x_t = x.transpose(1, 2)
        entity_time = self.entity_time(x_t)
        input_freq = torch.fft.rfft(x_t, dim=1)
        output_freq = torch.zeros_like(input_freq)
        output_freq[:, : self.num_freqs] = (
            torch.einsum("bti,tio->bto", input_freq[:, : self.num_freqs], self.freq_weight) + self.freq_bias
        )
        entity_freq = torch.fft.irfft(output_freq, n=x_t.size(1), dim=1)
        return environment, self.dropout(entity_time + entity_freq)


class _EnvEmbedding(nn.Module):
    def __init__(self, num_envs: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_envs, dim)
        self.embedding.weight.data.uniform_(-1.0 / num_envs, 1.0 / num_envs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(x, self.embedding.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices)
        straight_through = x + (quantized - x).detach()
        return straight_through, quantized, indices


class CaSTBackbone(BaseBackbone):
    """Fixed-node CaST native adapter.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/CaST/src/models/cast.py
      - /data/OuXiaoyu/mystg/baselines/CaST/src/layers/cast_cell.py
      - /data/OuXiaoyu/mystg/baselines/CaST/src/utils/dataset.py

    The official repo requires torch_geometric graph Data objects. This adapter
    keeps the CaST temporal disentangler, environment codebook, node embeddings,
    causal edge scoring, and node message passing, but implements the graph
    operations with dense PyTorch adjacency for the local fixed-node PEMS setup.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        hid_dim: int = 16,
        node_embed_dim: int = 5,
        K: int = 2,
        depth: int = 4,
        dropout: float = 0.2,
        n_envs: int = 5,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.hid_dim = int(hid_dim)
        self.node_embed_dim = int(node_embed_dim)
        self.K = int(K)
        self.start_encoder = _DilatedConvEncoder(input_dim, [input_dim] * int(depth) + [hid_dim], kernel_size=3)
        kernels = [2**i for i in range(max(1, int(math.log2(max(2, input_len // 2)))))]
        self.temporal = _TempDisentangler(hid_dim, kernels, input_len, dropout)
        self.codebook = _EnvEmbedding(n_envs, node_embed_dim * input_len)
        self.t_proj_env = nn.Linear(hid_dim, node_embed_dim)
        self.env_lin = nn.Linear(node_embed_dim * input_len, hid_dim)
        self.t_proj_cau = nn.Linear(input_len, 1)
        self.edge_causal = nn.Sequential(nn.Linear(2 + max(1, input_len // 6), hid_dim), nn.ReLU(), nn.Linear(hid_dim, K))
        self.node_embed = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        self.node_embed_lin_ent = nn.Linear(node_embed_dim, hid_dim)
        self.node_embed_lin_env = nn.Linear(node_embed_dim, hid_dim)
        self.end_mlp = nn.Sequential(
            nn.LayerNorm([num_nodes, hid_dim * 2]),
            nn.Linear(hid_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_len * output_dim),
        )
        self.representation_proj = nn.Linear(hid_dim * 2, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def _dense_adj(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if adj is None:
            base = torch.eye(self.num_nodes, device=device, dtype=dtype)
        else:
            base = adj[0] if adj.dim() == 3 else adj
            base = base.to(device=device, dtype=dtype).clamp_min(0)
        base = base + torch.eye(self.num_nodes, device=device, dtype=dtype)
        return base / base.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _edge_summary(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        lags = max(1, self.input_len // 6)
        src, dst = torch.nonzero(adj > 0, as_tuple=True)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if src.numel() == 0:
            return adj.unsqueeze(0).expand(self.K, -1, -1)
        value = x[..., 0]
        sims = []
        for lag in range(lags):
            rolled = torch.roll(value[:, :, dst], shifts=-lag, dims=1)
            sims.append((value[:, :, src] - rolled).abs().mean(dim=(0, 1)))
        edge_feat = torch.cat([adj[src, dst].unsqueeze(-1), adj[dst, src].unsqueeze(-1), torch.stack(sims, dim=-1)], dim=-1)
        weights = torch.softmax(self.edge_causal(edge_feat), dim=-1)
        supports = []
        for k in range(self.K):
            support = torch.zeros_like(adj)
            support[src, dst] = weights[:, k]
            support = support + torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
            supports.append(support / support.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return torch.stack(supports, dim=0)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        h_node = self.start_encoder(x.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, self.input_len, self.input_dim).permute(0, 2, 1))
        h_environment, h_entity = self.temporal(h_node)
        env_flat = self.t_proj_env(h_environment).reshape(batch_size * self.num_nodes, self.input_len * self.node_embed_dim)
        env_output, _, _ = self.codebook(env_flat)
        env_output = self.env_lin(env_output).reshape(batch_size, self.num_nodes, self.hid_dim)
        h_entity = self.t_proj_cau(h_entity.permute(0, 2, 1)).squeeze(-1).reshape(batch_size, self.num_nodes, self.hid_dim)
        dense_adj = self._dense_adj(adj, x.device, x.dtype)
        supports = self._edge_summary(x, dense_adj)
        msg = 0.0
        for support in supports:
            msg = msg + torch.einsum("ij,bjd->bid", support, h_entity)
        h_entity = h_entity + msg / max(1, self.K)
        h_entity = h_entity + self.node_embed_lin_ent(self.node_embed).unsqueeze(0)
        env_output = env_output + self.node_embed_lin_env(self.node_embed).unsqueeze(0)
        hidden = torch.cat([env_output, h_entity], dim=-1)
        pred = self.end_mlp(hidden)
        y_inv = pred.view(batch_size, self.num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        z_inv = self.representation_proj(hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
