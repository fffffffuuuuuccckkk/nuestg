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
    """Pure PyTorch copy of CaST `DilatedConvEncoder`."""

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
    """CaST `SelfAttention` without the external repo dependency chain."""

    def __init__(self, num_heads: int, in_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        if hid_dim % num_heads != 0:
            raise ValueError(f"hid_dim={hid_dim} must be divisible by num_heads={num_heads}")
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
    """Pure PyTorch copy of CaST `TempDisentangler`."""

    def __init__(self, dim: int, kernels: list[int], length: int, dropout: float) -> None:
        super().__init__()
        self.env_encoder = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k - 1) for k in kernels])
        self.entity_time = _SelfAttention(num_heads=4, in_dim=dim, hid_dim=dim, dropout=dropout)
        self.kernels = list(kernels)
        self.length = int(length)
        self.num_freqs = self.length // 2 + 1
        self.freq_weight = nn.Parameter(torch.empty(self.num_freqs, dim, dim, dtype=torch.cfloat))
        self.freq_bias = nn.Parameter(torch.empty(self.num_freqs, dim, dtype=torch.cfloat))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.freq_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.freq_bias, -bound, bound)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_rep = []
        for kernel, conv in zip(self.kernels, self.env_encoder):
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
    """CaST environment codebook with straight-through quantization."""

    def __init__(self, num_envs: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_envs, dim)
        self.embedding.weight.data.uniform_(-1.0 / num_envs, 1.0 / num_envs)

    def straight_through(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(x.contiguous(), self.embedding.weight.detach())
        indices = distances.argmin(dim=-1)
        quantized_bar = self.embedding(indices).contiguous()
        quantized = x + (quantized_bar - x).detach()
        return quantized, quantized_bar, indices


class _DenseHodgeLaguerreConv(nn.Module):
    """Dense PyTorch equivalent of CaST `HodgeLaguerreConv`.

    The official implementation is torch_geometric-based. The basicts
    environment has no torch_geometric, so this module applies the same
    Laguerre recurrence on a dense edge-line graph.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True) -> None:
        super().__init__()
        if K <= 0:
            raise ValueError("K must be positive")
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)])
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, edge_adj: torch.Tensor) -> torch.Tensor:
        tx0 = x
        tx1 = x
        out = self.lins[0](tx0)
        if len(self.lins) > 1:
            tx1 = x - torch.einsum("ef,bfd->bed", edge_adj, x)
            out = out + self.lins[1](tx1)
        k = 1
        for lin in self.lins[2:]:
            propagated = torch.einsum("ef,bfd->bed", edge_adj, tx1)
            tx2 = (-propagated + (2 * k + 1) * tx1 - k * tx0) / (k + 1)
            out = out + lin(tx2)
            tx0, tx1 = tx1, tx2
            k += 1
        if self.bias is not None:
            out = out + self.bias
        return out


class _DenseCaSTGCNConv(nn.Module):
    """Dense PyTorch equivalent of CaST `GCNConv` over directed causal scores."""

    def __init__(self, in_channels: int, num_nodes: int, out_channels: int, K: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.K = int(K)
        self.num_nodes = int(num_nodes)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D], edge_weight: [B,E,2,K] for src->dst and dst->src.
        hidden = self.lin(x)
        residual = hidden
        batch_size = hidden.shape[0]
        for k in range(self.K):
            support = hidden.new_zeros(batch_size, self.num_nodes, self.num_nodes)
            support[:, src, dst] = edge_weight[:, :, 0, k]
            support[:, dst, src] = edge_weight[:, :, 1, k]
            hidden = torch.einsum("bij,bjd->bid", support, hidden)
            hidden = F.relu(hidden)
        out = residual + hidden + self.bias
        return self.layer_norm(out)


class CaSTBackbone(BaseBackbone):
    """Faithful PyTorch CaST adapter for the local fixed-node runner.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/CaST/src/models/cast.py
      - /data/OuXiaoyu/mystg/baselines/CaST/src/layers/cast_cell.py
      - /data/OuXiaoyu/mystg/baselines/CaST/src/trainers/cast_trainer.py

    This keeps CaST's temporal disentangler, environment codebook, Hodge
    Laguerre edge convolution, directed causal node message passing, node
    embeddings, predictor, and VQ/commitment/MI auxiliary losses. It is still a
    fixed-node adapter because the current PEMS08 BasicTS batch does not expose
    official torch_geometric `PairData` samples.
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
        depth: int = 10,
        dropout: float = 0.2,
        n_envs: int = 5,
        time_delay_scaler: int = 6,
        beta1: float = 1.0,
        beta2: float = 1.0,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        self.hid_dim = int(hid_dim)
        self.node_embed_dim = int(node_embed_dim)
        self.K = int(K)
        self.time_delay_scaler = max(1, int(time_delay_scaler))
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.start_encoder = _DilatedConvEncoder(input_dim, [input_dim] * int(depth) + [hid_dim], kernel_size=3)
        kernels = [2**i for i in range(max(1, int(math.log2(max(2, input_len // 2)))))]
        self.temporal = _TempDisentangler(hid_dim, kernels, input_len, dropout)
        self.codebook = _EnvEmbedding(n_envs, node_embed_dim * input_len)
        self.t_proj_env = nn.Linear(hid_dim, node_embed_dim)
        self.env_lin = nn.Linear(node_embed_dim * input_len, hid_dim)
        self.t_proj_cau = nn.Linear(input_len, 1)
        self.edge_feature_dim = 2 + len(range(0, input_len, self.time_delay_scaler))
        self.start_mlp_edge = nn.Linear(self.edge_feature_dim, hid_dim)
        self.spatial_edge = _DenseHodgeLaguerreConv(hid_dim, hid_dim, K=self.K, bias=bool(bias))
        self.edge_causal = nn.Linear(hid_dim, self.K * 2)
        self.spatial_node = _DenseCaSTGCNConv(hid_dim, num_nodes, hid_dim, K=self.K)
        self.env_cla = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, n_envs),
            nn.Softmax(dim=1),
        )
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

    def _base_adj(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if adj is None:
            src = torch.arange(self.num_nodes, device=device)
            dst = torch.roll(src, shifts=-1)
            base = torch.zeros(self.num_nodes, self.num_nodes, device=device, dtype=dtype)
            base[src, dst] = 1.0
            base[dst, src] = 1.0
            return base
        base = adj[0] if adj.dim() == 3 else adj
        base = base.to(device=device, dtype=dtype)
        if base.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f"CaST adjacency must be [{self.num_nodes},{self.num_nodes}], got {tuple(base.shape)}")
        return base.clamp_min(0)

    def _edge_structures(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = adj.device, adj.dtype
        undirected = ((adj + adj.transpose(0, 1)) > 0)
        src, dst = torch.nonzero(torch.triu(undirected, diagonal=1), as_tuple=True)
        if src.numel() == 0:
            src = torch.arange(self.num_nodes, device=device)
            dst = torch.roll(src, shifts=-1)
        incidence = torch.zeros(self.num_nodes, src.numel(), device=device, dtype=dtype)
        incidence[src, torch.arange(src.numel(), device=device)] = -1.0
        incidence[dst, torch.arange(src.numel(), device=device)] = 1.0
        l0 = incidence @ incidence.transpose(0, 1)
        lambda_max = torch.linalg.eigvalsh(l0.float()).max().to(device=device, dtype=dtype).clamp_min(1e-6)
        line_adj = 2.0 * (incidence.transpose(0, 1) @ incidence) / lambda_max
        edge_static = torch.stack([adj[src, dst], adj[dst, src]], dim=-1)
        return src.long(), dst.long(), line_adj, edge_static

    def _edge_features(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_static: torch.Tensor) -> torch.Tensor:
        value = x[..., 0]
        sims = []
        for shift in range(0, self.input_len, self.time_delay_scaler):
            rolled = torch.roll(value[:, :, dst], shifts=-shift, dims=1)
            sims.append((value[:, :, src] - rolled).abs().mean(dim=1))
        delay = torch.stack(sims, dim=-1)
        static = edge_static.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([static, delay], dim=-1)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        x = self._check_input(x)
        batch_size = x.shape[0]
        h_node = self.start_encoder(
            x.permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_nodes, self.input_len, self.input_dim)
            .permute(0, 2, 1)
        )
        h_environment, h_entity = self.temporal(h_node)

        h_environment = self.t_proj_env(h_environment).reshape(
            batch_size * self.num_nodes,
            self.input_len * self.node_embed_dim,
        )
        env_output, env_q, env_ind = self.codebook.straight_through(h_environment)
        env_output = self.env_lin(env_output).reshape(batch_size, self.num_nodes, self.hid_dim)

        base_adj = self._base_adj(adj, x.device, x.dtype)
        src, dst, line_adj, edge_static = self._edge_structures(base_adj)
        edge_feat = self._edge_features(x, src, dst, edge_static)
        h_link = self.start_mlp_edge(edge_feat.float()).to(dtype=x.dtype)
        h_link_updated = self.spatial_edge(h_link, line_adj)
        causal_score = self.edge_causal(h_link_updated).reshape(batch_size, src.numel(), 2, self.K)

        h_entity = self.t_proj_cau(h_entity.permute(0, 2, 1)).squeeze(-1).reshape(
            batch_size,
            self.num_nodes,
            self.hid_dim,
        )
        h_entity = self.spatial_node(h_entity, src, dst, causal_score)
        h_entity = h_entity + self.node_embed_lin_ent(self.node_embed).unsqueeze(0)
        env_output = env_output + self.node_embed_lin_env(self.node_embed).unsqueeze(0)

        hidden = torch.cat([env_output, h_entity], dim=-1)
        pred = self.end_mlp(hidden)
        y_inv = pred.view(batch_size, self.num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        z_inv = self.representation_proj(hidden)
        env_cla_pred = self.env_cla(h_entity.reshape(batch_size * self.num_nodes, -1))

        loss_vq = F.mse_loss(h_environment, env_q)
        loss_commit = F.mse_loss(env_q, h_environment)
        loss_mi = -F.cross_entropy(env_cla_pred, env_ind)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {
            "z_inv": z_inv,
            "y_inv": y_inv,
            "backbone_aux_losses": {
                "cast_vq_loss": loss_vq,
                "cast_commit_loss": loss_commit,
                "cast_mi_loss": loss_mi,
            },
            "backbone_aux_weights": {
                "cast_vq_loss": 1.0,
                "cast_commit_loss": self.beta1,
                "cast_mi_loss": self.beta2,
            },
        }
