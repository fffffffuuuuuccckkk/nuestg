from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.backbones.base import BaseBackbone


def _as_node_mask(mask: Optional[torch.Tensor], num_nodes: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(num_nodes, device=device, dtype=torch.bool)
    if mask.dim() == 2:
        mask = mask.any(dim=0)
    mask = mask.to(device=device, dtype=torch.bool).flatten()
    if mask.numel() != num_nodes:
        raise ValueError(f"observed_mask must have {num_nodes} entries, got {mask.numel()}")
    if not bool(mask.any()):
        return torch.ones(num_nodes, device=device, dtype=torch.bool)
    return mask


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

    def forward(
        self,
        x: torch.Tensor,
        perturb_mask: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
        prior_weight: float = 0.0,
    ) -> torch.Tensor:
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
        adj = torch.softmax(F.relu(adj), dim=-1)
        if prior is not None and prior_weight > 0:
            prior = prior.to(device=adj.device, dtype=adj.dtype)
            if prior.dim() == 2:
                prior = prior.unsqueeze(0).expand_as(adj)
            adj = (1.0 - prior_weight) * adj + prior_weight * prior
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if perturb_mask is not None:
            mask = perturb_mask.to(device=adj.device, dtype=adj.dtype)
            if mask.dim() != 2:
                raise ValueError(f"graph perturbation mask must be [K,N], got {tuple(mask.shape)}")
            # Official Graph_Editer_Delete_Row uses einsum("kj,bij->kbij", m, adj).
            # We average the K perturbed graphs so BasicTS keeps a [B,N,N] graph.
            adj = (adj.unsqueeze(0) * mask[:, None, None, :]).mean(dim=0)
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj


class _GraphEditorDeleteRow(nn.Module):
    """Pure PyTorch STONE graph editor row/column deletion branch.

    The released STONE code trains two Graph_Editer_Delete_Row modules and
    applies their sampled masks to the semantic and temporal adaptive graphs.
    This adapter keeps that model-internal perturbation path, while averaging
    the K sampled graphs back to one graph for the fixed-node BasicTS runner.
    """

    def __init__(self, num_samples: int, num_nodes: int, sample_ratio: float) -> None:
        super().__init__()
        self.num_samples = max(1, int(num_samples))
        self.num_nodes = int(num_nodes)
        self.sample_ratio = float(sample_ratio)
        self.logits = nn.Parameter(torch.empty(self.num_samples, self.num_nodes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.logits)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.logits, dim=-1)
        sample_count = int(round(self.sample_ratio * self.num_nodes))
        sample_count = max(1, min(sample_count, self.num_nodes))
        selected = torch.multinomial(probs, num_samples=sample_count, replacement=False)
        mask = torch.ones_like(probs)
        if self.num_samples > 1:
            mask[1:].scatter_(1, selected[1:], 0.0)
        selected_probs = probs.gather(1, selected).sum(dim=-1)
        log_p = (selected_probs - torch.logsumexp(probs, dim=-1)).sum()
        return mask, log_p


class _SpatialSideInformation(nn.Module):
    """STONE spatial/Frechet side information for fixed-node BasicTS data.

    Official STONE builds Frechet embeddings from observed-node anchor sets and
    pairwise distances. PEMS08 here has no road metadata, so this module uses a
    cached shortest-path distance from the configured adjacency. If no adjacency
    reaches the backbone, it falls back to learnable anchor distances and marks
    that fallback through the config/documentation.
    """

    def __init__(self, num_nodes: int, anchor_repeats: int, seed: int) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.anchor_repeats = max(1, int(anchor_repeats))
        self.seed = int(seed)
        self.distortion = max(1, int(math.ceil(math.log2(max(self.num_nodes, 2)))))
        self.side_dim = self.anchor_repeats * self.distortion
        self.learnable_anchor_distance = nn.Parameter(torch.randn(self.num_nodes, self.side_dim) * 0.02)
        self._cache_key: tuple[int, torch.device, torch.dtype] | None = None
        self._cached_distance: Optional[torch.Tensor] = None

    def _distance_from_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        base = adj[0] if adj.dim() == 3 else adj
        base = base.clamp_min(0)
        connected = (base + base.transpose(0, 1)) > 0
        n = connected.shape[0]
        inf = torch.full((n, n), float("inf"), device=base.device, dtype=base.dtype)
        dist = torch.where(connected, torch.ones_like(inf), inf)
        dist.fill_diagonal_(0.0)
        for k in range(n):
            dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        finite = torch.isfinite(dist)
        max_dist = dist[finite].max().clamp_min(1.0) if bool(finite.any()) else dist.new_tensor(1.0)
        dist = torch.where(finite, dist / max_dist, torch.ones_like(dist))
        return dist

    def _cached_shortest_path(self, adj: torch.Tensor) -> torch.Tensor:
        version = int(adj._version)
        key = (version, adj.device, adj.dtype)
        if self._cache_key != key or self._cached_distance is None:
            with torch.no_grad():
                self._cached_distance = self._distance_from_adjacency(adj.detach())
                self._cache_key = key
        return self._cached_distance

    def _anchor_sets(self, observed_mask: torch.Tensor) -> list[torch.Tensor]:
        observed = torch.nonzero(observed_mask, as_tuple=False).flatten().cpu()
        if observed.numel() == 0:
            observed = torch.arange(self.num_nodes)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(observed.numel()))
        anchor_sets: list[torch.Tensor] = []
        n = int(observed.numel())
        for i in range(self.distortion):
            anchor_size = max(1, int(math.ceil(n / (2 ** (i + 1)))))
            for _ in range(self.anchor_repeats):
                perm = torch.randperm(n, generator=generator)[:anchor_size]
                anchor_sets.append(observed[perm])
        return anchor_sets

    def forward(
        self,
        adj: Optional[torch.Tensor],
        observed_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adj is None:
            frechet = F.softplus(self.learnable_anchor_distance).to(device=device, dtype=dtype)
            frechet = frechet / frechet.max().clamp_min(1e-6)
            prior = torch.softmax(-torch.cdist(frechet, frechet), dim=-1)
            return frechet, prior

        distance = self._cached_shortest_path(adj.to(device=device, dtype=dtype))
        observed = _as_node_mask(observed_mask, self.num_nodes, device)
        columns = []
        for anchors_cpu in self._anchor_sets(observed):
            anchors = anchors_cpu.to(device=device)
            columns.append(distance[:, anchors].min(dim=-1).values)
        frechet = torch.stack(columns, dim=-1)
        frechet = frechet / frechet.max().clamp_min(1e-6)
        prior = torch.softmax(-torch.cdist(frechet, frechet), dim=-1)
        return frechet, prior


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
        graph_perturb_samples: int,
        graph_perturb_ratio: float,
        graph_prior_weight: float,
    ) -> None:
        super().__init__()
        self.t_diff = _GraphConvLayer(sem_input_dim, sem_output_dim, ks_t)
        self.s_conv = _GraphConvLayer(x_input_dim, x_output_dim, ks_s)
        self.t_adpadj = _AdaptiveInteraction(x_input_dim, adp_s_dim, node_num)
        self.s_adpadj = _AdaptiveInteraction(sem_input_dim, adp_t_dim, node_num)
        self.graph_prior_weight = float(graph_prior_weight)
        self.sem_editor = _GraphEditorDeleteRow(graph_perturb_samples, node_num, graph_perturb_ratio)
        self.temporal_editor = _GraphEditorDeleteRow(graph_perturb_samples, node_num, graph_perturb_ratio)
        self.bn_sem = nn.BatchNorm1d(sem_output_dim)
        self.bn_x = nn.BatchNorm1d(x_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sem: torch.Tensor,
        spatial_prior: Optional[torch.Tensor],
        temporal_prior: Optional[torch.Tensor],
        use_graph_perturbation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        zero = x.new_zeros(())
        sem_mask = None
        temporal_mask = None
        log_p_sem = zero
        log_p_temporal = zero
        if self.training and use_graph_perturbation:
            sem_mask, log_p_sem = self.sem_editor()
            temporal_mask, log_p_temporal = self.temporal_editor()

        sem_adpadj = self.t_adpadj(
            x,
            perturb_mask=temporal_mask,
            prior=temporal_prior,
            prior_weight=self.graph_prior_weight,
        )
        x_adpadj = self.s_adpadj(
            sem,
            perturb_mask=sem_mask,
            prior=spatial_prior,
            prior_weight=self.graph_prior_weight,
        )
        x = self.s_conv(x, x_adpadj)
        sem = self.t_diff(sem, sem_adpadj)
        x = self.dropout(self.bn_x(F.relu(x).transpose(1, 2)).transpose(1, 2))
        sem = self.dropout(self.bn_sem(F.relu(sem).transpose(1, 2)).transpose(1, 2))
        aux = {
            "stone_graph_perturb_log_p": log_p_sem + log_p_temporal,
            "stone_spatial_graph_entropy": -(x_adpadj * x_adpadj.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "stone_temporal_graph_entropy": -(sem_adpadj * sem_adpadj.clamp_min(1e-8).log()).sum(dim=-1).mean(),
        }
        return x, sem, aux


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

    The module follows the official STONE path: Frechet-style spatial side
    information, semantic attention `STBlock`, gated temporal convolutions,
    adaptive spatial/temporal semantic graphs, graph-editor perturbation, graph
    aggregation `STAggBlock`, and `GatedFusionBlock`. PEMS08 still lacks the
    official road-coordinate/OOD split protocol, so side information falls back
    to adjacency shortest-path anchor distances with a learnable-distance final
    fallback only when no adjacency reaches the backbone.
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
        anchor_repeats: int = 4,
        side_info_seed: int = 2026,
        graph_prior_weight: float = 0.2,
        use_graph_perturbation: bool = True,
        graph_perturb_samples: int = 3,
        graph_perturb_ratio: float = 0.2,
        lambda_graph_perturb: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        self.use_graph_perturbation = bool(use_graph_perturbation)
        self.lambda_graph_perturb = float(lambda_graph_perturb)
        s_blocks = [[hidden_dim, 16, hidden_dim] for _ in range(int(SBlocks_num))]
        has_shallow_encode = [idx == 0 for idx in range(int(SBlocks_num))]
        t_blocks = [[input_dim, temporal_channels]]
        t_blocks.extend([[temporal_channels, temporal_channels] for _ in range(max(0, int(TBlocks_num) - 1))])
        temporal_total = self._temporal_total_length(input_len, len(t_blocks), int(Kt))
        self.side_info = _SpatialSideInformation(num_nodes, anchor_repeats, side_info_seed)
        self.side_info_proj = nn.Sequential(
            nn.Linear(self.side_info.side_dim, sem_dim),
            nn.ReLU(),
            nn.Linear(sem_dim, sem_dim),
        )
        self.semantic_residual = nn.Parameter(torch.randn(num_nodes, sem_dim) * 0.02)
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
            graph_perturb_samples=int(graph_perturb_samples),
            graph_perturb_ratio=float(graph_perturb_ratio),
            graph_prior_weight=float(graph_prior_weight),
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

    @staticmethod
    def _temporal_prior(x_repr: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(x_repr, dim=-1)
        sim = torch.einsum("bid,bjd->bij", normalized, normalized).clamp_min(0.0)
        return torch.softmax(sim, dim=-1)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self._check_input(x)
        batch_size = x.shape[0]
        observed_mask = kwargs.get("observed_mask", kwargs.get("node_subset_mask"))
        frechet, spatial_prior = self.side_info(adj, observed_mask, x.device, x.dtype)
        sem_base = self.side_info_proj(frechet.float()).to(dtype=x.dtype)
        sem = (sem_base + self.semantic_residual.to(device=x.device, dtype=x.dtype)).unsqueeze(0)
        sem = sem.expand(batch_size, -1, -1)
        x_temporal, sem = self.stblock(x.permute(0, 3, 1, 2), sem)
        x_temporal = self.x1(x_temporal.transpose(2, 3))
        x_temporal = F.relu(x_temporal)
        x_temporal = self.x2(x_temporal).transpose(2, 3)
        x_repr = x_temporal.permute(0, 3, 1, 2).squeeze(-1)
        temporal_prior = self._temporal_prior(x_repr)
        spatial_prior = spatial_prior.unsqueeze(0).expand(batch_size, -1, -1)
        x_repr, sem, stagg_aux = self.stagg(
            x_repr,
            sem,
            spatial_prior=spatial_prior,
            temporal_prior=temporal_prior,
            use_graph_perturbation=self.use_graph_perturbation,
        )
        y_inv, hidden = self.gatefusion(x_repr, sem)
        z_inv = self.representation_proj(hidden)
        self._assert_outputs(z_inv, y_inv, batch_size)
        graph_perturb_loss = -stagg_aux["stone_graph_perturb_log_p"] / float(
            max(1, self.num_nodes * 2)
        )
        return {
            "z_inv": z_inv,
            "y_inv": y_inv,
            "backbone_aux_losses": {
                "stone_graph_perturb_loss": graph_perturb_loss,
                "stone_spatial_graph_entropy": stagg_aux["stone_spatial_graph_entropy"],
                "stone_temporal_graph_entropy": stagg_aux["stone_temporal_graph_entropy"],
            },
            "backbone_aux_weights": {
                "stone_graph_perturb_loss": self.lambda_graph_perturb,
                "stone_spatial_graph_entropy": 0.0,
                "stone_temporal_graph_entropy": 0.0,
            },
        }
