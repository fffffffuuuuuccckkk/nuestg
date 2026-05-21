from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from models.backbones.base import BaseBackbone


class AdaptiveGraphConv(nn.Module):
    """Stable AGCRN-style adaptive graph convolution.

    This first version uses node embeddings to build adaptive Chebyshev
    supports, aggregates each support, concatenates the results, and applies a
    shared linear projection. It keeps the AGCRN adaptive-graph recurrent shape
    while avoiding the heavier per-node weight pool from the official model.
    """

    def __init__(self, input_dim: int, output_dim: int, cheb_k: int) -> None:
        super().__init__()
        self.cheb_k = max(1, int(cheb_k))
        self.proj = nn.Linear(input_dim * self.cheb_k, output_dim)

    @staticmethod
    def nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnc,nm->bmc", x, support)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for support in supports[: self.cheb_k]:
            out.append(self.nconv(x, support))
        while len(out) < self.cheb_k:
            out.append(torch.zeros_like(x))
        return self.proj(torch.cat(out, dim=-1))


class AGCRNCell(nn.Module):
    """GRU-like recurrent cell using adaptive graph convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, cheb_k: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AdaptiveGraphConv(input_dim + hidden_dim, 2 * hidden_dim, cheb_k)
        self.update = AdaptiveGraphConv(input_dim + hidden_dim, hidden_dim, cheb_k)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_gate, r_gate = torch.sigmoid(self.gate(gate_input, supports)).chunk(2, dim=-1)
        cand_input = torch.cat([x_t, r_gate * h_prev], dim=-1)
        candidate = torch.tanh(self.update(cand_input, supports))
        return z_gate * h_prev + (1.0 - z_gate) * candidate


class AGCRNBackbone(BaseBackbone):
    """AGCRN-style adaptive graph recurrent backbone for NUE-STG.

    It uses node embeddings to construct an adaptive adjacency, applies
    Chebyshev-style adaptive graph convolutions inside GRU-like recurrent
    cells, and returns the invariant node representation and invariant forecast
    expected by NUE-STG. This is an AGCRN-style invariant branch, not a
    line-by-line copy of the official implementation.
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
        embed_dim: int = 10,
        num_layers: int = 1,
        cheb_k: int = 2,
        dropout: float = 0.1,
        use_static_adj: bool = False,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.num_layers = max(1, int(num_layers))
        self.cheb_k = max(1, int(cheb_k))
        self.use_static_adj = bool(use_static_adj)
        self.dropout = nn.Dropout(dropout)

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        cells = []
        for layer_idx in range(self.num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
            cells.append(AGCRNCell(layer_input_dim, hidden_dim, self.cheb_k))
        self.cells = nn.ModuleList(cells)
        self.representation_proj = nn.Linear(hidden_dim, representation_dim)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def _adaptive_adj(self) -> torch.Tensor:
        return torch.softmax(torch.relu(self.node_embeddings.matmul(self.node_embeddings.transpose(0, 1))), dim=1)

    def _supports(self, adj: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        adp = self._adaptive_adj().to(device=device, dtype=dtype)
        if self.use_static_adj and adj is not None:
            static = adj.to(device=device, dtype=dtype)
            adp = 0.5 * adp + 0.5 * static
            adp = adp / adp.sum(dim=1, keepdim=True).clamp_min(1e-6)
        supports = [torch.eye(self.num_nodes, device=device, dtype=dtype), adp]
        for _ in range(2, self.cheb_k):
            supports.append(torch.matmul(supports[-1], adp))
        return supports[: self.cheb_k]

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self._check_input(x)
        batch_size, input_len, num_nodes, _ = x.shape
        supports = self._supports(adj, x.device, x.dtype)
        hidden_states = [
            x.new_zeros(batch_size, num_nodes, self.hidden_dim)
            for _ in range(self.num_layers)
        ]

        for t in range(input_len):
            layer_input = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                hidden_states[layer_idx] = cell(layer_input, hidden_states[layer_idx], supports)
                layer_input = self.dropout(hidden_states[layer_idx])

        z_inv = self.representation_proj(hidden_states[-1])
        y_inv = self.forecast_from_representation(z_inv)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
