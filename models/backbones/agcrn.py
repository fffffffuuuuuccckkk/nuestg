from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from models.backbones.base import BaseBackbone


class AVWGCN(nn.Module):
    """AGCRN adaptive graph convolution with node-specific weight pools.

    Reference: /data/OuXiaoyu/mystg/baselines/AGCRN/model/AGCN.py
    """

    def __init__(self, input_dim: int, output_dim: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.cheb_k = max(1, int(cheb_k))
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, self.cheb_k, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        node_num = node_embeddings.shape[0]
        supports = torch.softmax(torch.relu(node_embeddings.matmul(node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num, dtype=x.dtype, device=x.device), supports.to(dtype=x.dtype, device=x.device)]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set[: self.cheb_k], dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    """Official AGCRN recurrent cell.

    Reference: /data/OuXiaoyu/mystg/baselines/AGCRN/model/AGCRNCell.py
    """

    def __init__(self, input_dim: int, hidden_dim: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AVWGCN(input_dim + hidden_dim, 2 * hidden_dim, cheb_k, embed_dim)
        self.update = AVWGCN(input_dim + hidden_dim, hidden_dim, cheb_k, embed_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_gate, r_gate = torch.sigmoid(self.gate(gate_input, node_embeddings)).chunk(2, dim=-1)
        cand_input = torch.cat([x_t, z_gate * h_prev], dim=-1)
        candidate = torch.tanh(self.update(cand_input, node_embeddings))
        return r_gate * h_prev + (1.0 - r_gate) * candidate


class AGCRNBackbone(BaseBackbone):
    """Faithful native AGCRN backbone adapted to the local BaseBackbone API.

    Reference files:
      - /data/OuXiaoyu/mystg/baselines/AGCRN/model/AGCRN.py
      - /data/OuXiaoyu/mystg/baselines/AGCRN/model/AGCN.py
      - /data/OuXiaoyu/mystg/baselines/AGCRN/model/AGCRNCell.py

    Adaptation: the official predictor is kept for y_inv; an additional
    projection exposes z_inv for NUE-STG's shared backbone interface.
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
            cells.append(AGCRNCell(layer_input_dim, hidden_dim, self.cheb_k, embed_dim))
        self.cells = nn.ModuleList(cells)
        self.representation_proj = nn.Linear(hidden_dim, representation_dim)
        self.end_conv = nn.Conv2d(1, output_len * output_dim, kernel_size=(1, hidden_dim), bias=True)
        self.inv_head = nn.Linear(representation_dim, output_len * output_dim)

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del adj, kwargs
        x = self._check_input(x)
        batch_size, input_len, num_nodes, _ = x.shape
        hidden_states = [
            x.new_zeros(batch_size, num_nodes, self.hidden_dim)
            for _ in range(self.num_layers)
        ]

        for t in range(input_len):
            layer_input = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                hidden_states[layer_idx] = cell(layer_input, hidden_states[layer_idx], self.node_embeddings)
                layer_input = self.dropout(hidden_states[layer_idx])

        z_inv = self.representation_proj(hidden_states[-1])
        output = self.end_conv(hidden_states[-1].unsqueeze(1))
        output = output.squeeze(-1).reshape(batch_size, self.output_len, self.output_dim, num_nodes)
        y_inv = output.permute(0, 1, 3, 2)
        self._assert_outputs(z_inv, y_inv, batch_size)
        return {"z_inv": z_inv, "y_inv": y_inv}
