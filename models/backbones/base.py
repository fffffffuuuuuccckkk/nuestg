from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import nn

from utils.tensor_ops import ensure_blnc


class BaseBackbone(nn.Module, ABC):
    """Common interface for NUE-STG invariant forecasting backbones."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.representation_dim = int(representation_dim)

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_blnc(x, "x")
        batch_size, input_len, num_nodes, input_dim = x.shape
        if input_len != self.input_len:
            raise ValueError(f"expected input_len={self.input_len}, got {input_len}")
        if num_nodes != self.num_nodes:
            raise ValueError(f"expected num_nodes={self.num_nodes}, got {num_nodes}")
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")
        return x

    def _assert_outputs(self, z_inv: torch.Tensor, y_inv: torch.Tensor, batch_size: int) -> None:
        expected_z = (batch_size, self.num_nodes, self.representation_dim)
        expected_y = (batch_size, self.output_len, self.num_nodes, self.output_dim)
        if tuple(z_inv.shape) != expected_z:
            raise AssertionError(f"z_inv must be {expected_z}, got {tuple(z_inv.shape)}")
        if tuple(y_inv.shape) != expected_y:
            raise AssertionError(f"y_inv must be {expected_y}, got {tuple(y_inv.shape)}")

    @abstractmethod
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Return {"z_inv": [B,N,D], "y_inv": [B,H,N,C_out]}."""

    @abstractmethod
    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        """Decode y_inv from z_inv for shared NUE-STG environment decoding."""
