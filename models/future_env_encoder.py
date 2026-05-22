from __future__ import annotations

import torch
from torch import nn

from utils.tensor_ops import ensure_blnc


class FutureEnvEncoder(nn.Module):
    """Encode future residuals into node-wise future environments for training only.

    The input is the detached invariant residual y_true - stopgrad(y_inv). This
    module is never used for prediction and should not be called in eval/test.
    """

    def __init__(
        self,
        output_len: int,
        output_dim: int,
        env_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.env_dim = int(env_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.output_len * self.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
        )

    def forward(self, future_residual: torch.Tensor) -> torch.Tensor:
        future_residual = ensure_blnc(future_residual, "future_residual")
        batch_size, output_len, num_nodes, output_dim = future_residual.shape
        if output_len != self.output_len:
            raise ValueError(f"expected output_len={self.output_len}, got {output_len}")
        if output_dim != self.output_dim:
            raise ValueError(f"expected output_dim={self.output_dim}, got {output_dim}")
        node_residual = future_residual.permute(0, 2, 1, 3).reshape(
            batch_size, num_nodes, output_len * output_dim
        )
        env_fut = self.encoder(node_residual)
        expected = (batch_size, num_nodes, self.env_dim)
        if tuple(env_fut.shape) != expected:
            raise AssertionError(f"env_fut must be {expected}, got {tuple(env_fut.shape)}")
        return env_fut
