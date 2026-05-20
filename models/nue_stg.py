from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from basicts.configs import BasicTSModelConfig

from models.env_encoder import NodeWiseEnvironmentEncoder
from utils.tensor_ops import ensure_blnc, load_adjacency


@dataclass
class NUESTGConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    num_nodes: int = 170
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    env_dim: int = 32
    node_emb_dim: int = 32
    dropout: float = 0.1
    use_adj: bool = True
    adj_path: str = ""
    deterministic_env_eval: bool = True
    use_node_embedding: bool = True
    enable_swap: bool = True


class NUESTG(nn.Module):
    """NUE-STG: Node-wise Utility-aware Environment Learning.

    NUE-STG optimizes node-wise conditional environment utility. A local
    environment E_{v,t} is useful if it provides additional predictive
    information beyond invariant representation Z_{v,t}:

        I(Y_{v,t}; E_{v,t} | Z_{v,t}) > eta

    Since mutual information is hard to estimate, the loss approximates utility
    by prediction gain:

        Delta = loss(f_inv(Z), Y) - loss(f_inv(Z) + rho * R_env(Z,E), Y)

    The optimal binary gate opens when Delta exceeds a usage cost eta. KL limits
    I(E;X), the covariance penalty reduces redundancy I(E;Z), and sparse
    penalty prevents always using environment.
    """

    def __init__(self, config: NUESTGConfig) -> None:
        super().__init__()
        self.config = config
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_nodes = config.num_nodes
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.hidden_dim = config.hidden_dim
        self.env_dim = config.env_dim
        self.enable_swap = config.enable_swap

        self.temporal_encoder = nn.Sequential(
            nn.Linear(config.input_len * config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        if config.use_node_embedding:
            self.node_emb = nn.Parameter(torch.empty(config.num_nodes, config.node_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
            inv_input_dim = config.hidden_dim + config.node_emb_dim
        else:
            self.node_emb = None
            inv_input_dim = config.hidden_dim

        self.inv_projector = nn.Sequential(
            nn.Linear(inv_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.inv_head = nn.Linear(config.hidden_dim, config.output_len * config.output_dim)

        self.env_encoder = NodeWiseEnvironmentEncoder(
            input_len=config.input_len,
            input_dim=config.input_dim,
            env_dim=config.env_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        decode_in_dim = config.hidden_dim + config.env_dim
        self.env_head = nn.Sequential(
            nn.Linear(decode_in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_len * config.output_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(decode_in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_len),
        )

        adj_norm = load_adjacency(config.adj_path, config.num_nodes) if config.use_adj else None
        if adj_norm is None:
            self.adj_norm = None
        else:
            self.register_buffer("adj_norm", adj_norm)

    def encode_invariant(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_blnc(x, "x")
        batch_size, input_len, num_nodes, input_dim = x.shape
        if input_len != self.input_len:
            raise ValueError(f"expected input_len={self.input_len}, got {input_len}")
        if num_nodes != self.num_nodes:
            raise ValueError(f"expected num_nodes={self.num_nodes}, got {num_nodes}")
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")

        node_history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, input_len * input_dim)
        h = self.temporal_encoder(node_history)
        if self.node_emb is not None:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            h = torch.cat([h, node_emb], dim=-1)
        z_inv = self.inv_projector(h)
        return z_inv

    def invariant_predict(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head(z_inv)
        return y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

    def decode_with_env(
        self,
        z_inv: torch.Tensor,
        env: torch.Tensor,
        y_inv: Optional[torch.Tensor] = None,
        detach_inv: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if detach_inv:
            z_decode = z_inv.detach()
            y_base = self.invariant_predict(z_decode) if y_inv is None else y_inv.detach()
        else:
            z_decode = z_inv
            y_base = self.invariant_predict(z_decode) if y_inv is None else y_inv

        decode_input = torch.cat([z_decode, env], dim=-1)
        batch_size, num_nodes, _ = decode_input.shape
        r_env = self.env_head(decode_input)
        r_env = r_env.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        rho = torch.sigmoid(self.gate_net(decode_input))
        rho = rho.view(batch_size, num_nodes, self.output_len, 1).permute(0, 2, 1, 3)
        prediction = y_base + rho * r_env
        return {"r_env": r_env, "rho": rho, "prediction": prediction}

    def forward(self, inputs: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        x = ensure_blnc(inputs, "inputs")
        batch_size, _, num_nodes, _ = x.shape

        z_inv = self.encode_invariant(x)
        y_inv = self.invariant_predict(z_inv)
        env_mu, env_logvar, env = self.env_encoder(x, getattr(self, "adj_norm", None))
        decoded = self.decode_with_env(z_inv, env, y_inv=y_inv, detach_inv=False)

        prediction = decoded["prediction"]
        rho = decoded["rho"]
        r_env = decoded["r_env"]

        if not (env.dim() == 3 and env.shape[:2] == (batch_size, num_nodes)):
            raise AssertionError(f"env must be [B, N, D_env], got {tuple(env.shape)}")
        expected_gate_shape = (batch_size, self.output_len, num_nodes, 1)
        if tuple(rho.shape) != expected_gate_shape:
            raise AssertionError(f"rho must be {expected_gate_shape}, got {tuple(rho.shape)}")
        if prediction.shape != y_inv.shape:
            raise AssertionError(
                f"prediction and y_inv must share shape, got {tuple(prediction.shape)} and {tuple(y_inv.shape)}"
            )

        output = {
            "prediction": prediction,
            "y_inv": y_inv,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "env_mu": env_mu,
            "env_logvar": env_logvar,
            "env": env,
        }

        if self.training and self.enable_swap:
            env_flat = env.reshape(batch_size * num_nodes, self.env_dim)
            perm = torch.randperm(batch_size * num_nodes, device=env.device)
            env_perm = env_flat[perm].reshape(batch_size, num_nodes, self.env_dim)
            swap_decoded = self.decode_with_env(z_inv, env_perm, y_inv=y_inv, detach_inv=True)
            output["prediction_swap"] = swap_decoded["prediction"]
            output["rho_swap"] = swap_decoded["rho"]

        return output
