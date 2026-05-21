from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from basicts.configs import BasicTSModelConfig

from models.backbones import build_backbone
from models.env_encoder import NodeWiseEnvironmentEncoder
from utils.tensor_ops import ensure_blnc, load_adjacency


@dataclass
class NUESTGConfig(BasicTSModelConfig):
    name: str = "NUESTG"
    input_len: int = 12
    output_len: int = 12
    num_nodes: int = 1
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    node_emb_dim: int = 32
    time_emb_dim: int = 0
    dropout: float = 0.1
    use_node_embedding: bool = True
    use_time_embedding: bool = False
    use_adj: bool = True
    adj_path: str = ""
    adj_norm: str = "sym"
    adaptive_adj: bool = False
    backbone_name: str = "stid_mlp"
    backbone: Dict = field(default_factory=dict)
    env_dim: int = 32
    env_hidden_dim: int = 64
    env_encoder_type: str = "temporal_mlp"
    env_use_neighbor: bool = True
    env_neighbor_mix: str = "static_adj"
    env_global_mode: bool = False
    env_dropout: float = 0.1
    env_logvar_min: float = -10.0
    env_logvar_max: float = 10.0
    env_reparameterize: bool = True
    deterministic_env_eval: bool = True
    gate_hidden_dim: int = 64
    gate_type: str = "node_horizon"
    gate_horizon_aware: bool = True
    gate_init_bias: float = -1.0
    gate_temperature: float = 1.0
    force_gate_value: Optional[float] = None
    residual_hidden_dim: int = 64
    residual_dropout: float = 0.1
    prediction_activation: Optional[str] = None
    use_shuffled_env_train: bool = False
    use_shuffled_env_eval: bool = False
    swap: Dict = field(default_factory=dict)
    swap_detach_inv: bool = True


class NUESTG(nn.Module):
    """NUE-STG: Node-wise Utility-aware Environment Learning.

    NUE-STG learns node-wise conditional environment utility. For each node v
    and time window t, the local environment E_{v,t} is considered useful only
    if it provides additional predictive information beyond invariant
    representation Z_{v,t}:

        I(Y_{v,t+h}; E_{v,t} | Z_{v,t}) > eta.

    Since this mutual information is hard to estimate, we approximate utility
    by potential prediction gain:

        Delta = loss(y_inv, Y) - loss(y_inv + r_env, Y).

    The invariant backbone is pluggable. STID-MLP, Graph WaveNet-style, and
    AGCRN-style backbones all return the same contract:
    z_inv [B,N,D] and y_inv [B,H,N,C_out]. The node-wise environment encoder,
    residual correction, utility gate, swapping regularizer, and loss interface
    stay unchanged across backbones.
    """

    def __init__(self, config: NUESTGConfig) -> None:
        super().__init__()
        self.config = config
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_nodes = config.num_nodes
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.env_dim = config.env_dim
        self.gate_horizon_aware = config.gate_horizon_aware
        self.gate_temperature = config.gate_temperature
        self.force_gate_value = config.force_gate_value
        self.use_shuffled_env_train = config.use_shuffled_env_train
        self.use_shuffled_env_eval = config.use_shuffled_env_eval
        self.swap_cfg = config.swap or {}
        self.swap_detach_inv = config.swap_detach_inv
        self.backbone_name = config.backbone_name

        if config.use_time_embedding:
            raise NotImplementedError("MODEL.use_time_embedding=True is not implemented in the current backbones.")
        if config.adaptive_adj:
            raise NotImplementedError(
                "MODEL.adaptive_adj=True is not implemented at the NUE-STG top level; "
                "use backbone-specific adaptive adjacency options instead."
            )
        if config.env_neighbor_mix not in (None, "static_adj"):
            raise NotImplementedError(
                f"MODEL.env_neighbor_mix={config.env_neighbor_mix!r} is not implemented; "
                "current EnvEncoder supports static_adj aggregation or self-only fallback."
            )
        if self.swap_cfg.get("mode", "batch_node_random") != "batch_node_random":
            raise NotImplementedError(
                f"SWAP.mode={self.swap_cfg.get('mode')!r} is not implemented; current swap is batch_node_random."
            )
        if self.swap_cfg.get("pair_mining", False):
            raise NotImplementedError("SWAP.pair_mining=True is not implemented yet.")
        if int(self.swap_cfg.get("num_swaps", 1)) != 1:
            raise NotImplementedError("SWAP.num_swaps other than 1 is not implemented.")

        model_cfg = asdict(config)
        self.backbone = build_backbone({"MODEL": model_cfg})
        self.representation_dim = int(self.backbone.representation_dim)
        self.hidden_dim = self.representation_dim

        if config.env_encoder_type != "temporal_mlp":
            raise ValueError("Only env_encoder_type='temporal_mlp' is implemented in this version.")

        self.env_encoder = NodeWiseEnvironmentEncoder(
            input_len=config.input_len,
            input_dim=config.input_dim,
            env_dim=config.env_dim,
            hidden_dim=config.env_hidden_dim,
            dropout=config.env_dropout,
            use_neighbor=config.env_use_neighbor,
            global_mode=config.env_global_mode,
            logvar_min=config.env_logvar_min,
            logvar_max=config.env_logvar_max,
            reparameterize=config.env_reparameterize,
            deterministic_eval=config.deterministic_env_eval,
        )

        decode_in_dim = self.representation_dim + config.env_dim
        self.env_head = nn.Sequential(
            nn.Linear(decode_in_dim, config.residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.residual_dropout),
            nn.Linear(config.residual_hidden_dim, config.output_len * config.output_dim),
        )
        gate_out_dim = config.output_len if config.gate_horizon_aware else 1
        self.gate_net = nn.Sequential(
            nn.Linear(decode_in_dim, config.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden_dim, gate_out_dim),
        )
        self._init_gate_bias(config.gate_init_bias)

        adj_norm = (
            load_adjacency(config.adj_path, config.num_nodes, config.adj_norm)
            if config.use_adj
            else None
        )
        if adj_norm is not None:
            self.register_buffer("adj_norm", adj_norm)
        else:
            self.adj_norm = None

    def _init_gate_bias(self, bias: float) -> None:
        last = self.gate_net[-1]
        if isinstance(last, nn.Linear) and bias is not None:
            nn.init.constant_(last.bias, bias)

    def _permute_env(self, env: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, env_dim = env.shape
        flat = env.reshape(batch_size * num_nodes, env_dim)
        perm = torch.randperm(batch_size * num_nodes, device=env.device)
        if self.swap_cfg.get("avoid_self", True) and flat.shape[0] > 1:
            same = perm == torch.arange(flat.shape[0], device=env.device)
            if same.any():
                perm[same] = (perm[same] + 1) % flat.shape[0]
        return flat[perm].reshape(batch_size, num_nodes, env_dim)

    def _apply_prediction_activation(self, y: torch.Tensor) -> torch.Tensor:
        activation = self.config.prediction_activation
        if activation is None:
            return y
        if activation == "relu":
            return torch.relu(y)
        if activation == "softplus":
            return torch.nn.functional.softplus(y)
        raise ValueError(f"Unsupported prediction_activation={activation!r}")

    def decode_with_env(
        self,
        z_inv: torch.Tensor,
        env: torch.Tensor,
        y_inv: Optional[torch.Tensor] = None,
        detach_inv: bool = False,
        force_gate_value: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_inv.dim() != 3:
            raise AssertionError(f"z_inv must be [B, N, D], got {tuple(z_inv.shape)}")
        if env.dim() != 3:
            raise AssertionError(f"env must be [B, N, D_env], got {tuple(env.shape)}")
        if z_inv.shape[:2] != env.shape[:2]:
            raise AssertionError(f"z_inv/env batch-node shape mismatch: {tuple(z_inv.shape)} vs {tuple(env.shape)}")

        if detach_inv:
            z_decode = z_inv.detach()
            y_base = self.backbone.forecast_from_representation(z_decode) if y_inv is None else y_inv.detach()
        else:
            z_decode = z_inv
            y_base = self.backbone.forecast_from_representation(z_decode) if y_inv is None else y_inv

        decode_input = torch.cat([z_decode, env], dim=-1)
        batch_size, num_nodes, _ = decode_input.shape
        r_env = self.env_head(decode_input)
        r_env = r_env.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)

        expected_pred_shape = (batch_size, self.output_len, num_nodes, self.output_dim)
        if tuple(y_base.shape) != expected_pred_shape:
            raise AssertionError(f"y_inv must be {expected_pred_shape}, got {tuple(y_base.shape)}")

        gate_value = self.force_gate_value if force_gate_value is None else force_gate_value
        if gate_value is None:
            gate_logits = self.gate_net(decode_input) / max(float(self.gate_temperature), 1e-6)
            if self.gate_horizon_aware:
                rho = torch.sigmoid(gate_logits).view(batch_size, num_nodes, self.output_len, 1).permute(0, 2, 1, 3)
            else:
                rho = torch.sigmoid(gate_logits).view(batch_size, num_nodes, 1, 1).permute(0, 2, 1, 3)
                rho = rho.expand(-1, self.output_len, -1, -1)
        else:
            rho = torch.full(
                (batch_size, self.output_len, num_nodes, 1),
                float(gate_value),
                dtype=r_env.dtype,
                device=r_env.device,
            )

        y_potential = y_base + r_env
        prediction = y_base + rho * r_env
        prediction = self._apply_prediction_activation(prediction)
        y_potential = self._apply_prediction_activation(y_potential)
        return {
            "r_env": r_env,
            "rho": rho,
            "prediction": prediction,
            "y_potential": y_potential,
        }

    def forward(self, inputs: torch.Tensor, **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        x = ensure_blnc(inputs, "inputs")
        batch_size, _, num_nodes, _ = x.shape
        adj = getattr(self, "adj_norm", None)

        backbone_out = self.backbone(x, adj=adj)
        z_inv = backbone_out["z_inv"]
        y_inv = self._apply_prediction_activation(backbone_out["y_inv"])
        env_mu, env_logvar, env = self.env_encoder(x, adj)

        env_perm = None
        use_shuffled_env = (self.training and self.use_shuffled_env_train) or (
            (not self.training) and self.use_shuffled_env_eval
        )
        env_decode = env
        if use_shuffled_env:
            env_perm = self._permute_env(env)
            env_decode = env_perm

        decoded = self.decode_with_env(z_inv, env_decode, y_inv=y_inv, detach_inv=False)
        prediction = decoded["prediction"]
        rho = decoded["rho"]
        r_env = decoded["r_env"]
        y_potential = decoded["y_potential"]

        expected_pred_shape = (batch_size, self.output_len, num_nodes, self.output_dim)
        expected_gate_shape = (batch_size, self.output_len, num_nodes, 1)
        expected_z_shape = (batch_size, num_nodes, self.representation_dim)
        expected_env_shape = (batch_size, num_nodes, self.env_dim)
        shape_checks = {
            "prediction": (prediction, expected_pred_shape),
            "y_inv": (y_inv, expected_pred_shape),
            "y_potential": (y_potential, expected_pred_shape),
            "r_env": (r_env, expected_pred_shape),
            "rho": (rho, expected_gate_shape),
            "z_inv": (z_inv, expected_z_shape),
            "env_mu": (env_mu, expected_env_shape),
            "env_logvar": (env_logvar, expected_env_shape),
            "env": (env, expected_env_shape),
        }
        for name, (tensor, expected_shape) in shape_checks.items():
            if tuple(tensor.shape) != expected_shape:
                raise AssertionError(f"{name} must be {expected_shape}, got {tuple(tensor.shape)}")
        if prediction.shape != y_inv.shape:
            raise AssertionError(
                f"prediction and y_inv must share shape, got {tuple(prediction.shape)} and {tuple(y_inv.shape)}"
            )

        prediction_swap = None
        rho_swap = None
        if self.training and self.swap_cfg.get("enabled", True):
            if env_perm is None:
                env_perm = self._permute_env(env)
            swap_decoded = self.decode_with_env(
                z_inv,
                env_perm,
                y_inv=y_inv,
                detach_inv=self.swap_detach_inv,
            )
            prediction_swap = swap_decoded["prediction"]
            rho_swap = swap_decoded["rho"]

        return {
            "prediction": prediction,
            "y_inv": y_inv,
            "y_potential": y_potential,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "env_mu": env_mu,
            "env_logvar": env_logvar,
            "env": env,
            "prediction_swap": prediction_swap,
            "rho_swap": rho_swap,
            "env_perm": env_perm,
        }
