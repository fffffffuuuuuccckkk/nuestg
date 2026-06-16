from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from basicts.configs import BasicTSModelConfig

from models.backbones import build_backbone
from models.env_future_decoder import FutureEnvDistributionDecoder
from models.env_encoder import NodeWiseEnvironmentEncoder, TimeNodeEnvironmentEncoder
from models.env_mask import FuturePredictiveEnvMask
from models.future_env_encoder import FutureEnvEncoder
from models.separation import SeparationModule
from models.st_perturbation import STPerturbation
from models.time_embedding import TimestampEncoder
from utils.tensor_ops import align_target, ensure_blnc, load_adjacency, load_graph_supports


@dataclass
class NUESTGConfig(BasicTSModelConfig):
    name: str = "NUESTG"
    baseline_name: str = ""
    reference_status: str = "native_adapter"
    input_len: int = 12
    output_len: int = 12
    num_nodes: int = 1
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    node_emb_dim: int = 32
    time_emb_dim: int = 0
    tod_emb_dim: int = 16
    dow_emb_dim: int = 8
    num_time_in_day: int = 288
    num_day_in_week: int = 7
    timestamp_feature_dim: int = 0
    dropout: float = 0.1
    use_node_embedding: bool = True
    use_time_embedding: bool = False
    use_timestamp: bool = False
    time_encoding_type: str = "none"
    use_time_of_day: bool = True
    use_day_of_week: bool = True
    use_current_timestamp_for_z: bool = True
    use_current_timestamp_for_env: bool = True
    required_timestamp: bool = False
    use_adj: bool = True
    adj_path: str = ""
    adj_norm: str = "sym"
    adaptive_adj: bool = False
    backbone_name: str = "stid_mlp"
    backbone: Dict = field(default_factory=dict)
    GWNET: Dict = field(default_factory=dict)
    STNORM: Dict = field(default_factory=dict)
    external_path: str = ""
    official_requires_special_data: bool = False
    unsupported_reason: str = ""
    method_variant: str = "nue"
    separation: Dict = field(default_factory=dict)
    use_separated_z_for_y_inv: bool = True
    persistence: Dict = field(default_factory=dict)
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
    env_token_mode: bool = False
    mask_hidden_dim: int = 64
    mask_dropout: float = 0.1
    mask_init_bias: float = -1.0
    mask_temperature: float = 1.0
    mask_pooling: str = "masked_mean"
    mask_eps: float = 1e-6
    force_mask_value: Optional[float] = None
    mask_use_time: bool = True
    fusion_type: str = "film"
    fusion_hidden_dim: int = 64
    fusion_dropout: float = 0.1
    fusion_zero_init: bool = True
    env_fusion_scale: float = 0.1
    env_transition_hidden_dim: int = 64
    env_transition_dropout: float = 0.1
    future_decoder_hidden_dim: int = 64
    future_decoder_dropout: float = 0.1
    future_decoder_use_time: bool = True
    future_decoder_logvar_min: float = -8.0
    future_decoder_logvar_max: float = 4.0
    use_shuffled_env_train: bool = False
    use_shuffled_env_eval: bool = False
    perturb_enabled: bool = False
    perturb_prob: float = 1.0
    perturb_value_jitter: bool = True
    perturb_jitter_std: float = 0.01
    perturb_value_scale: bool = True
    perturb_scale_min: float = 0.9
    perturb_scale_max: float = 1.1
    perturb_time_node_mask: bool = True
    perturb_time_node_mask_ratio: float = 0.1
    perturb_mask_value: str = "zero"
    perturb_temporal_block: bool = True
    perturb_temporal_block_ratio: float = 0.1
    perturb_temporal_block_len: int = 2
    perturb_edge_dropout: bool = False
    perturb_edge_dropout_p: float = 0.1
    perturb_edge_dropout_for_env_only: bool = True
    swap: Dict = field(default_factory=dict)
    swap_detach_inv: bool = True
    swap_detach_env: bool = False
    z_inv_bottleneck: Dict = field(default_factory=dict)
    pseudo_env: Dict = field(default_factory=dict)
    env_routed_inv_heads: Dict = field(default_factory=dict)


class LatentFusion(nn.Module):
    """Fuse invariant Z and selected environment in latent space."""

    def __init__(
        self,
        z_dim: int,
        env_dim: int,
        fusion_type: str = "film",
        hidden_dim: int = 64,
        dropout: float = 0.1,
        zero_init: bool = True,
        env_fusion_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.env_dim = int(env_dim)
        self.fusion_type = str(fusion_type or "film").lower()
        self.env_fusion_scale = float(env_fusion_scale)
        if self.fusion_type in {"film", "weak_film"}:
            self.net = nn.Sequential(
                nn.Linear(env_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2 * z_dim),
            )
        elif self.fusion_type == "concat":
            self.net = nn.Sequential(
                nn.Linear(z_dim + env_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, z_dim),
            )
        elif self.fusion_type == "gated_add":
            self.gate = nn.Sequential(nn.Linear(env_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, z_dim))
            self.delta = nn.Sequential(nn.Linear(env_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, z_dim))
        elif self.fusion_type == "env_residual":
            self.delta = nn.Sequential(
                nn.Linear(env_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, z_dim),
            )
        else:
            raise NotImplementedError(
                f"MODEL.fusion_type={fusion_type!r} is not implemented; "
                "expected film, concat, gated_add, weak_film, or env_residual."
            )
        if zero_init:
            modules = []
            if hasattr(self, "net"):
                modules.append(self.net[-1])
            if hasattr(self, "gate"):
                modules.append(self.gate[-1])
            if hasattr(self, "delta"):
                modules.append(self.delta[-1])
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, z_inv: torch.Tensor, env: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_inv.dim() != 3:
            raise AssertionError(f"z_inv must be [B,N,D_z], got {tuple(z_inv.shape)}")
        if env.dim() != 3:
            raise AssertionError(f"env must be [B,N,D_env], got {tuple(env.shape)}")
        if z_inv.shape[:2] != env.shape[:2]:
            raise AssertionError(f"z/env batch-node mismatch: {tuple(z_inv.shape)} vs {tuple(env.shape)}")
        if z_inv.shape[-1] != self.z_dim or env.shape[-1] != self.env_dim:
            raise AssertionError(f"expected z/env dims {(self.z_dim, self.env_dim)}, got {(z_inv.shape[-1], env.shape[-1])}")
        if self.fusion_type == "film":
            gamma, beta = self.net(env).chunk(2, dim=-1)
            hidden = (1.0 + gamma) * z_inv + beta
            return {"hidden": hidden, "fusion_gamma": gamma, "fusion_beta": beta}
        if self.fusion_type == "weak_film":
            gamma, beta = self.net(env).chunk(2, dim=-1)
            scale = self.env_fusion_scale
            gamma_scaled = scale * torch.tanh(gamma)
            beta_scaled = scale * torch.tanh(beta)
            hidden = (1.0 + gamma_scaled) * z_inv + beta_scaled
            return {"hidden": hidden, "fusion_gamma": gamma_scaled, "fusion_beta": beta_scaled}
        if self.fusion_type == "concat":
            hidden = self.net(torch.cat([z_inv, env], dim=-1))
            zeros = torch.zeros_like(z_inv)
            return {"hidden": hidden, "fusion_gamma": zeros, "fusion_beta": zeros}
        if self.fusion_type == "env_residual":
            delta = self.delta(env)
            hidden = z_inv + self.env_fusion_scale * delta
            zeros = torch.zeros_like(z_inv)
            return {"hidden": hidden, "fusion_gamma": zeros, "fusion_beta": delta}
        gate = torch.sigmoid(self.gate(env))
        delta = self.delta(env)
        hidden = z_inv + gate * delta
        return {"hidden": hidden, "fusion_gamma": gate, "fusion_beta": delta}


class PseudoEnvHeads(nn.Module):
    """Competing prediction heads that read environment representations only."""

    def __init__(
        self,
        env_dim: int,
        output_len: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.env_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.output_len * self.output_dim),
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        if env.dim() != 3:
            raise AssertionError(f"pseudo env heads expect env [B,N,D], got {tuple(env.shape)}")
        if env.shape[-1] != self.env_dim:
            raise AssertionError(f"pseudo env dim mismatch: expected {self.env_dim}, got {env.shape[-1]}")
        preds = []
        batch_size, num_nodes, _ = env.shape
        for head in self.heads:
            pred = head(env)
            pred = pred.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
            preds.append(pred)
        return torch.stack(preds, dim=1)


class EnvRoutedInvariantHeads(nn.Module):
    """Invariant heads selected by an environment-only router.

    Heads consume only H_inv/Fuse(Z, 0). The router consumes only E/env_plus and
    produces mixture weights. No environment residual correction is generated.
    """

    def __init__(
        self,
        inv_dim: int,
        env_dim: int,
        output_len: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        tau: float = 1.0,
        mode: str = "confidence_mix",
        alpha_detach: bool = False,
    ) -> None:
        super().__init__()
        self.inv_dim = int(inv_dim)
        self.env_dim = int(env_dim)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.tau = max(float(tau), 1e-6)
        self.mode = str(mode or "confidence_mix").lower()
        self.alpha_detach = bool(alpha_detach)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.inv_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), self.output_len * self.output_dim),
            )
            for _ in range(self.num_heads)
        ])
        self.router = nn.Sequential(
            nn.Linear(self.env_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_heads),
        )

    @staticmethod
    def _pool_env(env: torch.Tensor) -> torch.Tensor:
        if env.dim() == 2:
            return env
        if env.dim() < 2:
            raise AssertionError(f"router env input must keep feature dim, got {tuple(env.shape)}")
        reduce_dims = tuple(range(1, env.dim() - 1))
        return env.mean(dim=reduce_dims) if reduce_dims else env

    def _predict_heads(self, h_inv: torch.Tensor) -> torch.Tensor:
        if h_inv.dim() != 3:
            raise AssertionError(f"invariant head input must be [B,N,D], got {tuple(h_inv.shape)}")
        batch_size, num_nodes, inv_dim = h_inv.shape
        if inv_dim != self.inv_dim:
            raise AssertionError(f"invariant head dim expected {self.inv_dim}, got {inv_dim}")
        preds = []
        for head in self.heads:
            pred = head(h_inv)
            pred = pred.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def forward(self, h_inv: torch.Tensor, env: torch.Tensor, y_inv: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_heads = self._predict_heads(h_inv)
        y_global = y_inv
        env_pooled = self._pool_env(env)
        if env_pooled.dim() != 2 or env_pooled.shape[-1] != self.env_dim:
            raise AssertionError(f"router env pooled input must be [B,{self.env_dim}], got {tuple(env_pooled.shape)}")
        logits = self.router(env_pooled)
        if self.mode == "uniform":
            q = torch.full_like(logits, 1.0 / self.num_heads)
        elif self.mode == "random":
            random_index = torch.randint(self.num_heads, (logits.shape[0],), device=logits.device)
            q = torch.nn.functional.one_hot(random_index, num_classes=self.num_heads).to(dtype=logits.dtype)
        else:
            q = torch.softmax(logits / self.tau, dim=-1)
        entropy = -(q * (q + 1e-8).log()).sum(dim=-1)
        if self.num_heads > 1:
            entropy_norm = entropy / q.new_tensor(float(self.num_heads)).log()
        else:
            entropy_norm = torch.zeros_like(entropy)
        alpha = (1.0 - entropy_norm).clamp(0.0, 1.0)
        alpha_for_mix = alpha.detach() if self.alpha_detach else alpha
        y_soft = (q.view(q.shape[0], self.num_heads, 1, 1, 1) * y_heads).sum(dim=1)
        hard_index = q.argmax(dim=-1)
        batch_index = torch.arange(q.shape[0], device=q.device)
        y_hard = y_heads[batch_index, hard_index]
        confidence = q.max(dim=-1).values
        if self.mode == "soft":
            y_route = y_soft
            y_final = y_route
        elif self.mode == "hard":
            y_route = y_hard
            y_final = y_route
        elif self.mode == "confidence_mix":
            y_route = y_soft
            alpha_view = alpha_for_mix.view(-1, 1, 1, 1).to(dtype=y_global.dtype)
            y_final = (1.0 - alpha_view) * y_global + alpha_view * y_route
        elif self.mode in {"uniform", "random"}:
            y_route = y_soft
            y_final = y_route
        else:
            raise ValueError("env_route_mode must be one of: soft, hard, confidence_mix, uniform, random")
        return {
            "y_route_heads": y_heads,
            "env_route_logits": logits,
            "env_route_q": q,
            "env_route_entropy": entropy,
            "env_route_alpha": alpha,
            "y_route_soft": y_soft,
            "y_route_hard": y_hard,
            "y_route": y_route,
            "y_route_selected": y_route,
            "y_global": y_global,
            "y_route_final": y_final,
            "route_confidence": confidence,
        }


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

    The invariant backbone is pluggable. STID-MLP, Graph WaveNet-style/full,
    and AGCRN-style backbones all return the same contract:
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
        self.perturb_enabled = bool(config.perturb_enabled)
        self.swap_cfg = config.swap or {}
        self.swap_detach_inv = bool(config.swap_detach_inv)
        self.swap_detach_env = bool(config.swap_detach_env)
        self.backbone_name = config.backbone_name
        self.baseline_name = config.baseline_name or config.name or config.backbone_name
        self.reference_status = config.reference_status
        self.method_variant = str(config.method_variant or "nue").lower()
        self.is_fpem = self.method_variant == "fpem"
        self.use_separated_z_for_y_inv = bool(config.use_separated_z_for_y_inv)
        self.persistence_cfg = config.persistence or {}
        self.persistence_enabled = bool(self.persistence_cfg.get("enabled", False))
        self.use_timestamp = bool(config.use_timestamp)
        self.time_emb_dim = int(config.time_emb_dim) if self.use_timestamp else 0
        self.use_current_timestamp_for_z = bool(config.use_current_timestamp_for_z)
        self.use_current_timestamp_for_env = bool(config.use_current_timestamp_for_env)
        if self.method_variant not in {"nue", "nuestg", "fpem"}:
            raise NotImplementedError(
                f"MODEL.method_variant={config.method_variant!r} is not implemented; expected 'nue' or 'fpem'."
            )

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
        self.z_inv_bottleneck_cfg = dict(config.z_inv_bottleneck or {})
        self.z_inv_ib_enabled = bool(self.z_inv_bottleneck_cfg.get("enabled", False))
        self.z_inv_ib_type = str(self.z_inv_bottleneck_cfg.get("type", "vib") or "vib").lower()
        self.z_inv_ib_apply_to = str(self.z_inv_bottleneck_cfg.get("apply_to", "z_inv") or "z_inv").lower()
        self.z_inv_ib_noise_std = float(self.z_inv_bottleneck_cfg.get("noise_std", 0.05))
        self.z_inv_ib_predict_from_sampled_z = bool(
            self.z_inv_bottleneck_cfg.get("predict_from_sampled_z", True)
        )
        self.pseudo_env_cfg = dict(config.pseudo_env or {})
        self.pseudo_env_enabled = bool(self.pseudo_env_cfg.get("enabled", False))
        self.pseudo_env_k = int(self.pseudo_env_cfg.get("k", 3))
        self.env_route_cfg = dict(config.env_routed_inv_heads or {})
        self.env_route_enabled = bool(self.env_route_cfg.get("enabled", False))
        self.env_route_k = int(self.env_route_cfg.get("k", 3))
        self.env_route_mode = str(self.env_route_cfg.get("mode", "confidence_mix") or "confidence_mix").lower()
        self.env_route_replace_final = bool(self.env_route_cfg.get("replace_final", False))
        self.env_route_alpha_detach = bool(self.env_route_cfg.get("alpha_detach", False))
        if self.z_inv_ib_enabled:
            if self.z_inv_ib_apply_to != "z_inv":
                raise ValueError("LOSS.z_inv_bottleneck.apply_to currently supports 'z_inv' only.")
            if self.z_inv_ib_type not in {"vib", "gaussian_noise", "l2_norm"}:
                raise ValueError("LOSS.z_inv_bottleneck.type must be one of: vib, gaussian_noise, l2_norm.")
        if self.pseudo_env_enabled and self.pseudo_env_k < 1:
            raise ValueError("LOSS.pseudo_env_k must be >= 1 when pseudo-env heads are enabled.")
        if self.env_route_enabled:
            if self.env_route_k < 1:
                raise ValueError("LOSS.env_route_k must be >= 1 when env-routed invariant heads are enabled.")
            if self.env_route_mode not in {"soft", "hard", "confidence_mix", "uniform", "random"}:
                raise ValueError("LOSS.env_route_mode must be one of: soft, hard, confidence_mix, uniform, random.")
        self.time_encoder = TimestampEncoder(
            encoding_type=config.time_encoding_type if self.use_timestamp else "none",
            time_emb_dim=self.time_emb_dim,
            tod_emb_dim=config.tod_emb_dim,
            dow_emb_dim=config.dow_emb_dim,
            num_time_in_day=config.num_time_in_day,
            num_day_in_week=config.num_day_in_week,
            timestamp_feature_dim=config.timestamp_feature_dim,
            use_time_of_day=config.use_time_of_day,
            use_day_of_week=config.use_day_of_week,
            required_timestamp=config.required_timestamp,
            dropout=config.dropout,
        )
        self.z_time_adapter = (
            nn.Linear(self.time_emb_dim, self.representation_dim, bias=False)
            if self.use_timestamp and self.time_emb_dim > 0 and self.use_current_timestamp_for_z
            else None
        )
        self.inv_head_from_z = nn.Linear(self.representation_dim, config.output_len * config.output_dim)
        if hasattr(self.backbone, "inv_head") and isinstance(self.backbone.inv_head, nn.Linear):
            if (
                self.backbone.inv_head.in_features == self.inv_head_from_z.in_features
                and self.backbone.inv_head.out_features == self.inv_head_from_z.out_features
            ):
                self.inv_head_from_z.load_state_dict(self.backbone.inv_head.state_dict())
        if self.z_inv_ib_enabled and self.z_inv_ib_type == "vib":
            self.z_inv_ib_mu = nn.Linear(self.representation_dim, self.representation_dim)
            self.z_inv_ib_logvar = nn.Linear(self.representation_dim, self.representation_dim)
            self._init_z_inv_ib_vib()
        else:
            self.z_inv_ib_mu = None
            self.z_inv_ib_logvar = None

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
        self.env_token_encoder = TimeNodeEnvironmentEncoder(
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
            time_emb_dim=self.time_emb_dim if self.use_current_timestamp_for_env else 0,
            use_node_embedding=config.use_node_embedding,
            num_nodes=config.num_nodes,
            node_emb_dim=config.node_emb_dim,
        )
        self.separation = SeparationModule(
            cfg=config.separation,
            num_nodes=config.num_nodes,
            z_dim=self.representation_dim,
            env_dim=config.env_dim,
            input_len=config.input_len,
            input_dim=config.input_dim,
        )
        self.future_env_encoder = FutureEnvEncoder(
            output_len=config.output_len,
            output_dim=config.output_dim,
            env_dim=config.env_dim,
            hidden_dim=int(self.persistence_cfg.get("future_env_hidden_dim", 64)),
            dropout=float(self.persistence_cfg.get("dropout", 0.1)),
        )
        projection_dim = int(self.persistence_cfg.get("projection_dim", 32))
        projection_hidden_dim = int(self.persistence_cfg.get("projection_hidden_dim", 64))
        projection_dropout = float(self.persistence_cfg.get("dropout", 0.1))
        self.persist_q = self._make_persistence_head(config.env_dim, projection_hidden_dim, projection_dim, projection_dropout)
        self.persist_k = self._make_persistence_head(config.env_dim, projection_hidden_dim, projection_dim, projection_dropout)

        self.env_mask = FuturePredictiveEnvMask(
            env_dim=config.env_dim,
            hidden_dim=config.mask_hidden_dim,
            dropout=config.mask_dropout,
            init_bias=config.mask_init_bias,
            temperature=config.mask_temperature,
            force_mask_value=config.force_mask_value,
            pooling=config.mask_pooling,
            eps=config.mask_eps,
            time_emb_dim=self.time_emb_dim if self.use_current_timestamp_for_env else 0,
            use_time=config.mask_use_time,
        )
        self.latent_fusion = LatentFusion(
            z_dim=self.representation_dim,
            env_dim=config.env_dim,
            fusion_type=config.fusion_type,
            hidden_dim=config.fusion_hidden_dim,
            dropout=config.fusion_dropout,
            zero_init=config.fusion_zero_init,
            env_fusion_scale=config.env_fusion_scale,
        )
        self.fpem_predictor = nn.Linear(self.representation_dim, config.output_len * config.output_dim)
        if self.inv_head_from_z.in_features == self.fpem_predictor.in_features and (
            self.inv_head_from_z.out_features == self.fpem_predictor.out_features
        ):
            self.fpem_predictor.load_state_dict(self.inv_head_from_z.state_dict())
        self.env_transition_head = nn.Sequential(
            nn.Linear(config.env_dim, config.env_transition_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.env_transition_dropout),
            nn.Linear(config.env_transition_hidden_dim, config.env_dim),
        )
        self.future_env_decoder = FutureEnvDistributionDecoder(
            env_dim=config.env_dim,
            time_emb_dim=self.time_emb_dim,
            hidden_dim=config.future_decoder_hidden_dim,
            dropout=config.future_decoder_dropout,
            use_time=config.future_decoder_use_time,
            logvar_min=config.future_decoder_logvar_min,
            logvar_max=config.future_decoder_logvar_max,
        )
        self.st_perturbation = STPerturbation(
            enabled=config.perturb_enabled,
            prob=config.perturb_prob,
            value_jitter=config.perturb_value_jitter,
            jitter_std=config.perturb_jitter_std,
            value_scale=config.perturb_value_scale,
            scale_min=config.perturb_scale_min,
            scale_max=config.perturb_scale_max,
            time_node_mask=config.perturb_time_node_mask,
            time_node_mask_ratio=config.perturb_time_node_mask_ratio,
            mask_value=config.perturb_mask_value,
            temporal_block=config.perturb_temporal_block,
            temporal_block_ratio=config.perturb_temporal_block_ratio,
            temporal_block_len=config.perturb_temporal_block_len,
            edge_dropout=config.perturb_edge_dropout,
            edge_dropout_p=config.perturb_edge_dropout_p,
            edge_dropout_for_env_only=config.perturb_edge_dropout_for_env_only,
        )

        decode_in_dim = self.representation_dim + config.env_dim
        self.env_head = nn.Sequential(
            nn.Linear(decode_in_dim, config.residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.residual_dropout),
            nn.Linear(config.residual_hidden_dim, config.output_len * config.output_dim),
        )
        self.pseudo_env_heads = (
            PseudoEnvHeads(
                env_dim=config.env_dim,
                output_len=config.output_len,
                output_dim=config.output_dim,
                num_heads=self.pseudo_env_k,
                hidden_dim=int(self.pseudo_env_cfg.get("hidden_dim", config.residual_hidden_dim)),
                dropout=float(self.pseudo_env_cfg.get("dropout", config.residual_dropout)),
            )
            if self.pseudo_env_enabled
            else None
        )
        self.env_routed_inv_heads = (
            EnvRoutedInvariantHeads(
                inv_dim=self.representation_dim,
                env_dim=config.env_dim,
                output_len=config.output_len,
                output_dim=config.output_dim,
                num_heads=self.env_route_k,
                hidden_dim=int(self.env_route_cfg.get("hidden_dim", config.residual_hidden_dim)),
                dropout=float(self.env_route_cfg.get("dropout", config.residual_dropout)),
                tau=float(self.env_route_cfg.get("tau", 1.0)),
                mode=self.env_route_mode,
                alpha_detach=self.env_route_alpha_detach,
            )
            if self.env_route_enabled
            else None
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

        graph_supports = None
        backbone_cfg = config.backbone or {}
        backbone_name_lower = str(config.backbone_name).lower()
        graphwavenet_full_names = {"graphwavenet_full", "graph_wavenet_full", "gwnet_full", "graphwavenet-full"}
        graphwavenet_adapter_names = {"graphwavenet", "gwnet", "graph_wavenet"}
        if isinstance(backbone_cfg, dict) and backbone_name_lower in graphwavenet_full_names:
            gw_cfg = backbone_cfg.get("graph_wavenet_full", {})
        else:
            gw_cfg = backbone_cfg.get("graph_wavenet", {}) if isinstance(backbone_cfg, dict) else {}
        if config.use_adj and backbone_name_lower in graphwavenet_adapter_names | graphwavenet_full_names:
            graph_supports = load_graph_supports(
                config.adj_path,
                config.num_nodes,
                adjtype=str(gw_cfg.get("adjtype", "doubletransition")),
                add_self_loop=bool(gw_cfg.get("support_add_self_loop", False)),
            )
        if graph_supports is not None:
            self.register_buffer("backbone_adj", graph_supports)
        else:
            self.backbone_adj = None

    @staticmethod
    def _make_persistence_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _init_z_inv_ib_vib(self) -> None:
        if isinstance(self.z_inv_ib_mu, nn.Linear):
            if self.z_inv_ib_mu.weight.shape[0] == self.z_inv_ib_mu.weight.shape[1]:
                nn.init.eye_(self.z_inv_ib_mu.weight)
            else:
                nn.init.xavier_uniform_(self.z_inv_ib_mu.weight)
            nn.init.zeros_(self.z_inv_ib_mu.bias)
        if isinstance(self.z_inv_ib_logvar, nn.Linear):
            nn.init.zeros_(self.z_inv_ib_logvar.weight)
            nn.init.zeros_(self.z_inv_ib_logvar.bias)

    @staticmethod
    def _z_inv_ib_type_id(kind: str) -> float:
        return {"vib": 1.0, "gaussian_noise": 2.0, "l2_norm": 3.0}.get(kind, 0.0)

    def _apply_z_inv_bottleneck(self, z_inv: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.z_inv_ib_enabled:
            return {"z": z_inv}
        if self.z_inv_ib_type == "vib":
            if self.z_inv_ib_mu is None or self.z_inv_ib_logvar is None:
                raise RuntimeError("z_inv VIB is enabled but its mu/logvar heads are missing.")
            z_mu = self.z_inv_ib_mu(z_inv)
            z_logvar = self.z_inv_ib_logvar(z_inv)
            if self.training:
                eps = torch.randn_like(z_mu)
                z_sample = z_mu + eps * torch.exp(0.5 * z_logvar)
            else:
                z_sample = z_mu
            use_sample = self.training and self.z_inv_ib_predict_from_sampled_z
            z_for_prediction = z_sample if use_sample else z_mu
            kl = -0.5 * (1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()).mean()
            return {
                "z": z_for_prediction,
                "z_inv_before_bottleneck": z_inv,
                "z_inv_ib_kl": kl,
                "z_inv_ib_type_id": z_inv.new_tensor(self._z_inv_ib_type_id(self.z_inv_ib_type)),
                "z_inv_ib_z_mu_abs_mean": z_mu.detach().abs().mean(),
                "z_inv_ib_z_logvar_mean": z_logvar.detach().mean(),
                "z_inv_ib_z_sample_std": z_sample.detach().std(unbiased=False),
            }
        if self.z_inv_ib_type == "gaussian_noise":
            if self.training and self.z_inv_ib_noise_std > 0:
                z_for_prediction = z_inv + torch.randn_like(z_inv) * self.z_inv_ib_noise_std
            else:
                z_for_prediction = z_inv
            return {
                "z": z_for_prediction,
                "z_inv_before_bottleneck": z_inv,
                "z_inv_ib_type_id": z_inv.new_tensor(self._z_inv_ib_type_id(self.z_inv_ib_type)),
                "z_inv_ib_z_sample_std": z_for_prediction.detach().std(unbiased=False),
            }
        l2 = z_inv.pow(2).mean()
        return {
            "z": z_inv,
            "z_inv_before_bottleneck": z_inv,
            "z_inv_ib_l2": l2,
            "z_inv_ib_type_id": z_inv.new_tensor(self._z_inv_ib_type_id(self.z_inv_ib_type)),
            "z_inv_ib_z_sample_std": z_inv.detach().std(unbiased=False),
        }

    def invariant_predict_from_z(self, z_inv: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = z_inv.shape
        y_inv = self.inv_head_from_z(z_inv)
        y_inv = y_inv.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        return self._apply_prediction_activation(y_inv)

    def _predict_from_hidden(self, hidden: torch.Tensor, predictor: nn.Linear) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = hidden.shape
        if hidden_dim != self.representation_dim:
            raise AssertionError(f"hidden must end with D_z={self.representation_dim}, got {tuple(hidden.shape)}")
        pred = predictor(hidden)
        pred = pred.view(batch_size, num_nodes, self.output_len, self.output_dim).permute(0, 2, 1, 3)
        return self._apply_prediction_activation(pred)

    @staticmethod
    @contextmanager
    def _frozen_module_parameters(*modules: nn.Module):
        states = []
        for module in modules:
            for param in module.parameters():
                states.append((param, param.requires_grad))
                param.requires_grad_(False)
        try:
            yield
        finally:
            for param, requires_grad in states:
                param.requires_grad_(requires_grad)

    def fpem_predict_from_z_env(self, z_inv: torch.Tensor, env_plus: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_inv.dim() != 3:
            raise AssertionError(f"z_inv must be [B, N, D_z], got {tuple(z_inv.shape)}")
        if env_plus.dim() != 3:
            raise AssertionError(f"env_plus must be [B, N, D_env], got {tuple(env_plus.shape)}")
        if z_inv.shape[:2] != env_plus.shape[:2]:
            raise AssertionError(f"z_inv/env_plus shape mismatch: {tuple(z_inv.shape)} vs {tuple(env_plus.shape)}")
        fusion_out = self.latent_fusion(z_inv, env_plus)
        h_fuse = fusion_out["hidden"]
        prediction = self._predict_from_hidden(h_fuse, self.fpem_predictor)
        return {
            "prediction": prediction,
            "hidden": h_fuse,
            "fusion_gamma": fusion_out["fusion_gamma"],
            "fusion_beta": fusion_out["fusion_beta"],
        }

    def _init_gate_bias(self, bias: float) -> None:
        last = self.gate_net[-1]
        if isinstance(last, nn.Linear) and bias is not None:
            nn.init.constant_(last.bias, bias)

    def _permute_env_with_indices(self, env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, env_dim = env.shape
        flat = env.reshape(batch_size * num_nodes, env_dim)
        perm = torch.randperm(batch_size * num_nodes, device=env.device)
        if self.swap_cfg.get("avoid_self", True) and flat.shape[0] > 1:
            same = perm == torch.arange(flat.shape[0], device=env.device)
            if same.any():
                perm[same] = (perm[same] + 1) % flat.shape[0]
        return flat[perm].reshape(batch_size, num_nodes, env_dim), perm.reshape(batch_size, num_nodes)

    def _permute_env(self, env: torch.Tensor) -> torch.Tensor:
        env_perm, _ = self._permute_env_with_indices(env)
        return env_perm

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
            y_base = self.invariant_predict_from_z(z_decode) if y_inv is None else y_inv.detach()
        else:
            z_decode = z_inv
            y_base = self.invariant_predict_from_z(z_decode) if y_inv is None else y_inv

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

    def _compute_persistence(
        self,
        env_hist: torch.Tensor,
        y_inv: torch.Tensor,
        y_true: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        env_fut = None
        q_hist = None
        k_fut = None
        persist_score = None
        if self.persistence_enabled and self.training and y_true is not None:
            aligned_true = align_target(y_true, y_inv)
            future_residual = aligned_true - y_inv.detach()
            env_fut = self.future_env_encoder(future_residual)
            q_hist = self.persist_q(env_hist)
            k_fut_raw = self.persist_k(env_fut)
            k_fut = (
                k_fut_raw.detach()
                if bool(self.persistence_cfg.get("detach_future_key", True))
                else k_fut_raw
            )
            q_norm = torch.nn.functional.normalize(q_hist, dim=-1)
            k_norm = torch.nn.functional.normalize(k_fut, dim=-1)
            persist_score = (q_norm * k_norm).sum(dim=-1, keepdim=True)
        return {
            "env_fut": env_fut,
            "persist_q": q_hist,
            "persist_k": k_fut,
            "persist_score": persist_score,
        }

    def _encode_time(
        self,
        x: torch.Tensor,
        seq_time: Optional[torch.Tensor],
        cur_time: Optional[torch.Tensor],
        future_time: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self.time_encoder(
            seq_time=seq_time,
            cur_time=cur_time,
            future_time=future_time,
            batch_size=x.shape[0],
            seq_len=x.shape[1],
            future_len=self.output_len,
            device=x.device,
            dtype=x.dtype,
        )

    def _match_env_input_dim(self, y: torch.Tensor) -> torch.Tensor:
        y = ensure_blnc(y, "future_env_input")
        if y.shape[-1] == self.input_dim:
            return y
        if y.shape[-1] > self.input_dim:
            return y[..., : self.input_dim]
        pad = self.input_dim - y.shape[-1]
        return torch.nn.functional.pad(y, (0, pad))

    def _apply_time_to_z(self, z_raw: torch.Tensor, cur_time_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.z_time_adapter is None or cur_time_emb is None:
            return z_raw
        if tuple(cur_time_emb.shape) != (z_raw.shape[0], self.time_emb_dim):
            raise AssertionError(
                f"cur_time_emb must be {(z_raw.shape[0], self.time_emb_dim)}, got {tuple(cur_time_emb.shape)}"
            )
        return z_raw + self.z_time_adapter(cur_time_emb).unsqueeze(1)

    def _forward_fpem(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        seq_time: Optional[torch.Tensor] = None,
        cur_time: Optional[torch.Tensor] = None,
        future_time: Optional[torch.Tensor] = None,
        compute_aux: bool = False,
        adj_override: Optional[torch.Tensor] = None,
        backbone_adj_override: Optional[torch.Tensor] = None,
        compute_swap: bool = True,
        apply_perturb: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size, input_len, num_nodes, _ = x.shape
        adj = adj_override if adj_override is not None else getattr(self, "adj_norm", None)
        backbone_adj = backbone_adj_override if backbone_adj_override is not None else getattr(self, "backbone_adj", None)
        time_out = self._encode_time(x, seq_time, cur_time, future_time)
        seq_time_emb = time_out["seq_time_emb"] if self.use_current_timestamp_for_env else None
        cur_time_emb = time_out["cur_time_emb"] if self.use_current_timestamp_for_env else None
        future_time_emb = time_out["future_time_emb"]

        backbone_out = self.backbone(
            x,
            adj=backbone_adj if backbone_adj is not None else adj,
            seq_time=seq_time,
            cur_time=cur_time,
            future_time=future_time,
        )
        z_encoder = backbone_out["z_inv"]
        z_raw = self._apply_time_to_z(z_encoder, time_out["cur_time_emb"])
        z_seq = backbone_out.get("z_seq")
        y_inv_raw = self._apply_prediction_activation(backbone_out["y_inv"])
        env_mu_tokens, env_logvar_tokens, env_tokens = self.env_token_encoder(
            x,
            seq_time_emb=seq_time_emb,
            cur_time_emb=cur_time_emb,
            adj_norm=adj,
        )
        mask_out = self.env_mask(env_tokens, seq_time_emb=seq_time_emb, cur_time_emb=cur_time_emb)
        env_hist_raw = mask_out["env_hist"]

        sep_out = self.separation(x=x, z_raw=z_raw, env_raw=env_hist_raw, y_inv_raw=y_inv_raw)
        z_inv = sep_out["z_inv"]
        env_hist = sep_out["env"]
        separation_extra = sep_out["extra"]
        z_inv_ib_out = self._apply_z_inv_bottleneck(z_inv)
        z_for_prediction = z_inv_ib_out["z"]

        mask = mask_out["mask"]
        env_plus = mask_out["env_plus"]
        env_minus = mask_out["env_minus"]

        pred_out = self.fpem_predict_from_z_env(z_for_prediction, env_plus)
        prediction = pred_out["prediction"]
        zero_env = torch.zeros_like(env_plus)
        inv_out = self.fpem_predict_from_z_env(z_for_prediction, zero_env)
        y_inv = inv_out["prediction"]
        pseudo_env_head_pred = self.pseudo_env_heads(env_plus) if self.pseudo_env_heads is not None else None
        env_route_out = (
            self.env_routed_inv_heads(inv_out["hidden"], env_plus, y_inv)
            if self.env_routed_inv_heads is not None
            else None
        )
        prediction_base = prediction
        if env_route_out is not None and self.env_route_replace_final:
            prediction = env_route_out["y_route_final"]

        env_fut_tokens = None
        env_fut_mu_tokens = None
        env_fut_logvar_tokens = None
        env_fut = None
        pred_fut_mu = None
        pred_fut_logvar = None
        pred_fut_mu_minus = None
        pred_fut_logvar_minus = None
        persist_q = None
        persist_k = None
        persist_score = None
        compute_future_aux = y_true is not None and (self.training or compute_aux)
        if compute_future_aux:
            aligned_true = align_target(y_true, y_inv)
            future_env_input = self._match_env_input_dim(aligned_true)
            env_fut_mu_tokens, env_fut_logvar_tokens, env_fut_tokens = self.env_token_encoder(
                future_env_input,
                seq_time_emb=future_time_emb if self.use_current_timestamp_for_env else None,
                cur_time_emb=cur_time_emb,
                adj_norm=adj,
            )
            env_fut = env_fut_tokens.mean(dim=1)
            decoder_plus = self.future_env_decoder(
                env_plus,
                future_time_emb=future_time_emb,
                cur_time_emb=time_out["cur_time_emb"],
                future_len=self.output_len,
            )
            decoder_minus = self.future_env_decoder(
                env_minus,
                future_time_emb=future_time_emb,
                cur_time_emb=time_out["cur_time_emb"],
                future_len=self.output_len,
            )
            pred_fut_mu = decoder_plus["pred_fut_mu"]
            pred_fut_logvar = decoder_plus["pred_fut_logvar"]
            pred_fut_mu_minus = decoder_minus["pred_fut_mu"]
            pred_fut_logvar_minus = decoder_minus["pred_fut_logvar"]
            if self.persistence_enabled:
                persist_q = self.persist_q(env_plus)
                k_fut_raw = self.persist_k(env_fut)
                persist_k = (
                    k_fut_raw.detach()
                    if bool(self.persistence_cfg.get("detach_future_key", True))
                    else k_fut_raw
                )
                q_norm = torch.nn.functional.normalize(persist_q, dim=-1)
                k_norm = torch.nn.functional.normalize(persist_k, dim=-1)
                persist_score = (q_norm * k_norm).sum(dim=-1, keepdim=True)

        prediction_swap = None
        env_perm = None
        env_perm_index = None
        if compute_swap and (self.training or compute_aux) and self.swap_cfg.get("enabled", True):
            env_perm, env_perm_index = self._permute_env_with_indices(env_plus)
            z_swap_decode = z_for_prediction.detach() if self.swap_detach_inv else z_for_prediction
            detach_env = bool(self.swap_cfg.get("detach_env", self.swap_detach_env))
            env_swap_decode = env_perm.detach() if detach_env else env_perm
            if bool(self.swap_cfg.get("freeze_predictor", True)):
                with self._frozen_module_parameters(self.latent_fusion, self.fpem_predictor):
                    swap_out = self.fpem_predict_from_z_env(z_swap_decode, env_swap_decode)
            else:
                swap_out = self.fpem_predict_from_z_env(z_swap_decode, env_swap_decode)
            prediction_swap = swap_out["prediction"]

        expected_pred_shape = (batch_size, self.output_len, num_nodes, self.output_dim)
        expected_gate_shape = (batch_size, self.output_len, num_nodes, 1)
        expected_z_shape = (batch_size, num_nodes, self.representation_dim)
        expected_env_shape = (batch_size, num_nodes, self.env_dim)
        expected_token_shape = (batch_size, input_len, num_nodes, self.env_dim)
        expected_mask_shape = (batch_size, input_len, num_nodes, 1)
        shape_checks = {
            "prediction": (prediction, expected_pred_shape),
            "y_inv": (y_inv, expected_pred_shape),
            "z_inv": (z_inv, expected_z_shape),
            "z_raw": (z_raw, expected_z_shape),
            "env_mu_tokens": (env_mu_tokens, expected_token_shape),
            "env_logvar_tokens": (env_logvar_tokens, expected_token_shape),
            "env_tokens": (env_tokens, expected_token_shape),
            "env_hist": (env_hist, expected_env_shape),
            "env_plus": (env_plus, expected_env_shape),
            "env_minus": (env_minus, expected_env_shape),
            "mask": (mask, expected_mask_shape),
            "y_inv_raw": (y_inv_raw, expected_pred_shape),
        }
        for name, (tensor, expected_shape) in shape_checks.items():
            if tuple(tensor.shape) != expected_shape:
                raise AssertionError(f"{name} must be {expected_shape}, got {tuple(tensor.shape)}")
        expected_fut_token_shape = (batch_size, self.output_len, num_nodes, self.env_dim)
        for name, tensor in {
            "env_fut_tokens": env_fut_tokens,
            "env_fut_mu_tokens": env_fut_mu_tokens,
            "env_fut_logvar_tokens": env_fut_logvar_tokens,
            "pred_fut_mu": pred_fut_mu,
            "pred_fut_logvar": pred_fut_logvar,
        }.items():
            if tensor is not None and tuple(tensor.shape) != expected_fut_token_shape:
                raise AssertionError(f"{name} must be {expected_fut_token_shape}, got {tuple(tensor.shape)}")
        if pseudo_env_head_pred is not None:
            expected_pseudo_shape = (batch_size, self.pseudo_env_k, self.output_len, num_nodes, self.output_dim)
            if tuple(pseudo_env_head_pred.shape) != expected_pseudo_shape:
                raise AssertionError(
                    f"pseudo_env_head_pred must be {expected_pseudo_shape}, got {tuple(pseudo_env_head_pred.shape)}"
                )
        if env_route_out is not None:
            expected_route_heads_shape = (batch_size, self.env_route_k, self.output_len, num_nodes, self.output_dim)
            expected_route_q_shape = (batch_size, self.env_route_k)
            for name in ["y_route_heads"]:
                if tuple(env_route_out[name].shape) != expected_route_heads_shape:
                    raise AssertionError(
                        f"{name} must be {expected_route_heads_shape}, got {tuple(env_route_out[name].shape)}"
                    )
            for name in ["env_route_logits", "env_route_q"]:
                if tuple(env_route_out[name].shape) != expected_route_q_shape:
                    raise AssertionError(f"{name} must be {expected_route_q_shape}, got {tuple(env_route_out[name].shape)}")
            for name in ["y_route_soft", "y_route_hard", "y_route", "y_route_selected", "y_global", "y_route_final"]:
                if tuple(env_route_out[name].shape) != expected_pred_shape:
                    raise AssertionError(f"{name} must be {expected_pred_shape}, got {tuple(env_route_out[name].shape)}")
            for name in ["route_confidence", "env_route_entropy", "env_route_alpha"]:
                if tuple(env_route_out[name].shape) != (batch_size,):
                    raise AssertionError(f"{name} must be {(batch_size,)}, got {tuple(env_route_out[name].shape)}")
            if (env_route_out["env_route_alpha"].detach() < 0).any() or (env_route_out["env_route_alpha"].detach() > 1).any():
                raise AssertionError(
                    "env_route_alpha must stay in [0, 1]"
                )

        rho = mask.mean(dim=1).unsqueeze(1).expand(-1, self.output_len, -1, -1)
        if tuple(rho.shape) != expected_gate_shape:
            raise AssertionError(f"rho placeholder must be {expected_gate_shape}, got {tuple(rho.shape)}")
        r_env = torch.zeros_like(prediction)
        y_potential = prediction

        output = {
            "method_variant": "fpem",
            "baseline_name": self.baseline_name,
            "reference_status": self.reference_status,
            "prediction": prediction,
            "y_inv": y_inv,
            "y_potential": y_potential,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_for_prediction,
            "z_raw": z_raw,
            "z_seq": z_seq,
            # Hook-only invariant encoder tensors for TC-SGC. These are not
            # consumed by losses/predictors; a tensor hook here only changes
            # grad_z returned to the invariant encoder.
            "grad_consensus_z_inv": z_encoder,
            "grad_consensus_z_seq": z_seq,
            "env_mu": env_mu_tokens,
            "env_logvar": env_logvar_tokens,
            "env": env_hist,
            "env_hist": env_hist,
            "env_hist_bar": env_hist,
            "env_hist_tokens": env_tokens,
            "env_hist_mu_tokens": env_mu_tokens,
            "env_hist_logvar_tokens": env_logvar_tokens,
            "env_raw": env_hist_raw,
            "env_tokens": env_tokens,
            "env_plus": env_plus,
            "env_minus": env_minus,
            "env_plus_tokens": mask_out["env_plus_tokens"],
            "env_minus_tokens": mask_out["env_minus_tokens"],
            "mask": mask,
            "y_inv_raw": y_inv_raw,
            "separation_mode": sep_out["mode"],
            "separation_extra": separation_extra,
            "env_fut": env_fut,
            "env_fut_tokens": env_fut_tokens,
            "env_fut_mu_tokens": env_fut_mu_tokens,
            "env_fut_logvar_tokens": env_fut_logvar_tokens,
            "pred_fut_mu": pred_fut_mu,
            "pred_fut_logvar": pred_fut_logvar,
            "pred_fut_mu_minus": pred_fut_mu_minus,
            "pred_fut_logvar_minus": pred_fut_logvar_minus,
            "env_fut_pred": pred_fut_mu.mean(dim=1) if pred_fut_mu is not None else None,
            "env_fut_pred_minus": pred_fut_mu_minus.mean(dim=1) if pred_fut_mu_minus is not None else None,
            "persist_q": persist_q,
            "persist_k": persist_k,
            "persist_score": persist_score,
            "persistence_enabled": self.persistence_enabled,
            "prediction_swap": prediction_swap,
            "rho_swap": None,
            "env_perm": env_perm,
            "env_perm_index": env_perm_index,
            "fusion_gamma": pred_out["fusion_gamma"],
            "fusion_beta": pred_out["fusion_beta"],
            "fusion_gamma_inv": inv_out["fusion_gamma"],
            "fusion_beta_inv": inv_out["fusion_beta"],
            "seq_time_emb": time_out["seq_time_emb"],
            "cur_time_emb": time_out["cur_time_emb"],
            "future_time_emb": time_out["future_time_emb"],
            "timestamp_valid": time_out["timestamp_valid"],
            "time_encoding_type_id": time_out["time_encoding_type_id"],
            "backbone_aux_losses": backbone_out.get("backbone_aux_losses", {}),
            "backbone_aux_weights": backbone_out.get("backbone_aux_weights", {}),
        }
        if self.z_inv_ib_enabled:
            output.update({key: value for key, value in z_inv_ib_out.items() if key != "z"})
        if pseudo_env_head_pred is not None:
            output["pseudo_env_head_pred"] = pseudo_env_head_pred
        if env_route_out is not None:
            output["prediction_fused"] = prediction_base
            output["env_route_replace_final"] = self.env_route_replace_final
            output.update(env_route_out)
        if apply_perturb:
            output["perturb_info"] = {"enabled": self.perturb_enabled, "applied": False}
        if apply_perturb and self.training and self.perturb_enabled:
            x_aug, adj_aug, perturb_info = self.st_perturbation(x, adj=adj)
            output["perturb_info"] = perturb_info
            if bool(perturb_info.get("applied", False)):
                aug_backbone_adj = backbone_adj
                if not bool(perturb_info.get("edge_dropout_for_env_only", True)) and backbone_adj is None:
                    aug_backbone_adj = adj_aug
                aug_out = self._forward_fpem(
                    x_aug,
                    y_true=None,
                    seq_time=seq_time,
                    cur_time=cur_time,
                    future_time=future_time,
                    compute_aux=False,
                    adj_override=adj_aug,
                    backbone_adj_override=aug_backbone_adj,
                    compute_swap=False,
                    apply_perturb=False,
                )
                output["z_inv_aug"] = aug_out["z_inv"]
                output["y_inv_aug"] = aug_out["y_inv"]
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        seq_time: Optional[torch.Tensor] = None,
        cur_time: Optional[torch.Tensor] = None,
        future_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        x = ensure_blnc(inputs, "inputs")
        if self.is_fpem:
            return self._forward_fpem(
                x,
                y_true=y_true,
                seq_time=seq_time,
                cur_time=cur_time,
                future_time=future_time,
                compute_aux=bool(kwargs.get("compute_aux", False)),
            )
        batch_size, _, num_nodes, _ = x.shape
        adj = getattr(self, "adj_norm", None)
        backbone_adj = getattr(self, "backbone_adj", None)

        backbone_out = self.backbone(
            x,
            adj=backbone_adj if backbone_adj is not None else adj,
            seq_time=seq_time,
            cur_time=cur_time,
            future_time=future_time,
        )
        z_encoder = backbone_out["z_inv"]
        z_raw = z_encoder
        z_seq = backbone_out.get("z_seq")
        y_inv_raw = self._apply_prediction_activation(backbone_out["y_inv"])
        env_mu, env_logvar, env_raw = self.env_encoder(x, adj)
        sep_out = self.separation(x=x, z_raw=z_raw, env_raw=env_raw, y_inv_raw=y_inv_raw)
        z_inv = sep_out["z_inv"]
        env = sep_out["env"]
        separation_extra = sep_out["extra"]
        if self.use_separated_z_for_y_inv:
            y_inv = self.invariant_predict_from_z(z_inv)
        else:
            y_inv = y_inv_raw
        persistence_out = self._compute_persistence(env, y_inv, y_true)

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
            "z_raw": (z_raw, expected_z_shape),
            "env_mu": (env_mu, expected_env_shape),
            "env_logvar": (env_logvar, expected_env_shape),
            "env": (env, expected_env_shape),
            "env_hist": (env, expected_env_shape),
            "env_raw": (env_raw, expected_env_shape),
            "y_inv_raw": (y_inv_raw, expected_pred_shape),
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
            "method_variant": "nue",
            "baseline_name": self.baseline_name,
            "reference_status": self.reference_status,
            "prediction": prediction,
            "y_inv": y_inv,
            "y_potential": y_potential,
            "r_env": r_env,
            "rho": rho,
            "z_inv": z_inv,
            "z_raw": z_raw,
            "z_seq": z_seq,
            "grad_consensus_z_inv": z_encoder,
            "grad_consensus_z_seq": z_seq,
            "env_mu": env_mu,
            "env_logvar": env_logvar,
            "env": env,
            "env_hist": env,
            "env_raw": env_raw,
            "y_inv_raw": y_inv_raw,
            "separation_mode": sep_out["mode"],
            "separation_extra": separation_extra,
            "env_fut": persistence_out["env_fut"],
            "persist_q": persistence_out["persist_q"],
            "persist_k": persistence_out["persist_k"],
            "persist_score": persistence_out["persist_score"],
            "persistence_enabled": self.persistence_enabled,
            "prediction_swap": prediction_swap,
            "rho_swap": rho_swap,
            "env_perm": env_perm,
            "env_perm_index": None,
            "backbone_aux_losses": backbone_out.get("backbone_aux_losses", {}),
            "backbone_aux_weights": backbone_out.get("backbone_aux_weights", {}),
        }
