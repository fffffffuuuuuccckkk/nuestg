from __future__ import annotations

from typing import Dict, Optional

import torch

from models.backbones.base import BaseBackbone
from models.backbones.official_utils import OfficialBaselineSkip, ensure_repo_exists_or_skip


class STOPOfficialBackbone(BaseBackbone):
    """Full-official STOP wrapper gate for SOOD/OOD protocol configs only."""

    CANDIDATE_NAMES = ("STOP", "stop", "STOP-ICML-2025")

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        representation_dim: int,
        external_path: str = "",
        official_requires_special_data: bool = True,
        unsupported_reason: str = (
            "Full official STOP requires official SOOD/LargeST/KnowAir/TrafficStream "
            "protocol. Current PEMS08 config only supports STOP-adapter."
        ),
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        self.repo_path = ensure_repo_exists_or_skip("STOP", self.CANDIDATE_NAMES, explicit_path=external_path or None)
        required = [
            "LargeST/src/models/stop.py",
            "LargeST/src/engines/stop_engine.py",
            "LargeST/experiments/stop/main.py",
            "KnowAir/src/models/stop.py",
            "TrafficStream/src/models/stop.py",
        ]
        missing = [item for item in required if not (self.repo_path / item).exists()]
        if missing:
            raise OfficialBaselineSkip("STOP", f"official files missing: {', '.join(missing)}")
        if official_requires_special_data:
            raise OfficialBaselineSkip("STOP", unsupported_reason)
        raise OfficialBaselineSkip(
            "STOP",
            "official repo is available, but no official SOOD/LargeST/KnowAir/TrafficStream config was selected",
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        raise RuntimeError("STOPOfficialBackbone should skip during construction for unsupported PEMS08.")

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("STOPOfficialBackbone does not expose an adapter forecast path.")
