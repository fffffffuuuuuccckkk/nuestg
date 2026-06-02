from __future__ import annotations

from typing import Dict, Optional

import torch

from models.backbones.base import BaseBackbone
from models.backbones.official_utils import OfficialBaselineSkip, ensure_repo_exists_or_skip


class STONEOfficialBackbone(BaseBackbone):
    """Full-official STONE wrapper gate for official data/protocol only."""

    CANDIDATE_NAMES = ("STONE-KDD-2024", "STONE", "stone")

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
            "Full official STONE requires spatial side information and structural-shift "
            "metadata; current PEMS08 fixed-node config only supports STONE-adapter."
        ),
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        self.repo_path = ensure_repo_exists_or_skip("STONE", self.CANDIDATE_NAMES, explicit_path=external_path or None)
        required = [
            "Knowair/model/STONE.py",
            "Knowair/frechet.py",
            "Knowair/spatial_side_information.py",
            "Knowair/train.py",
            "src/base/stone.py",
            "src/base/stone_engine.py",
        ]
        missing = [item for item in required if not (self.repo_path / item).exists()]
        if missing:
            raise OfficialBaselineSkip("STONE", f"official files missing: {', '.join(missing)}")
        if official_requires_special_data:
            raise OfficialBaselineSkip("STONE", unsupported_reason)
        raise OfficialBaselineSkip(
            "STONE",
            "official repo is available, but no official STONE data protocol config was selected",
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        raise RuntimeError("STONEOfficialBackbone should skip during construction for unsupported PEMS08.")

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("STONEOfficialBackbone does not expose an adapter forecast path.")
