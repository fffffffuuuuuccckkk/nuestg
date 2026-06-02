from __future__ import annotations

from typing import Dict, Optional

import torch

from models.backbones.base import BaseBackbone
from models.backbones.official_utils import OfficialBaselineSkip, ensure_repo_exists_or_skip


class CaSTOfficialBackbone(BaseBackbone):
    """Full-official CaST wrapper gate.

    The local CaST official code requires PyTorch Geometric `Data` objects,
    Hodge-Laplacian edge-level convolutions, graph preprocessing, and CaST's
    VQ/MI training losses. The current PEMS08 BasicTS fixed-node batch cannot
    represent that full protocol without a dedicated official data converter.
    """

    CANDIDATE_NAMES = ("CaST", "cast", "yutong-xia-CaST")

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
            "current PEMS08 BasicTS fixed-node config cannot provide the official "
            "PyG graph Data object, Hodge-Laplacian edge graph, CaST preprocessing, "
            "and VQ/MI loss protocol required by full CaST"
        ),
        **kwargs,
    ) -> None:
        super().__init__(input_len, output_len, num_nodes, input_dim, output_dim, representation_dim)
        del kwargs
        self.repo_path = ensure_repo_exists_or_skip("CaST", self.CANDIDATE_NAMES, explicit_path=external_path or None)
        required = [
            "src/models/cast.py",
            "src/layers/cast_cell.py",
            "src/utils/dataset.py",
            "src/trainers/cast_trainer.py",
        ]
        missing = [item for item in required if not (self.repo_path / item).exists()]
        if missing:
            raise OfficialBaselineSkip("CaST", f"official files missing: {', '.join(missing)}")
        if official_requires_special_data:
            raise OfficialBaselineSkip("CaST", unsupported_reason)
        raise OfficialBaselineSkip(
            "CaST",
            "official import path is available, but full CaST data/loss adapter has not been enabled",
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        raise RuntimeError("CaSTOfficialBackbone should skip during construction for unsupported PEMS08.")

    def forecast_from_representation(self, z_inv: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("CaSTOfficialBackbone does not expose an adapter forecast path.")
