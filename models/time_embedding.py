from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

import torch
from torch import nn


class TimestampEncoder(nn.Module):
    """Encode historical/current/future timestamps for FPEM.

    The encoder accepts BasicTS timestamp tensors when available and falls back
    to zero embeddings when timestamps are absent and not required. For PEMS
    style BasicTS data, timestamps are typically [time_of_day, day_of_week].
    """

    TYPE_IDS = {"none": 0, "stid": 1, "sinusoidal": 2, "mlp": 3}

    def __init__(
        self,
        encoding_type: str,
        time_emb_dim: int,
        tod_emb_dim: int = 16,
        dow_emb_dim: int = 8,
        num_time_in_day: int = 288,
        num_day_in_week: int = 7,
        timestamp_feature_dim: int = 0,
        use_time_of_day: bool = True,
        use_day_of_week: bool = True,
        required_timestamp: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoding_type = str(encoding_type or "none").lower()
        self.time_emb_dim = int(time_emb_dim)
        self.tod_emb_dim = int(tod_emb_dim)
        self.dow_emb_dim = int(dow_emb_dim)
        self.num_time_in_day = int(num_time_in_day)
        self.num_day_in_week = int(num_day_in_week)
        self.timestamp_feature_dim = int(timestamp_feature_dim)
        self.use_time_of_day = bool(use_time_of_day)
        self.use_day_of_week = bool(use_day_of_week)
        self.required_timestamp = bool(required_timestamp)
        self.dropout = nn.Dropout(dropout)
        self._warned_missing = False

        if self.encoding_type not in self.TYPE_IDS:
            raise ValueError(
                f"Unsupported MODEL.time_encoding_type={encoding_type!r}; "
                "expected stid, sinusoidal, mlp, or none."
            )
        if self.time_emb_dim <= 0 or self.encoding_type == "none":
            self.enabled = False
            self.tod_embedding = None
            self.dow_embedding = None
            self.raw_mlp = None
            self.out_proj = None
            return

        self.enabled = True
        if self.encoding_type == "stid":
            parts_dim = 0
            self.tod_embedding = (
                nn.Embedding(self.num_time_in_day, self.tod_emb_dim)
                if self.use_time_of_day and self.tod_emb_dim > 0
                else None
            )
            self.dow_embedding = (
                nn.Embedding(self.num_day_in_week, self.dow_emb_dim)
                if self.use_day_of_week and self.dow_emb_dim > 0
                else None
            )
            parts_dim += self.tod_emb_dim if self.tod_embedding is not None else 0
            parts_dim += self.dow_emb_dim if self.dow_embedding is not None else 0
            raw_dim = max(self.timestamp_feature_dim, 0)
            self.raw_mlp = (
                nn.Sequential(nn.Linear(raw_dim, self.time_emb_dim), nn.GELU())
                if raw_dim > 0
                else None
            )
            parts_dim += self.time_emb_dim if self.raw_mlp is not None else 0
            if parts_dim <= 0:
                parts_dim = self.time_emb_dim
            self.out_proj = nn.Linear(parts_dim, self.time_emb_dim)
        elif self.encoding_type == "mlp":
            raw_dim = max(self.timestamp_feature_dim, 1)
            self.tod_embedding = None
            self.dow_embedding = None
            self.raw_mlp = nn.Sequential(
                nn.Linear(raw_dim, self.time_emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
            )
            self.out_proj = None
        else:
            self.tod_embedding = None
            self.dow_embedding = None
            self.raw_mlp = None
            self.out_proj = None

    def _zero(
        self,
        batch_size: int,
        length: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        shape = (batch_size, self.time_emb_dim) if length is None else (batch_size, length, self.time_emb_dim)
        return torch.zeros(shape, device=device, dtype=dtype)

    def _canonical(self, timestamps: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        if timestamps is None:
            return None
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        elif timestamps.dim() not in {2, 3}:
            raise AssertionError(f"{name} must be [B], [B,S], [B,D], or [B,S,D], got {tuple(timestamps.shape)}")
        return timestamps.float()

    def _tod_index(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.long()
        vmax = values.detach().max()
        vmin = values.detach().min()
        if vmax <= 1.0 + 1e-4 and vmin >= -1e-4:
            idx = torch.round(values.clamp(0.0, 1.0) * (self.num_time_in_day - 1))
        else:
            idx = torch.round(values).remainder(self.num_time_in_day)
        return idx.long().clamp(0, self.num_time_in_day - 1)

    def _dow_index(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.long()
        vmax = values.detach().max()
        vmin = values.detach().min()
        if vmax <= 1.0 + 1e-4 and vmin >= -1e-4:
            idx = torch.round(values.clamp(0.0, 1.0) * (self.num_day_in_week - 1))
        else:
            idx = torch.round(values).remainder(self.num_day_in_week)
        return idx.long().clamp(0, self.num_day_in_week - 1)

    def _sinusoidal(self, timestamps: Optional[torch.Tensor], batch_size: int, length: Optional[int], device, dtype):
        if not self.enabled:
            return None
        if timestamps is not None:
            base = timestamps[..., 0]
            if base.dim() == 1 and length is not None:
                base = base.unsqueeze(1).expand(-1, length)
            elif base.dim() == 2 and length is None:
                base = base[:, -1]
        else:
            if length is None:
                base = torch.zeros(batch_size, device=device, dtype=dtype)
            else:
                base = torch.arange(length, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        half = max(self.time_emb_dim // 2, 1)
        freq = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = base.to(device=device, dtype=dtype).unsqueeze(-1) * freq
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.time_emb_dim:
            emb = torch.nn.functional.pad(emb, (0, self.time_emb_dim - emb.shape[-1]))
        return emb[..., : self.time_emb_dim]

    def _encode_one(
        self,
        timestamps: Optional[torch.Tensor],
        batch_size: int,
        length: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        timestamps = self._canonical(timestamps, name)
        if timestamps is None:
            if self.required_timestamp:
                raise ValueError(f"MODEL.required_timestamp=True but {name} is None")
            if not self._warned_missing:
                warnings.warn(
                    "TimestampEncoder received no timestamps; using zero time embeddings.",
                    RuntimeWarning,
                )
                self._warned_missing = True
            return self._zero(batch_size, length, device, dtype)

        timestamps = timestamps.to(device=device, dtype=dtype)
        if length is not None and timestamps.dim() == 2:
            if timestamps.shape[1] == length:
                timestamps = timestamps.unsqueeze(-1)
            else:
                timestamps = timestamps.unsqueeze(1).expand(-1, length, -1)
        if length is None and timestamps.dim() == 3:
            timestamps = timestamps[:, -1]
        if length is not None and timestamps.shape[1] != length:
            raise AssertionError(f"{name} expected length={length}, got {tuple(timestamps.shape)}")

        if self.encoding_type == "sinusoidal":
            return self._sinusoidal(timestamps, batch_size, length, device, dtype)
        if self.encoding_type == "mlp":
            raw = timestamps
            raw_dim = self.raw_mlp[0].in_features
            if raw.shape[-1] < raw_dim:
                raw = torch.nn.functional.pad(raw, (0, raw_dim - raw.shape[-1]))
            raw = raw[..., :raw_dim]
            return self.dropout(self.raw_mlp(raw))

        parts = []
        if self.tod_embedding is not None:
            parts.append(self.tod_embedding(self._tod_index(timestamps[..., 0])))
        if self.dow_embedding is not None:
            dow_values = timestamps[..., 1] if timestamps.shape[-1] > 1 else timestamps[..., 0]
            parts.append(self.dow_embedding(self._dow_index(dow_values)))
        if self.raw_mlp is not None:
            raw_dim = self.raw_mlp[0].in_features
            raw = timestamps
            if raw.shape[-1] < raw_dim:
                raw = torch.nn.functional.pad(raw, (0, raw_dim - raw.shape[-1]))
            parts.append(self.raw_mlp(raw[..., :raw_dim]))
        if not parts:
            return self._zero(batch_size, length, device, dtype)
        emb = torch.cat(parts, dim=-1)
        emb = self.out_proj(emb) if self.out_proj is not None else emb
        return self.dropout(emb)

    def forward(
        self,
        seq_time: Optional[torch.Tensor] = None,
        cur_time: Optional[torch.Tensor] = None,
        future_time: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        future_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ref = seq_time if seq_time is not None else future_time if future_time is not None else cur_time
        if ref is not None:
            batch_size = int(ref.shape[0])
            device = ref.device
            dtype = ref.dtype if ref.is_floating_point() else torch.float32
        if batch_size is None or device is None:
            raise ValueError("TimestampEncoder needs batch_size and device when all timestamps are None.")
        dtype = dtype or torch.float32
        if cur_time is None and seq_time is not None:
            seq_canon = self._canonical(seq_time, "seq_time")
            if seq_canon is not None and seq_canon.dim() == 3:
                cur_time = seq_canon[:, -1]

        seq_emb = self._encode_one(seq_time, batch_size, seq_len, device, dtype, "seq_time")
        cur_emb = self._encode_one(cur_time, batch_size, None, device, dtype, "cur_time")
        future_emb = self._encode_one(future_time, batch_size, future_len, device, dtype, "future_time")
        valid = ref is not None and self.enabled
        return {
            "seq_time_emb": seq_emb,
            "cur_time_emb": cur_emb,
            "future_time_emb": future_emb,
            "timestamp_valid": valid,
            "time_encoding_type_id": self.TYPE_IDS.get(self.encoding_type, 0),
        }
