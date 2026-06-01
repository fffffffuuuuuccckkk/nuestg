from __future__ import annotations

from typing import Optional, Tuple

import torch


def _time_values(
    seq_time: Optional[torch.Tensor],
    feature_idx: int,
    batch_size: int,
    input_len: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if seq_time is None:
        return torch.zeros(batch_size, input_len, num_nodes, device=device, dtype=dtype)
    seq_time = seq_time.to(device=device, dtype=dtype)
    if seq_time.dim() == 3:
        if seq_time.shape[0] != batch_size or seq_time.shape[1] != input_len:
            raise ValueError(f"seq_time must be [B,L,T], got {tuple(seq_time.shape)}")
        return seq_time[:, :, feature_idx].unsqueeze(-1).expand(-1, -1, num_nodes)
    if seq_time.dim() == 4:
        if seq_time.shape[:3] != (batch_size, input_len, num_nodes):
            raise ValueError(f"seq_time must be [B,L,N,T], got {tuple(seq_time.shape)}")
        return seq_time[:, :, :, feature_idx]
    raise ValueError(f"seq_time must be [B,L,T] or [B,L,N,T], got {tuple(seq_time.shape)}")


def normalized_tod_and_dow(
    seq_time: Optional[torch.Tensor],
    batch_size: int,
    input_len: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    num_time_in_day: int = 288,
    num_day_in_week: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tod = _time_values(seq_time, 0, batch_size, input_len, num_nodes, device, dtype)
    dow = _time_values(seq_time, 1, batch_size, input_len, num_nodes, device, dtype)
    tod = torch.nan_to_num(tod, nan=0.0)
    dow = torch.nan_to_num(dow, nan=0.0)
    if tod.numel() > 0 and float(tod.detach().max().cpu()) > 1.0 + 1e-6:
        tod = tod / float(num_time_in_day)
    if dow.numel() > 0 and float(dow.detach().max().cpu()) > 1.0 + 1e-6:
        dow = dow / float(num_day_in_week)
    return tod.clamp(0.0, 1.0), dow.clamp(0.0, 1.0)


def d2stgnn_time_features(
    seq_time: Optional[torch.Tensor],
    batch_size: int,
    input_len: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    num_time_in_day: int = 288,
    num_day_in_week: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tod, dow_norm = normalized_tod_and_dow(
        seq_time,
        batch_size,
        input_len,
        num_nodes,
        device,
        dtype,
        num_time_in_day=num_time_in_day,
        num_day_in_week=num_day_in_week,
    )
    dow_index = torch.floor(dow_norm * float(num_day_in_week)).clamp(0, num_day_in_week - 1)
    return tod, dow_index


def append_normalized_time_features(
    x: torch.Tensor,
    seq_time: Optional[torch.Tensor],
    num_time_in_day: int = 288,
    num_day_in_week: int = 7,
) -> torch.Tensor:
    batch_size, input_len, num_nodes, _ = x.shape
    tod, dow = normalized_tod_and_dow(
        seq_time,
        batch_size,
        input_len,
        num_nodes,
        x.device,
        x.dtype,
        num_time_in_day=num_time_in_day,
        num_day_in_week=num_day_in_week,
    )
    return torch.cat([x, tod.unsqueeze(-1), dow.unsqueeze(-1)], dim=-1)
