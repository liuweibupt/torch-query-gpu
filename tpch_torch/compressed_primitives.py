"""Low-level tensor primitives for lightweight-compressed execution."""

from __future__ import annotations

import torch

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def range_arange(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Generate concatenated `arange(start, start + length)` segments."""

    _validate_segment_inputs(starts, lengths)
    total_size = int(lengths.sum().cpu().item())
    if total_size == 0:
        return torch.empty(0, dtype=torch.int64, device=starts.device)
    offsets = torch.cumsum(lengths.to(dtype=torch.int64), dim=0) - lengths.to(dtype=torch.int64)
    return (
        torch.repeat_interleave(starts.to(dtype=torch.int64), lengths.to(dtype=torch.int64))
        + torch.arange(total_size, dtype=torch.int64, device=starts.device)
        - torch.repeat_interleave(offsets, lengths.to(dtype=torch.int64))
    )


def _validate_segment_inputs(starts: torch.Tensor, lengths: torch.Tensor) -> None:
    if starts.ndim != 1 or lengths.ndim != 1:
        raise ValueError("starts and lengths must be 1-D tensors")
    if starts.shape != lengths.shape:
        raise ValueError("starts and lengths must have the same shape")
    if starts.device != lengths.device:
        raise ValueError("starts and lengths must be on the same device")
    if starts.dtype not in _INTEGER_DTYPES or lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError("starts and lengths must use integer dtypes")
    if starts.numel() == 0:
        return
    if bool(torch.any(starts < 0).cpu().item()) or bool(torch.any(lengths < 0).cpu().item()):
        raise ValueError("starts and lengths must be non-negative")
