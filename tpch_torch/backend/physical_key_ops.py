"""Shared key comparison helpers for physical joins and membership probes."""

from __future__ import annotations

import torch

from tpch_torch.backend.physical_types import PhysicalValue


def comparable_key_tensors(
    left_key: torch.Tensor,
    right_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce join keys to a shared dtype without truncating fractional values."""

    common = _common_key_dtype(left_key.dtype, right_key.dtype)
    return left_key.to(dtype=common), right_key.to(dtype=common)


def pairwise_equal_values(
    left_value: PhysicalValue,
    right_value: PhysicalValue,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
) -> torch.Tensor:
    """Compare already paired rows while respecting optional validity masks."""

    left_tensor = left_value.require_tensor().index_select(0, left_rows)
    right_tensor = right_value.require_tensor().index_select(0, right_rows)
    left_key, right_key = comparable_key_tensors(left_tensor, right_tensor)
    matched = left_key == right_key
    if left_value.valid is not None:
        matched = matched & left_value.valid.index_select(0, left_rows).to(device=matched.device)
    if right_value.valid is not None:
        matched = matched & right_value.valid.index_select(0, right_rows).to(device=matched.device)
    return matched


def _common_key_dtype(left: torch.dtype, right: torch.dtype) -> torch.dtype:
    if left.is_floating_point or right.is_floating_point:
        return torch.float64
    return torch.promote_types(left, right)
