"""Shared key comparison helpers for physical joins and membership probes."""

from __future__ import annotations

import torch

from tpch_torch.backend.physical_types import PhysicalValue
from tpch_torch.backend.type_mapping import align_decimal_tensors
from tpch_torch.record_batch import LogicalDType


def comparable_key_tensors(
    left_key: torch.Tensor,
    right_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce join keys to a shared dtype without truncating fractional values."""

    common = _common_key_dtype(left_key.dtype, right_key.dtype)
    return left_key.to(dtype=common), right_key.to(dtype=common)


def comparable_value_tensors(
    left_value: PhysicalValue,
    right_value: PhysicalValue,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_decimal_value(left_value) and _is_decimal_value(right_value):
        left, right, _ = align_decimal_tensors(
            left_value.require_tensor(),
            left_value.meta,
            right_value.require_tensor(),
            right_value.meta,
        )
        return left, right
    return comparable_key_tensors(left_value.require_tensor(), right_value.require_tensor())


def pairwise_equal_values(
    left_value: PhysicalValue,
    right_value: PhysicalValue,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
) -> torch.Tensor:
    """Compare already paired rows while respecting optional validity masks."""

    left_key, right_key = comparable_value_tensors(
        left_value.gather(left_rows),
        right_value.gather(right_rows),
    )
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


def _is_decimal_value(value: PhysicalValue) -> bool:
    return value.meta is not None and value.meta.logical_dtype == LogicalDType.DECIMAL
