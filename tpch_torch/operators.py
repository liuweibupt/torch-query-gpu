"""Reusable tensor operators for the supported analytical query subset."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def composite_group_ids(key_columns: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row inverse group ids and unique composite key rows."""

    if not key_columns:
        raise ValueError("at least one key column is required")
    stacked_keys = torch.stack(tuple(key_columns), dim=1).to(dtype=torch.int64)
    unique_keys, inverse = torch.unique(stacked_keys, dim=0, sorted=True, return_inverse=True)
    return inverse.to(dtype=torch.int64), unique_keys


def grouped_sum(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Sum `values` per group id using tensor scatter/index_add."""

    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    result.index_add_(0, group_ids, values)
    return result


def grouped_count(group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Count rows per group id."""

    ones = torch.ones(group_ids.numel(), dtype=torch.int64, device=group_ids.device)
    result = torch.zeros(group_count, dtype=torch.int64, device=group_ids.device)
    result.index_add_(0, group_ids, ones)
    return result
