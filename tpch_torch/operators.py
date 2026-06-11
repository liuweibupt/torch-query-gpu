"""Reusable tensor operators for the supported analytical query subset."""

from __future__ import annotations

from collections.abc import Sequence

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




def low_cardinality_group_ids(
    key_columns: Sequence[torch.Tensor], cardinalities: Sequence[int]
) -> tuple[torch.Tensor, int]:
    """Encode dense low-cardinality key columns as composite group ids."""

    if not key_columns:
        raise ValueError("at least one key column is required")
    if len(key_columns) != len(cardinalities):
        raise ValueError("key_columns and cardinalities must have the same length")
    first = key_columns[0]
    _validate_low_cardinality_inputs(key_columns, cardinalities, first)
    group_ids = torch.zeros(first.shape, dtype=torch.int64, device=first.device)
    multiplier = 1
    for key, cardinality in reversed(tuple(zip(key_columns, cardinalities))):
        group_ids = group_ids + key.to(dtype=torch.int64) * multiplier
        multiplier *= cardinality
    return group_ids, multiplier


def grouped_sum_bincount(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Sum values per dense group id using torch.bincount."""

    _validate_grouped_reduction_inputs(values, group_ids, group_count)
    result = torch.bincount(group_ids.to(dtype=torch.int64), weights=values, minlength=group_count)
    return result[:group_count].to(dtype=values.dtype)


def grouped_count_bincount(group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Count rows per dense group id using torch.bincount."""

    if group_ids.ndim != 1:
        raise ValueError("group_ids must be 1-D")
    if group_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("group_ids must use an integer dtype")
    if not isinstance(group_count, int) or group_count < 0:
        raise ValueError("group_count must be a non-negative integer")
    if group_ids.numel() == 0:
        return torch.zeros(group_count, dtype=torch.int64, device=group_ids.device)
    if bool(torch.any(group_ids < 0).cpu().item()) or bool(torch.any(group_ids >= group_count).cpu().item()):
        raise ValueError("group_ids contain values out of range")
    return torch.bincount(group_ids.to(dtype=torch.int64), minlength=group_count)[:group_count]


def logical_and_all(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the element-wise conjunction of boolean masks."""

    checked_masks = _validate_masks(masks)
    result = checked_masks[0].clone()
    for mask in checked_masks[1:]:
        result = torch.logical_and(result, mask)
    return result


def logical_or_all(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the element-wise disjunction of boolean masks."""

    checked_masks = _validate_masks(masks)
    result = checked_masks[0].clone()
    for mask in checked_masks[1:]:
        result = torch.logical_or(result, mask)
    return result


def gather_by_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select rows from the first dimension using a boolean mask."""

    if mask.dtype is not torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.ndim != 1:
        raise ValueError("mask must be a 1-D tensor")
    if values.ndim == 0 or values.shape[0] != mask.numel():
        raise ValueError("mask length must match values first dimension")
    return values[mask]


def grouped_min(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Compute the minimum value per group id."""

    _validate_grouped_reduction_inputs(values, group_ids, group_count)
    _require_non_empty_groups(group_ids, group_count)
    result = torch.full(
        (group_count,),
        _max_value(values.dtype),
        dtype=values.dtype,
        device=values.device,
    )
    return result.scatter_reduce(0, group_ids, values, reduce="amin", include_self=True)


def grouped_max(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Compute the maximum value per group id."""

    _validate_grouped_reduction_inputs(values, group_ids, group_count)
    _require_non_empty_groups(group_ids, group_count)
    result = torch.full(
        (group_count,),
        _min_value(values.dtype),
        dtype=values.dtype,
        device=values.device,
    )
    return result.scatter_reduce(0, group_ids, values, reduce="amax", include_self=True)


def grouped_mean(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    """Compute the arithmetic mean per group id."""

    _validate_grouped_reduction_inputs(values, group_ids, group_count)
    _require_non_empty_groups(group_ids, group_count)
    mean_values = values if torch.is_floating_point(values) else values.to(dtype=torch.float64)
    sums = grouped_sum(mean_values, group_ids, group_count)
    counts = grouped_count(group_ids, group_count).to(dtype=sums.dtype)
    return sums / counts


def topk_indices(values: torch.Tensor, k: int, descending: bool = True) -> torch.Tensor:
    """Return row indices for the top or bottom `k` values in sorted value order."""

    if values.ndim != 1:
        raise ValueError("topk_indices expects a 1-D tensor")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k > values.numel():
        raise ValueError("k cannot exceed the number of values")
    if k == 0:
        return torch.empty(0, dtype=torch.int64, device=values.device)
    return torch.topk(values, k, largest=descending, sorted=True).indices.to(dtype=torch.int64)



def _validate_low_cardinality_inputs(
    key_columns: Sequence[torch.Tensor], cardinalities: Sequence[int], first: torch.Tensor
) -> None:
    if first.ndim != 1:
        raise ValueError("key columns must be 1-D")
    for key, cardinality in zip(key_columns, cardinalities):
        if key.ndim != 1:
            raise ValueError("key columns must be 1-D")
        if key.shape != first.shape:
            raise ValueError("key columns must have the same shape")
        if key.device != first.device:
            raise ValueError("key columns must be on the same device")
        if key.dtype not in _INTEGER_DTYPES:
            raise TypeError("key columns must use integer dtypes")
        if not isinstance(cardinality, int) or cardinality <= 0:
            raise ValueError("cardinalities must be positive integers")
        if key.numel() == 0:
            continue
        out_of_range = torch.logical_or(key < 0, key >= cardinality)
        if bool(torch.any(out_of_range).cpu().item()):
            raise ValueError("key values contain values out of cardinality range")


def _validate_masks(masks: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    checked_masks = tuple(masks)
    if not checked_masks:
        raise ValueError("at least one mask is required")
    first = checked_masks[0]
    for mask in checked_masks:
        if mask.dtype is not torch.bool:
            raise TypeError("all masks must be boolean tensors")
        if mask.shape != first.shape:
            raise ValueError("all masks must have the same shape")
        if mask.device != first.device:
            raise ValueError("all masks must be on the same device")
    return checked_masks


def _validate_grouped_reduction_inputs(
    values: torch.Tensor, group_ids: torch.Tensor, group_count: int
) -> None:
    if values.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("values and group_ids must be 1-D tensors")
    if values.numel() != group_ids.numel():
        raise ValueError("values and group_ids must have the same length")
    if values.device != group_ids.device:
        raise ValueError("values and group_ids must be on the same device")
    if group_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("group_ids must use an integer dtype")
    if not isinstance(group_count, int) or group_count < 0:
        raise ValueError("group_count must be a non-negative integer")
    if values.dtype not in _INTEGER_DTYPES and not torch.is_floating_point(values):
        raise TypeError("values must use an integer or floating-point dtype")
    if group_ids.numel() == 0:
        return
    if bool(torch.any(group_ids < 0).cpu().item()) or bool(torch.any(group_ids >= group_count).cpu().item()):
        raise ValueError("group_ids contain values out of range")


def _require_non_empty_groups(group_ids: torch.Tensor, group_count: int) -> None:
    counts = grouped_count(group_ids, group_count)
    missing = torch.nonzero(counts == 0).flatten()
    if missing.numel() == 0:
        return
    missing_groups = missing.cpu().tolist()
    raise ValueError(f"grouped reduction received empty group(s): {missing_groups}")


def _max_value(dtype: torch.dtype) -> float | int:
    if dtype in _INTEGER_DTYPES:
        return torch.iinfo(dtype).max
    return float("inf")


def _min_value(dtype: torch.dtype) -> float | int:
    if dtype in _INTEGER_DTYPES:
        return torch.iinfo(dtype).min
    return float("-inf")
