"""Tensor join kernels for DuckDB physical-plan execution."""

from __future__ import annotations

import torch


def inner_join_indices(left_key: torch.Tensor, right_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching row indices for an inner equi-join without host key materialization."""

    _validate_join_keys(left_key, right_key)
    device = left_key.device
    if left_key.numel() == 0 or right_key.numel() == 0:
        return _empty_indices(device)

    left_values = left_key.to(dtype=torch.int64)
    right_values = right_key.to(dtype=torch.int64)
    right_order, sorted_right_values = _sorted_build_keys(right_values)
    starts = torch.searchsorted(sorted_right_values, left_values, right=False)
    ends = torch.searchsorted(sorted_right_values, left_values, right=True)
    match_counts = ends - starts
    if _is_strictly_increasing(sorted_right_values):
        return _unique_build_join_indices(starts, match_counts, right_order)
    match_count = int(match_counts.sum().cpu().item())
    if match_count == 0:
        return _empty_indices(device)

    left_rows = torch.repeat_interleave(
        torch.arange(left_values.numel(), dtype=torch.int64, device=device),
        match_counts,
    )
    right_positions = _matching_sorted_positions(starts, match_counts, match_count)
    right_rows = right_positions if right_order is None else right_order[right_positions]
    return left_rows, right_rows


def _unique_build_join_indices(
    starts: torch.Tensor,
    match_counts: torch.Tensor,
    right_order: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched = match_counts > 0
    left_rows = torch.nonzero(matched).flatten().to(dtype=torch.int64)
    right_positions = starts[matched].to(dtype=torch.int64)
    right_rows = right_positions if right_order is None else right_order[right_positions]
    return left_rows, right_rows


def _sorted_build_keys(right_values: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    if _is_sorted_non_decreasing(right_values):
        return None, right_values
    right_order = torch.argsort(right_values, stable=True)
    return right_order, right_values[right_order]


def _is_sorted_non_decreasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return not bool(torch.any(values[1:] < values[:-1]).cpu().item())


def _is_strictly_increasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return bool(torch.all(values[1:] > values[:-1]).cpu().item())


def _matching_sorted_positions(
    starts: torch.Tensor,
    match_counts: torch.Tensor,
    match_count: int,
) -> torch.Tensor:
    segment_offsets = torch.cumsum(match_counts, dim=0) - match_counts
    repeated_starts = torch.repeat_interleave(starts, match_counts)
    repeated_offsets = torch.repeat_interleave(segment_offsets, match_counts)
    local_offsets = torch.arange(match_count, dtype=torch.int64, device=starts.device)
    return repeated_starts + local_offsets - repeated_offsets


def _validate_join_keys(left_key: torch.Tensor, right_key: torch.Tensor) -> None:
    if left_key.ndim != 1 or right_key.ndim != 1:
        raise ValueError("physical join keys must be 1-D tensors")
    if left_key.device != right_key.device:
        raise ValueError("physical join keys must be on the same device")


def _empty_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    return empty, empty.clone()
