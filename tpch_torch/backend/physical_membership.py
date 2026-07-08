"""Membership probes for physical SEMI/ANTI joins."""

from __future__ import annotations

from typing import Sequence

import torch

from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_key_ops import comparable_key_tensors
from tpch_torch.backend.physical_types import PhysicalTable


def membership_join_indices(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
    *,
    matched: bool,
) -> torch.Tensor:
    """Return preserved-side row indices selected by SEMI/ANTI membership."""

    if left.row_count == 0:
        return _empty(left)
    if right.row_count == 0:
        return _preserved_rows(left) if not matched else _empty(left)
    left_key, right_key = _membership_keys(left, right, conditions)
    mask = _membership_mask(left_key, right_key)
    return torch.nonzero(mask if matched else ~mask).flatten().to(dtype=torch.int64)


def _membership_keys(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(conditions) == 1:
        left_expr, right_expr = conditions[0]
        return comparable_key_tensors(
            evaluate_expression(left, left_expr).require_tensor(),
            evaluate_expression(right, right_expr).require_tensor(),
        )
    left_stacked = _stack_key_columns(left, tuple(left for left, _ in conditions))
    right_stacked = _stack_key_columns(right, tuple(right for _, right in conditions))
    _, inverse = torch.unique(torch.cat((left_stacked, right_stacked), dim=0), dim=0, sorted=True, return_inverse=True)
    return inverse[: left.row_count], inverse[left.row_count :]


def _membership_mask(left_key: torch.Tensor, right_key: torch.Tensor) -> torch.Tensor:
    _validate_key_tensors(left_key, right_key)
    sorted_right = (
        right_key.contiguous()
        if _is_sorted_non_decreasing(right_key)
        else right_key[torch.argsort(right_key)]
    )
    positions = torch.searchsorted(sorted_right, left_key)
    in_bounds = positions < sorted_right.numel()
    safe_positions = torch.clamp(positions, max=max(int(sorted_right.numel()) - 1, 0))
    return in_bounds & (sorted_right[safe_positions] == left_key)


def _stack_key_columns(table: PhysicalTable, expressions: tuple[str, ...]) -> torch.Tensor:
    columns = [
        evaluate_expression(table, expression).require_tensor().to(dtype=torch.int64)
        for expression in expressions
    ]
    return torch.stack(columns, dim=1)


def _validate_key_tensors(left_key: torch.Tensor, right_key: torch.Tensor) -> None:
    if left_key.ndim != 1 or right_key.ndim != 1:
        raise ValueError("membership join keys must be 1-D tensors")
    if left_key.device != right_key.device:
        raise ValueError("membership join keys must be on the same device")


def _is_sorted_non_decreasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return not bool(torch.any(values[1:] < values[:-1]).cpu().item())


def _empty(table: PhysicalTable) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=_table_device(table))


def _preserved_rows(table: PhysicalTable) -> torch.Tensor:
    return torch.arange(table.row_count, dtype=torch.int64, device=_table_device(table))


def _table_device(table: PhysicalTable) -> torch.device:
    for value in table.columns.values():
        if value.tensor is not None:
            return value.tensor.device
    return torch.device("cpu")
