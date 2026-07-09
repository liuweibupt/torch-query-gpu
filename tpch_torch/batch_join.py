"""Typed TensorRecordBatch join helpers."""

from __future__ import annotations

import torch

from tpch_torch.backend.physical_join import inner_join_indices_for_values
from tpch_torch.backend.physical_types import PhysicalValue
from tpch_torch.record_batch import ColumnType, TensorRecordBatch


def inner_join_indices_batch(
    left: TensorRecordBatch,
    right: TensorRecordBatch,
    *,
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(left_keys) != len(right_keys):
        raise ValueError("left_keys and right_keys must have equal length")
    if not left_keys:
        raise ValueError("at least one join key is required")
    left_rows, right_rows = _join_first_key(left, right, left_keys[0], right_keys[0])
    for left_key, right_key in zip(left_keys[1:], right_keys[1:], strict=True):
        keep = _rows_equal(left, right, left_key, right_key, left_rows, right_rows)
        left_rows = left_rows[keep]
        right_rows = right_rows[keep]
    return left_rows.to(device=left.batch_meta.device), right_rows.to(device=right.batch_meta.device)


def _join_first_key(
    left: TensorRecordBatch,
    right: TensorRecordBatch,
    left_key: str,
    right_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return inner_join_indices_for_values(
        _physical_value(left, left_key),
        _physical_value(right, right_key),
    )


def _rows_equal(
    left: TensorRecordBatch,
    right: TensorRecordBatch,
    left_key: str,
    right_key: str,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
) -> torch.Tensor:
    left_values = left.columns[left_key].index_select(0, left_rows)
    right_values = right.columns[right_key].index_select(0, right_rows)
    return left_values == right_values


def _physical_value(batch: TensorRecordBatch, key: str) -> PhysicalValue:
    return PhysicalValue(
        tensor=batch.columns[key],
        dictionary=batch.storage[key].dictionary,
        valid=batch.validity.get(key),
        meta=_column_meta(batch.types[key], batch.storage[key].dictionary),
    )


def _column_meta(column_type: ColumnType, dictionary: tuple[str, ...] | None):
    return column_type.to_column_meta(dictionary)
