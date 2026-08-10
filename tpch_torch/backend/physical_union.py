"""Set-operation helpers for DuckDB physical UNION nodes."""

from __future__ import annotations

from typing import Sequence

import torch

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue, table_device
from tpch_torch.errors import UnsupportedPlanError


def execute_union_node(children: Sequence[PhysicalTable]) -> PhysicalTable:
    """Concatenate UNION ALL children by output position.

    DuckDB represents DISTINCT UNION as UNION followed by HASH_GROUP_BY, so this
    node only implements the append part.  Type coercion must already have been
    planned by DuckDB; incompatible tensor metadata is reported explicitly.
    """

    tables = tuple(children)
    if len(tables) < 2:
        raise UnsupportedPlanError("UNION expects at least two children")
    _validate_arity(tables)
    row_count = sum(table.row_count for table in tables)
    items = []
    for index, name in enumerate(tables[0].order):
        values = tuple(table.value_at(index) for table in tables)
        items.append((name, _concat_values(values, tables), (name,)))
    return PhysicalTable.projected("union", items, row_count)


def _validate_arity(tables: Sequence[PhysicalTable]) -> None:
    arity = len(tables[0].order)
    for table in tables[1:]:
        if len(table.order) != arity:
            raise UnsupportedPlanError("UNION children have different output arity")


def _concat_values(values: Sequence[PhysicalValue], tables: Sequence[PhysicalTable]) -> PhysicalValue:
    first = values[0]
    _validate_compatible_values(values)
    tensors = tuple(value.require_tensor() for value in values)
    valid = _concat_validity(values, tables)
    return PhysicalValue(
        torch.cat(tensors, dim=0),
        dictionary=first.dictionary,
        is_date=first.is_date,
        valid=valid,
        meta=first.meta,
    )


def _validate_compatible_values(values: Sequence[PhysicalValue]) -> None:
    first = values[0]
    first_tensor = first.require_tensor()
    for value in values[1:]:
        tensor = value.require_tensor()
        if tensor.dtype != first_tensor.dtype:
            raise UnsupportedPlanError(f"UNION column dtype mismatch: {first_tensor.dtype} != {tensor.dtype}")
        if tensor.device != first_tensor.device:
            raise UnsupportedPlanError("UNION column tensors must be on the same device")
        if value.dictionary != first.dictionary:
            raise UnsupportedPlanError("UNION dictionary columns require identical dictionaries")
        if value.is_date != first.is_date or value.meta != first.meta:
            raise UnsupportedPlanError("UNION column logical metadata mismatch")


def _concat_validity(values: Sequence[PhysicalValue], tables: Sequence[PhysicalTable]) -> torch.Tensor | None:
    if all(value.valid is None for value in values):
        return None
    masks = []
    for value, table in zip(values, tables):
        if value.valid is not None:
            masks.append(value.valid)
            continue
        masks.append(torch.ones(table.row_count, dtype=torch.bool, device=table_device(table)))
    return torch.cat(tuple(masks), dim=0)
