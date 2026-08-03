"""Tensor final merge for partitionable aggregate fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from tpch_torch.backend.physical_aggregate import group_key_mapping
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.record_batch import ColumnMeta


@dataclass(frozen=True)
class FinalAggregateColumn:
    name: str
    function: str


@dataclass(frozen=True)
class FinalAggregatePlan:
    group_columns: tuple[str, ...]
    aggregates: tuple[FinalAggregateColumn, ...]
    count_column: str | None
    sort_by_group_keys: bool


def merge_partitioned_aggregate_tables(
    partial_tables: Sequence[PhysicalTable],
    plan: FinalAggregatePlan,
) -> PhysicalTable:
    """Merge local aggregate batches with tensor reductions."""

    if not partial_tables:
        return _empty_final_table(plan)
    combined = concat_physical_tables(partial_tables, name="partition_partial")
    if plan.group_columns:
        return _merge_grouped_partials(combined, plan)
    return _merge_ungrouped_partials(combined, plan)


def concat_physical_tables(tables: Sequence[PhysicalTable], *, name: str) -> PhysicalTable:
    """Concatenate tables with the same output schema."""

    if not tables:
        return PhysicalTable(name, {}, (), 0)
    order = tables[0].order
    _validate_compatible_tables(tables, order)
    row_count = sum(table.row_count for table in tables)
    items = [(_concat_column(column_name, tables), (column_name,)) for column_name in order]
    return PhysicalTable.projected(
        name,
        [(column_name, value, aliases) for (column_name, value), aliases in items],
        row_count,
    )


def _merge_grouped_partials(table: PhysicalTable, plan: FinalAggregatePlan) -> PhysicalTable:
    mapping = group_key_mapping(table, plan.group_columns)
    group_count = int(mapping.unique_keys.shape[0])
    items = _group_key_items(plan, mapping)
    items.extend(_aggregate_items(table, mapping.inverse, group_count, plan))
    result = PhysicalTable.projected("partition_final", items, group_count)
    return _sort_by_group_keys(result, plan.group_columns) if plan.sort_by_group_keys else result


def _merge_ungrouped_partials(table: PhysicalTable, plan: FinalAggregatePlan) -> PhysicalTable:
    device = _table_device(table)
    group_ids = torch.zeros(table.row_count, dtype=torch.int64, device=device)
    items = _aggregate_items(table, group_ids, 1, plan)
    return PhysicalTable.projected("partition_final", items, 1)


def _group_key_items(plan: FinalAggregatePlan, mapping) -> list[tuple[str, PhysicalValue, tuple[str, ...]]]:
    items = []
    for index, column in enumerate(plan.group_columns):
        source = mapping.key_values[index]
        key_tensor = mapping.unique_keys[:, index].to(dtype=source.require_tensor().dtype)
        value = PhysicalValue(
            key_tensor,
            dictionary=source.dictionary,
            is_date=source.is_date,
            sorted_non_decreasing=len(plan.group_columns) == 1 and mapping.keys_sorted,
            unique=len(plan.group_columns) == 1,
            meta=source.meta,
        )
        items.append((column, value, (column,)))
    return items


def _aggregate_items(
    table: PhysicalTable,
    group_ids: torch.Tensor,
    group_count: int,
    plan: FinalAggregatePlan,
) -> list[tuple[str, PhysicalValue, tuple[str, ...]]]:
    return [
        (aggregate.name, _merge_aggregate(table, group_ids, group_count, aggregate, plan), (aggregate.name,))
        for aggregate in plan.aggregates
    ]


def _merge_aggregate(
    table: PhysicalTable,
    group_ids: torch.Tensor,
    group_count: int,
    aggregate: FinalAggregateColumn,
    plan: FinalAggregatePlan,
) -> PhysicalValue:
    if aggregate.function in {"sum", "count", "count_star"}:
        return _merge_sum_like(
            table.value_named(aggregate.name),
            group_ids,
            group_count,
            count_like=aggregate.function in {"count", "count_star"},
        )
    if aggregate.function == "min":
        return _merge_min_max(table.value_named(aggregate.name), group_ids, group_count, "amin")
    if aggregate.function == "max":
        return _merge_min_max(table.value_named(aggregate.name), group_ids, group_count, "amax")
    if aggregate.function == "avg":
        return _merge_weighted_avg(table, group_ids, group_count, aggregate.name, plan.count_column)
    raise UnsupportedPlanError(f"unsupported tensor final aggregate: {aggregate.function}")


def _merge_sum_like(
    value: PhysicalValue,
    group_ids: torch.Tensor,
    group_count: int,
    *,
    count_like: bool = False,
) -> PhysicalValue:
    tensor = value.require_tensor()
    valid = _validity_or_ones(value, tensor)
    safe_values = torch.where(valid, tensor, torch.zeros_like(tensor))
    result = _scatter_sum(safe_values, group_ids, group_count)
    counts = _scatter_sum(valid.to(dtype=torch.int64), group_ids, group_count)
    return PhysicalValue(result, valid=None if count_like else counts > 0, meta=value.meta)


def _merge_min_max(
    value: PhysicalValue,
    group_ids: torch.Tensor,
    group_count: int,
    reduce: str,
) -> PhysicalValue:
    tensor = value.require_tensor()
    valid = _validity_or_ones(value, tensor)
    fill = _reduce_fill_value(tensor.dtype, reduce)
    safe_values = torch.where(valid, tensor, torch.full_like(tensor, fill))
    counts = _scatter_sum(valid.to(dtype=torch.int64), group_ids, group_count)
    result = _scatter_reduce(safe_values, group_ids, group_count, reduce)
    return PhysicalValue(result, valid=counts > 0, meta=value.meta)


def _merge_weighted_avg(
    table: PhysicalTable,
    group_ids: torch.Tensor,
    group_count: int,
    value_column: str,
    count_column: str | None,
) -> PhysicalValue:
    if count_column is None:
        raise UnsupportedPlanError("tensor final AVG requires count_column")
    value = table.value_named(value_column)
    counts = table.value_named(count_column).require_tensor().to(dtype=torch.float64)
    valid = _validity_or_ones(value, value.require_tensor()) & (counts > 0)
    weighted = value.require_tensor().to(dtype=torch.float64) * counts
    numerator = _scatter_sum(torch.where(valid, weighted, torch.zeros_like(weighted)), group_ids, group_count)
    denominator = _scatter_sum(torch.where(valid, counts, torch.zeros_like(counts)), group_ids, group_count)
    result = numerator / torch.clamp(denominator, min=1.0)
    return PhysicalValue(result, valid=denominator > 0, meta=ColumnMeta.fp64())


def _empty_final_table(plan: FinalAggregatePlan) -> PhysicalTable:
    if plan.group_columns:
        return PhysicalTable("partition_final", {}, (), 0)
    items = [
        (aggregate.name, _empty_aggregate_value(aggregate), (aggregate.name,))
        for aggregate in plan.aggregates
    ]
    return PhysicalTable.projected("partition_final", items, 1)


def _empty_aggregate_value(aggregate: FinalAggregateColumn) -> PhysicalValue:
    if aggregate.function in {"count", "count_star"}:
        return PhysicalValue(torch.zeros(1, dtype=torch.int64))
    tensor = torch.zeros(1, dtype=torch.float64)
    valid = torch.zeros(1, dtype=torch.bool)
    return PhysicalValue(tensor, valid=valid, meta=ColumnMeta.fp64() if aggregate.function == "avg" else None)


def _concat_column(column_name: str, tables: Sequence[PhysicalTable]) -> tuple[str, PhysicalValue]:
    values = [table.value_named(column_name) for table in tables]
    _validate_compatible_values(column_name, values)
    tensors = [value.require_tensor() for value in values]
    validities = [value.valid for value in values]
    validity = None if all(valid is None for valid in validities) else _concat_validities(tensors, validities)
    first = values[0]
    return column_name, PhysicalValue(
        torch.cat(tensors, dim=0),
        dictionary=first.dictionary,
        is_date=first.is_date,
        valid=validity,
        meta=first.meta,
    )


def _concat_validities(
    tensors: Sequence[torch.Tensor],
    validities: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    pieces = [
        torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device) if valid is None else valid
        for tensor, valid in zip(tensors, validities)
    ]
    return torch.cat(pieces, dim=0)


def _sort_by_group_keys(table: PhysicalTable, group_columns: Sequence[str]) -> PhysicalTable:
    result = table
    for column in reversed(tuple(group_columns)):
        value = result.value_named(column)
        order = torch.argsort(_sort_key(value), stable=True)
        result = result.gather(order)
    return result


def _sort_key(value: PhysicalValue) -> torch.Tensor:
    tensor = value.require_tensor()
    if value.dictionary is None:
        return tensor
    ranks = sorted(range(len(value.dictionary)), key=lambda index: value.dictionary[index])
    rank_map = torch.empty(len(ranks), dtype=torch.int64, device=tensor.device)
    rank_map[torch.tensor(ranks, dtype=torch.int64, device=tensor.device)] = torch.arange(
        len(ranks), dtype=torch.int64, device=tensor.device
    )
    return rank_map[tensor.to(dtype=torch.int64)]


def _validate_compatible_tables(tables: Sequence[PhysicalTable], order: tuple[str, ...]) -> None:
    for table in tables[1:]:
        if table.order != order:
            raise UnsupportedPlanError("partition final merge requires identical partial schemas")


def _validate_compatible_values(column_name: str, values: Sequence[PhysicalValue]) -> None:
    first = values[0]
    for value in values[1:]:
        if value.dictionary != first.dictionary:
            raise UnsupportedPlanError(f"cannot merge changed dictionary for column: {column_name}")
        if value.require_tensor().dtype != first.require_tensor().dtype:
            raise UnsupportedPlanError(f"cannot merge changed dtype for column: {column_name}")


def _validity_or_ones(value: PhysicalValue, tensor: torch.Tensor) -> torch.Tensor:
    if value.valid is not None:
        return value.valid
    return torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)


def _scatter_sum(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    return result.index_add(0, group_ids.to(dtype=torch.int64), values)


def _scatter_reduce(values: torch.Tensor, group_ids: torch.Tensor, group_count: int, reduce: str) -> torch.Tensor:
    result = torch.full((group_count,), _reduce_fill_value(values.dtype, reduce), dtype=values.dtype, device=values.device)
    return result.scatter_reduce(0, group_ids.to(dtype=torch.int64), values, reduce=reduce, include_self=True)


def _reduce_fill_value(dtype: torch.dtype, reduce: str) -> int | float:
    if dtype.is_floating_point:
        return float("inf") if reduce == "amin" else float("-inf")
    info = torch.iinfo(dtype)
    return info.max if reduce == "amin" else info.min


def _table_device(table: PhysicalTable) -> torch.device:
    for value in table.columns.values():
        return value.require_tensor().device
    return torch.device("cpu")
