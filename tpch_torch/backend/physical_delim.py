"""DuckDB DELIM_SCAN / RIGHT_DELIM_JOIN helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from tpch_torch.backend.physical_expr import evaluate_expression, projection_name
from tpch_torch.backend.physical_join import anti_join_indices, combine_join_tables, semi_join_indices, semi_join_table
from tpch_torch.backend.physical_join_exec import join_conditions
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import TQPOperatorNode


def build_delim_table(
    outer: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> PhysicalTable:
    """Build a distinct correlated-key table for DuckDB DELIM_SCAN nodes."""

    key_exprs = tuple(left for left, _ in conditions)
    key_values = tuple(evaluate_expression(outer, expression) for expression in key_exprs)
    if not key_values:
        raise UnsupportedPlanError("RIGHT_DELIM_JOIN requires correlated key conditions")
    key_tensors = tuple(value.require_tensor() for value in key_values)
    unique_keys = _unique_key_rows(key_tensors)
    items = []
    for index, (expression, value) in enumerate(zip(key_exprs, key_values)):
        name, aliases = projection_name(outer, expression, index)
        tensor = unique_keys[:, index].to(dtype=value.require_tensor().dtype)
        items.append((name, PhysicalValue(tensor, value.dictionary, value.is_date), aliases))
    return PhysicalTable.projected("delim", items, int(unique_keys.shape[0]))


def execute_delim_join_result(
    node: TQPOperatorNode,
    outer: PhysicalTable,
    subquery: PhysicalTable,
    source_sql: str,
    required_columns: Sequence[str],
) -> PhysicalTable:
    """Combine a RIGHT_DELIM_JOIN outer child with its correlated subquery result."""

    join_type = _join_type(node)
    conditions = _correlated_conditions(subquery, join_conditions(node))
    if join_type == "RIGHT_SEMI":
        rows = semi_join_indices(outer, subquery, conditions)
        return semi_join_table(outer, rows, tuple(left for left, _ in conditions), source_sql, required_columns)
    if join_type == "RIGHT_ANTI":
        rows = anti_join_indices(outer, subquery, conditions)
        return semi_join_table(outer, rows, tuple(left for left, _ in conditions), source_sql, required_columns)
    if join_type == "RIGHT":
        return _add_subquery_alias(_left_join_subquery(outer, subquery, conditions, source_sql, required_columns), subquery)
    raise UnsupportedPlanError(f"unsupported delimiter join type: {join_type}")


def _left_join_subquery(
    outer: PhysicalTable,
    subquery: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
    source_sql: str,
    required_columns: Sequence[str],
) -> PhysicalTable:
    left_rows, right_rows = _inner_indices(outer, subquery, conditions)
    left_rows, right_rows, right_valid = _outer_join_indices(outer, left_rows, right_rows)
    return combine_join_tables(
        outer,
        subquery,
        left_rows,
        right_rows,
        tuple(left for left, _ in conditions),
        tuple(right for _, right in conditions),
        source_sql,
        required_columns,
        None,
        right_valid,
        "outer_first",
    )


def _add_subquery_alias(table: PhysicalTable, subquery: PhysicalTable) -> PhysicalTable:
    for name in subquery.order:
        if _matches_any_key(name, subquery.order[1:]):
            continue
        try:
            value = table.value_named(name)
        except KeyError:
            continue
        columns = dict(table.columns)
        columns.setdefault("SUBQUERY", value)
        return PhysicalTable(table.name, columns, table.order, table.row_count)
    return table


def _unique_key_rows(key_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    dtype = torch.float64 if any(tensor.dtype.is_floating_point for tensor in key_tensors) else torch.int64
    stacked = torch.stack([tensor.to(dtype=dtype) for tensor in key_tensors], dim=1)
    return torch.unique(stacked, dim=0, sorted=True)


def _inner_indices(
    outer: PhysicalTable,
    subquery: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    from tpch_torch.backend.physical_join import join_indices_for_conditions

    return join_indices_for_conditions(outer, subquery, conditions)


def _outer_join_indices(
    outer: PhysicalTable,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tpch_torch.backend.physical_join import outer_join_indices

    return outer_join_indices(outer.row_count, left_rows, right_rows)


def _matches_any_key(column: str, keys: Sequence[str]) -> bool:
    return any(column == key or column.rsplit(".", 1)[-1] == key.rsplit(".", 1)[-1] for key in keys)


def _join_type(node: TQPOperatorNode) -> str:
    value = node.metadata.get("Join Type")
    return str(value or "").upper()


def _correlated_conditions(
    subquery: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    rewritten = []
    for left, right in conditions:
        correlated = f"delim.{right}"
        rewritten.append((left, correlated if correlated in subquery.columns else right))
    return tuple(rewritten)
