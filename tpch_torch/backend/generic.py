"""Generic PyTorch execution for a small SQL subset."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from numbers import Integral
from typing import Any

import duckdb
import torch

from tpch_torch.generic_sql import GenericFilter, GenericProjection, GenericSQLPlan
from tpch_torch.relational import decode
from tpch_torch.storage import TensorTable
from tpch_torch.errors import UnsupportedPlanError


def execute_generic_sql_plan(
    con: duckdb.DuckDBPyConnection,
    plan: GenericSQLPlan,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    table = _fetch_required_table(con, plan, device)
    mask = _filter_mask(table, plan.filters)
    if plan.group_by:
        rows = _execute_grouped(table, plan, mask)
    else:
        rows = _execute_ungrouped(table, plan, mask)
    rows = _sort_rows(rows, plan.order_by)
    if plan.limit is not None:
        return rows[: plan.limit]
    return rows


def _fetch_required_table(con: duckdb.DuckDBPyConnection, plan: GenericSQLPlan, device: str) -> TensorTable:
    _ensure_columns_exist(con, plan.table, plan.required_columns)
    columns = plan.required_columns or (_first_table_column(con, plan.table),)
    return _fetch_generic_tensor_table(con, plan.table, columns, device)


def _fetch_generic_tensor_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: tuple[str, ...],
    device: str,
) -> TensorTable:
    select_list = ", ".join(columns)
    rows = con.execute(f"select {select_list} from {table}").fetchall()
    by_column = {column: [row[index] for row in rows] for index, column in enumerate(columns)}
    encoded_columns: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}
    for column, values in by_column.items():
        tensor, vocabulary = _encode_generic_column(values, device)
        encoded_columns[column] = tensor
        if vocabulary is not None:
            dictionaries[column] = vocabulary
    return TensorTable(columns=encoded_columns, dictionaries=dictionaries)


def _encode_generic_column(values: list[Any], device: str) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if _is_string_column(values):
        vocabulary = tuple(sorted({str(value) for value in values}))
        ids = {value: index for index, value in enumerate(vocabulary)}
        return torch.tensor([ids[str(value)] for value in values], dtype=torch.int64, device=device), vocabulary
    if _is_date_column(values):
        return torch.tensor([_date_to_yyyymmdd(value) for value in values], dtype=torch.int32, device=device), None
    if _is_int_column(values):
        return torch.tensor([int(value) for value in values], dtype=torch.int64, device=device), None
    normalized = [float(value) if isinstance(value, Decimal) else value for value in values]
    return torch.tensor(normalized, dtype=torch.float64, device=device), None


def _is_string_column(values: list[Any]) -> bool:
    return any(isinstance(value, str) for value in values if value is not None)


def _is_date_column(values: list[Any]) -> bool:
    return any(isinstance(value, (date, datetime)) for value in values if value is not None)


def _is_int_column(values: list[Any]) -> bool:
    return all(isinstance(value, Integral) for value in values if value is not None)


def _ensure_columns_exist(con: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]) -> None:
    existing = {str(row[1]) for row in con.execute(f"pragma table_info('{table}')").fetchall()}
    missing = [column for column in columns if column not in existing]
    if missing:
        raise UnsupportedPlanError(f"generic SQL references missing column(s): {', '.join(missing)}")


def _first_table_column(con: duckdb.DuckDBPyConnection, table: str) -> str:
    row = con.execute(f"pragma table_info('{table}')").fetchone()
    if row is None:
        raise UnsupportedPlanError(f"generic SQL references missing table: {table}")
    return str(row[1])


def _filter_mask(table: TensorTable, filters: tuple[GenericFilter, ...]) -> torch.Tensor:
    row_count = len(table)
    device = next(iter(table.columns.values())).device
    mask = torch.ones(row_count, dtype=torch.bool, device=device)
    for filter_ in filters:
        mask &= _evaluate_filter(table, filter_)
    return mask


def _evaluate_filter(table: TensorTable, filter_: GenericFilter) -> torch.Tensor:
    values = table.columns[filter_.column]
    literal = _literal_tensor(table, filter_.column, filter_.value)
    if filter_.operator == "=":
        return values == literal
    if filter_.operator in {"!=", "<>"}:
        return values != literal
    if filter_.operator == ">":
        return values > literal
    if filter_.operator == ">=":
        return values >= literal
    if filter_.operator == "<":
        return values < literal
    if filter_.operator == "<=":
        return values <= literal
    raise UnsupportedPlanError(f"generic SQL filter operator is not supported: {filter_.operator}")


def _literal_tensor(table: TensorTable, column: str, value: int | float | str) -> torch.Tensor:
    values = table.columns[column]
    if isinstance(value, str):
        vocabulary = table.dictionaries.get(column)
        if vocabulary is None or value not in vocabulary:
            return torch.tensor(-1, dtype=values.dtype, device=values.device)
        return torch.tensor(vocabulary.index(value), dtype=values.dtype, device=values.device)
    return torch.tensor(value, dtype=values.dtype, device=values.device)


def _execute_ungrouped(table: TensorTable, plan: GenericSQLPlan, mask: torch.Tensor) -> list[dict[str, Any]]:
    if _is_scalar_aggregate(plan.projections):
        return [_aggregate_row(table, plan.projections, mask)]
    rows: list[dict[str, Any]] = []
    indices = mask.nonzero().flatten().cpu().tolist()
    for raw_index in indices:
        rows.append(_projection_row(table, plan.projections, int(raw_index)))
    return rows


def _execute_grouped(table: TensorTable, plan: GenericSQLPlan, mask: torch.Tensor) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    indices = mask.nonzero().flatten().cpu().tolist()
    for raw_index in indices:
        index = int(raw_index)
        key = tuple(_decode_cell(table, column, index) for column in plan.group_by)
        groups.setdefault(key, []).append(index)
    rows = []
    for key, group_indices in groups.items():
        rows.append(_grouped_row(table, plan.projections, plan.group_by, key, group_indices))
    return rows


def _projection_row(table: TensorTable, projections: tuple[GenericProjection, ...], index: int) -> dict[str, Any]:
    return {projection.alias: _evaluate_projection(table, projection, index) for projection in projections}


def _aggregate_row(
    table: TensorTable,
    projections: tuple[GenericProjection, ...],
    mask: torch.Tensor,
) -> dict[str, Any]:
    return {projection.alias: _evaluate_aggregate(table, projection, mask) for projection in projections}


def _grouped_row(
    table: TensorTable,
    projections: tuple[GenericProjection, ...],
    group_by: tuple[str, ...],
    key: tuple[Any, ...],
    indices: list[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    group_values = dict(zip(group_by, key))
    index_tensor = torch.tensor(indices, dtype=torch.int64, device=next(iter(table.columns.values())).device)
    for projection in projections:
        if projection.kind == "column" and projection.column in group_values:
            row[projection.alias] = group_values[projection.column]
            continue
        row[projection.alias] = _evaluate_group_aggregate(table, projection, index_tensor)
    return row


def _evaluate_projection(table: TensorTable, projection: GenericProjection, index: int) -> Any:
    if projection.kind == "column" and projection.column is not None:
        return _decode_cell(table, projection.column, index)
    if projection.kind == "mul_const" and projection.column is not None and projection.value is not None:
        value = table.columns[projection.column][index].cpu().item()
        return float(value) * projection.value
    raise UnsupportedPlanError(f"generic SQL projection requires aggregation or is unsupported: {projection.kind}")


def _evaluate_aggregate(table: TensorTable, projection: GenericProjection, mask: torch.Tensor) -> Any:
    if projection.kind == "count_star":
        return int(mask.sum().cpu().item())
    if projection.kind == "sum" and projection.column is not None:
        return float(table.columns[projection.column][mask].sum().cpu().item())
    raise UnsupportedPlanError(f"generic SQL aggregate is not supported: {projection.kind}")


def _evaluate_group_aggregate(table: TensorTable, projection: GenericProjection, indices: torch.Tensor) -> Any:
    if projection.kind == "count_star":
        return int(indices.numel())
    if projection.kind == "sum" and projection.column is not None:
        return float(table.columns[projection.column][indices].sum().cpu().item())
    raise UnsupportedPlanError(f"generic SQL grouped projection is not supported: {projection.kind}")


def _decode_cell(table: TensorTable, column: str, index: int) -> Any:
    value = table.columns[column][index].cpu().item()
    if column in table.dictionaries:
        return decode(table, column, int(value))
    if isinstance(value, float):
        return float(value)
    return int(value)


def _is_scalar_aggregate(projections: tuple[GenericProjection, ...]) -> bool:
    return all(projection.kind in {"count_star", "sum"} for projection in projections)


def _sort_rows(rows: list[dict[str, Any]], order_by: tuple[str, ...]) -> list[dict[str, Any]]:
    if not order_by:
        return rows
    return sorted(rows, key=lambda row: tuple(row[column] for column in order_by))


def _date_to_yyyymmdd(value: Any) -> int:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return (value.year * 10_000) + (value.month * 100) + value.day
    if isinstance(value, str):
        return int(value.replace("-", ""))
    if isinstance(value, Integral):
        return int(value)
    raise TypeError(f"unsupported date value: {value!r}")
