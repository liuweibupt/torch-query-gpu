"""Generic PyTorch execution for a small SQL subset."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from numbers import Integral
from typing import Any, Iterable

import duckdb
import numpy as np
import torch

from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.generic_sql import GenericFilter, GenericOrderBy, GenericProjection, GenericSQLPlan
from tpch_torch.operators import (
    composite_group_ids,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
    membership_mask,
)
from tpch_torch.backend.static_dictionaries import static_string_dictionary
from tpch_torch.relational import DATE_COLUMNS_EXTENDED, STRING_COLUMNS_EXTENDED, decode
from tpch_torch.storage import TensorTable


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
    columnar = con.execute(f"select {select_list} from {table}").fetchnumpy()
    encoded_columns: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}
    for column, values in columnar.items():
        tensor, vocabulary = _encode_generic_column(values, device, column_name=column, table_name=table)
        encoded_columns[column] = tensor
        if vocabulary is not None:
            dictionaries[column] = vocabulary
    return TensorTable(columns=encoded_columns, dictionaries=dictionaries)


def _encode_generic_column(
    values: Iterable[Any] | np.ndarray,
    device: str,
    column_name: str | None = None,
    table_name: str | None = None,
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if isinstance(values, np.ndarray):
        return _encode_numpy_generic_column(values, device, column_name, table_name)
    return _encode_generic_sequence(list(values), device)


def _encode_numpy_generic_column(
    values: np.ndarray,
    device: str,
    column_name: str | None = None,
    table_name: str | None = None,
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    static_dictionary = static_string_dictionary(table_name, column_name)
    if static_dictionary is not None:
        return _encode_numpy_static_string_column(values, static_dictionary, device), static_dictionary
    if _is_known_date_column(column_name):
        return _encode_known_numpy_date_column(values, device), None
    if _is_known_string_column(column_name) or values.dtype.kind in {"U", "S"} or _is_numpy_string_object_array(values):
        return _encode_numpy_string_column(values, device)
    if np.issubdtype(values.dtype, np.datetime64):
        return _encode_numpy_datetime64_column(values, device), None
    if values.dtype == np.dtype("O"):
        return _encode_generic_sequence(values.tolist(), device)
    if values.dtype.kind in {"i", "u"}:
        return torch.as_tensor(values, dtype=torch.int64, device=device), None
    return torch.as_tensor(values, dtype=torch.float64, device=device), None


def _encode_numpy_string_column(
    values: np.ndarray, device: str
) -> tuple[torch.Tensor, tuple[str, ...]]:
    vocabulary, inverse = np.unique(values.astype(str), return_inverse=True)
    tensor = torch.as_tensor(inverse, dtype=torch.int64, device=device)
    return tensor, tuple(str(value) for value in np.asarray(vocabulary))


def _encode_numpy_static_string_column(
    values: np.ndarray,
    vocabulary: tuple[str, ...],
    device: str,
) -> torch.Tensor:
    codes = np.full(values.shape, -1, dtype=np.int64)
    for index, literal in enumerate(vocabulary):
        codes[values == literal] = index
    if bool(np.any(codes < 0)):
        missing_values = np.unique(values[codes < 0].astype(str))
        formatted = ", ".join(str(value) for value in missing_values[:5])
        raise UnsupportedPlanError(f"static dictionary does not cover value(s): {formatted}")
    return torch.as_tensor(codes, dtype=torch.int64, device=device)


def _encode_known_numpy_date_column(values: np.ndarray, device: str) -> torch.Tensor:
    if np.issubdtype(values.dtype, np.datetime64):
        return _encode_numpy_datetime64_column(values, device)
    if values.dtype.kind in {"i", "u"}:
        return torch.as_tensor(values, dtype=torch.int32, device=device)
    encoded = [_date_to_yyyymmdd(value) for value in values.tolist()]
    return torch.tensor(encoded, dtype=torch.int32, device=device)


def _encode_numpy_datetime64_column(values: np.ndarray, device: str) -> torch.Tensor:
    dates = values.astype("datetime64[D]").astype(np.int64)
    year = dates.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    month = dates.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    day = dates - values.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + 1
    encoded = (year * 10_000 + month * 100 + day).astype(np.int32, copy=False)
    return torch.as_tensor(encoded, dtype=torch.int32, device=device)


def _encode_generic_sequence(values: list[Any], device: str) -> tuple[torch.Tensor, tuple[str, ...] | None]:
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


def _is_known_string_column(column_name: str | None) -> bool:
    return column_name in STRING_COLUMNS_EXTENDED


def _is_known_date_column(column_name: str | None) -> bool:
    return column_name in DATE_COLUMNS_EXTENDED


def _is_numpy_string_object_array(values: np.ndarray) -> bool:
    if values.dtype != np.dtype("O") or values.size == 0:
        return False
    sample = values.reshape(-1)[: min(int(values.size), 32)]
    is_string_or_null = np.frompyfunc(lambda value: isinstance(value, str) or value is None, 1, 1)
    return bool(np.all(is_string_or_null(sample)))


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


def _filter_mask(table: TensorTable, filters: GenericFilter | None) -> torch.Tensor:
    row_count = len(table)
    device = next(iter(table.columns.values())).device
    if filters is None:
        return torch.ones(row_count, dtype=torch.bool, device=device)
    return _evaluate_filter(table, filters)


def _evaluate_filter(table: TensorTable, filter_: GenericFilter) -> torch.Tensor:
    if filter_.kind == "and":
        masks = tuple(_evaluate_filter(table, child) for child in filter_.children)
        result = masks[0]
        for mask in masks[1:]:
            result = torch.logical_and(result, mask)
        return result
    if filter_.kind == "or":
        masks = tuple(_evaluate_filter(table, child) for child in filter_.children)
        result = masks[0]
        for mask in masks[1:]:
            result = torch.logical_or(result, mask)
        return result
    if filter_.kind == "not":
        return torch.logical_not(_evaluate_filter(table, filter_.children[0]))
    if filter_.kind == "in":
        return _evaluate_in_filter(table, filter_)
    if filter_.kind == "like":
        return _evaluate_like_filter(table, filter_)
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


def _evaluate_in_filter(table: TensorTable, filter_: GenericFilter) -> torch.Tensor:
    values = table.columns[filter_.column]
    literal_ids = [_literal_scalar_id(table, filter_.column, value) for value in filter_.values]
    return membership_mask(values, literal_ids)


def _evaluate_like_filter(table: TensorTable, filter_: GenericFilter) -> torch.Tensor:
    if not isinstance(filter_.value, str):
        raise UnsupportedPlanError("generic SQL LIKE requires a string literal")
    if filter_.column not in table.dictionaries:
        raise UnsupportedPlanError("generic SQL LIKE is only supported for string columns")
    pattern = filter_.value
    vocabulary = table.dictionaries[filter_.column]
    matching_ids = [
        index for index, item in enumerate(vocabulary) if _matches_like_pattern(item, pattern)
    ]
    if not matching_ids:
        return torch.zeros(table.columns[filter_.column].shape, dtype=torch.bool, device=table.columns[filter_.column].device)
    return membership_mask(table.columns[filter_.column], matching_ids)


def _literal_tensor(table: TensorTable, column: str, value: int | float | str) -> torch.Tensor:
    values = table.columns[column]
    return torch.tensor(_literal_scalar_id(table, column, value), dtype=values.dtype, device=values.device)


def _literal_scalar_id(table: TensorTable, column: str, value: int | float | str) -> int | float:
    if isinstance(value, str):
        vocabulary = table.dictionaries.get(column)
        if vocabulary is None or value not in vocabulary:
            return -1
        return vocabulary.index(value)
    return value


def _execute_ungrouped(table: TensorTable, plan: GenericSQLPlan, mask: torch.Tensor) -> list[dict[str, Any]]:
    if _is_scalar_aggregate(plan.projections):
        return [_aggregate_row(table, plan.projections, mask)]
    rows: list[dict[str, Any]] = []
    indices = mask.nonzero().flatten().cpu().tolist()
    for raw_index in indices:
        rows.append(_projection_row(table, plan.projections, int(raw_index)))
    return rows


def _execute_grouped(table: TensorTable, plan: GenericSQLPlan, mask: torch.Tensor) -> list[dict[str, Any]]:
    selected_indices = mask.nonzero().flatten()
    if selected_indices.numel() == 0:
        return []
    key_columns = [table.columns[column][selected_indices].to(dtype=torch.int64) for column in plan.group_by]
    group_ids, unique_keys = composite_group_ids(key_columns)
    aggregate_columns = _grouped_aggregate_columns(table, plan.projections, selected_indices, group_ids, unique_keys)
    return [
        _tensorized_grouped_row(table, plan.projections, plan.group_by, unique_keys, aggregate_columns, group_index)
        for group_index in range(int(unique_keys.shape[0]))
    ]



def _grouped_aggregate_columns(
    table: TensorTable,
    projections: tuple[GenericProjection, ...],
    selected_indices: torch.Tensor,
    group_ids: torch.Tensor,
    unique_keys: torch.Tensor,
) -> dict[str, torch.Tensor]:
    group_count = int(unique_keys.shape[0])
    aggregates: dict[str, torch.Tensor] = {}
    for projection in projections:
        if projection.kind == "column":
            continue
        aggregates[projection.alias] = _evaluate_group_aggregate_tensor(
            table, projection, selected_indices, group_ids, group_count
        )
    return aggregates


def _evaluate_group_aggregate_tensor(
    table: TensorTable,
    projection: GenericProjection,
    selected_indices: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    if projection.kind == "count_star":
        return grouped_count(group_ids, group_count)
    if projection.kind == "count" and projection.column is not None:
        return grouped_count(group_ids, group_count)
    if projection.kind == "sum" and projection.column is not None:
        values = table.columns[projection.column][selected_indices]
        return grouped_sum(values, group_ids, group_count)
    if projection.kind == "min" and projection.column is not None:
        values = table.columns[projection.column][selected_indices]
        return grouped_min(values, group_ids, group_count)
    if projection.kind == "max" and projection.column is not None:
        values = table.columns[projection.column][selected_indices]
        return grouped_max(values, group_ids, group_count)
    if projection.kind == "avg" and projection.column is not None:
        values = table.columns[projection.column][selected_indices]
        return grouped_mean(values, group_ids, group_count)
    raise UnsupportedPlanError(f"generic SQL grouped projection is not supported: {projection.kind}")


def _tensorized_grouped_row(
    table: TensorTable,
    projections: tuple[GenericProjection, ...],
    group_by: tuple[str, ...],
    unique_keys: torch.Tensor,
    aggregate_columns: dict[str, torch.Tensor],
    group_index: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    group_positions = {column: index for index, column in enumerate(group_by)}
    for projection in projections:
        if projection.kind == "column" and projection.column in group_positions:
            key_index = group_positions[projection.column]
            value = unique_keys[group_index, key_index].cpu().item()
            row[projection.alias] = _decode_encoded_value(table, projection.column, value)
            continue
        if projection.alias in aggregate_columns:
            row[projection.alias] = _normalize_tensor_scalar(aggregate_columns[projection.alias][group_index])
            continue
        raise UnsupportedPlanError(f"generic SQL grouped projection is not supported: {projection.kind}")
    return row


def _projection_row(table: TensorTable, projections: tuple[GenericProjection, ...], index: int) -> dict[str, Any]:
    return {projection.alias: _evaluate_projection(table, projection, index) for projection in projections}


def _aggregate_row(
    table: TensorTable,
    projections: tuple[GenericProjection, ...],
    mask: torch.Tensor,
) -> dict[str, Any]:
    return {projection.alias: _evaluate_aggregate(table, projection, mask) for projection in projections}



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
    if projection.kind == "count" and projection.column is not None:
        return int(mask.sum().cpu().item())
    if projection.kind == "sum" and projection.column is not None:
        return float(table.columns[projection.column][mask].sum().cpu().item())
    if projection.kind == "min" and projection.column is not None:
        return float(table.columns[projection.column][mask].min().cpu().item())
    if projection.kind == "max" and projection.column is not None:
        return float(table.columns[projection.column][mask].max().cpu().item())
    if projection.kind == "avg" and projection.column is not None:
        return float(table.columns[projection.column][mask].mean().cpu().item())
    raise UnsupportedPlanError(f"generic SQL aggregate is not supported: {projection.kind}")


def _decode_cell(table: TensorTable, column: str, index: int) -> Any:
    value = table.columns[column][index].cpu().item()
    if column in table.dictionaries:
        return decode(table, column, int(value))
    if isinstance(value, float):
        return float(value)
    return int(value)



def _decode_encoded_value(table: TensorTable, column: str, value: Any) -> Any:
    if column in table.dictionaries:
        return decode(table, column, int(value))
    if isinstance(value, float):
        return float(value)
    return int(value)


def _normalize_tensor_scalar(value: torch.Tensor) -> Any:
    raw = value.cpu().item()
    if isinstance(raw, float):
        return float(raw)
    return int(raw)


def _is_scalar_aggregate(projections: tuple[GenericProjection, ...]) -> bool:
    return all(projection.kind in {"count_star", "count", "sum", "min", "max", "avg"} for projection in projections)


def _sort_rows(rows: list[dict[str, Any]], order_by: tuple[GenericOrderBy, ...]) -> list[dict[str, Any]]:
    if not order_by:
        return rows
    sorted_rows = rows
    for item in reversed(order_by):
        sorted_rows = sorted(sorted_rows, key=lambda row: row[item.column], reverse=item.descending)
    return sorted_rows


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


def _matches_like_pattern(value: str, pattern: str) -> bool:
    if pattern == "%":
        return True
    if pattern.startswith("%") and pattern.endswith("%"):
        return pattern[1:-1] in value
    if pattern.startswith("%"):
        return value.endswith(pattern[1:])
    if pattern.endswith("%"):
        return value.startswith(pattern[:-1])
    return value == pattern
