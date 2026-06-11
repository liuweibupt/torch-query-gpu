"""Correctness-first tensor helpers for supported TPC-H query executors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import torch

from tpch_torch.operators import (
    composite_group_ids,
    grouped_count,
    grouped_count_bincount,
    grouped_sum,
    grouped_sum_bincount,
    low_cardinality_group_ids,
)
from tpch_torch.storage import DATE_COLUMNS, STRING_COLUMNS, TensorTable

INT_COLUMNS = frozenset(
    {
        "c_custkey",
        "c_nationkey",
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "n_nationkey",
        "n_regionkey",
        "o_custkey",
        "o_orderkey",
        "o_shippriority",
        "p_partkey",
        "p_size",
        "ps_availqty",
        "ps_partkey",
        "ps_suppkey",
        "r_regionkey",
        "s_nationkey",
        "s_suppkey",
    }
)

STRING_COLUMNS_EXTENDED = STRING_COLUMNS | frozenset(
    {
        "c_address",
        "c_comment",
        "c_mktsegment",
        "c_name",
        "c_phone",
        "l_comment",
        "l_shipinstruct",
        "l_shipmode",
        "n_comment",
        "n_name",
        "o_clerk",
        "o_comment",
        "o_orderpriority",
        "o_orderstatus",
        "p_brand",
        "p_comment",
        "p_container",
        "p_mfgr",
        "p_name",
        "p_type",
        "r_comment",
        "r_name",
        "s_address",
        "s_comment",
        "s_name",
        "s_phone",
    }
)

DATE_COLUMNS_EXTENDED = DATE_COLUMNS | frozenset({"o_orderdate"})




@dataclass(frozen=True)
class LookupIndex:
    """Sorted dimension-key index for repeated lookup probes."""

    sorted_keys: torch.Tensor
    sorted_values: torch.Tensor

    def __post_init__(self) -> None:
        if self.sorted_keys.ndim != 1 or self.sorted_values.ndim != 1:
            raise ValueError("lookup index tensors must be 1-D")
        if self.sorted_keys.shape != self.sorted_values.shape:
            raise ValueError("lookup index keys and values must have the same shape")
        if self.sorted_keys.device != self.sorted_values.device:
            raise ValueError("lookup index keys and values must be on the same device")
        if self.sorted_keys.numel() <= 1:
            return
        if bool(torch.any(self.sorted_keys[1:] < self.sorted_keys[:-1]).cpu().item()):
            raise ValueError("lookup index keys must be sorted")


@dataclass(frozen=True)
class QueryResult:
    """Rows produced by a supported PyTorch query executor."""

    query_id: int | None
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SQLValidationResult:
    """Summary of a DuckDB-vs-PyTorch SQL comparison."""

    query_id: int | None
    row_count: int
    max_abs_error: float
    duckdb_rows: list[dict[str, Any]]
    pytorch_rows: list[dict[str, Any]]


def fetch_tensor_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: Sequence[str],
    device: str | torch.device = "cpu",
) -> TensorTable:
    """Fetch selected DuckDB columns into a typed TensorTable."""

    select_list = ", ".join(_select_expression(column) for column in columns)
    columnar = con.execute(f"select {select_list} from {table_name}").fetchnumpy()
    return table_from_columnar_typed(columnar, device=device)


def table_from_columnar_typed(
    columnar: Mapping[str, Iterable[Any]], device: str | torch.device = "cpu"
) -> TensorTable:
    columns: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}
    for column_name, values_iterable in columnar.items():
        tensor, vocabulary = encode_column(column_name, values_iterable, device)
        columns[column_name] = tensor
        if vocabulary is not None:
            dictionaries[column_name] = vocabulary
    return TensorTable(columns=columns, dictionaries=dictionaries)


def encode_column(
    column_name: str, values: Iterable[Any], device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if isinstance(values, np.ndarray):
        return _encode_numpy_typed_column(column_name, values, device)
    return _encode_typed_sequence(column_name, list(values), device)


def _encode_numpy_typed_column(
    column_name: str, values: np.ndarray, device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if column_name in STRING_COLUMNS_EXTENDED:
        vocabulary, inverse = np.unique(values.astype(str), return_inverse=True)
        tensor = torch.as_tensor(inverse, dtype=torch.int64, device=device)
        return tensor, tuple(str(value) for value in vocabulary.tolist())
    if values.dtype == np.dtype("O"):
        return _encode_typed_sequence(column_name, values.tolist(), device)
    if column_name in DATE_COLUMNS_EXTENDED:
        tensor = torch.as_tensor(values, dtype=torch.int32, device=device)
        return tensor, None
    if column_name in INT_COLUMNS:
        tensor = torch.as_tensor(values, dtype=torch.int64, device=device)
        return tensor, None
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    return tensor, None


def _encode_typed_sequence(
    column_name: str, values: list[Any], device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if column_name in STRING_COLUMNS_EXTENDED:
        vocabulary = tuple(sorted({str(value) for value in values}))
        ids = {value: index for index, value in enumerate(vocabulary)}
        encoded = [ids[str(value)] for value in values]
        return torch.tensor(encoded, dtype=torch.int64, device=device), vocabulary
    if column_name in DATE_COLUMNS_EXTENDED:
        encoded = [_date_to_yyyymmdd(value) for value in values]
        return torch.tensor(encoded, dtype=torch.int32, device=device), None
    if column_name in INT_COLUMNS:
        return torch.tensor([int(value) for value in values], dtype=torch.int64, device=device), None
    normalized = [float(value) if isinstance(value, Decimal) else value for value in values]
    return torch.tensor(normalized, dtype=torch.float64, device=device), None


def build_lookup_index(dimension_keys: torch.Tensor, dimension_values: torch.Tensor) -> LookupIndex:
    """Build a reusable sorted-key index for dimension lookups."""

    if dimension_keys.ndim != 1 or dimension_values.ndim != 1:
        raise ValueError("lookup inputs must be 1-D")
    if dimension_keys.shape != dimension_values.shape:
        raise ValueError("lookup keys and values must have the same shape")
    if dimension_keys.device != dimension_values.device:
        raise ValueError("lookup keys and values must be on the same device")
    sorted_keys, order = torch.sort(dimension_keys.to(dtype=torch.int64))
    return LookupIndex(sorted_keys=sorted_keys, sorted_values=dimension_values[order])


def lookup_values_from_index(
    index: LookupIndex, fact_keys: torch.Tensor, missing_value: int | float = -1
) -> torch.Tensor:
    """Map fact keys through a pre-sorted lookup index."""

    if fact_keys.device != index.sorted_keys.device:
        raise ValueError("fact keys and lookup index must be on the same device")
    positions = torch.searchsorted(index.sorted_keys, fact_keys.to(dtype=torch.int64))
    in_bounds = positions < index.sorted_keys.numel()
    safe_positions = torch.clamp(positions, max=max(int(index.sorted_keys.numel()) - 1, 0))
    matched = in_bounds & (index.sorted_keys[safe_positions] == fact_keys.to(dtype=torch.int64))
    result = torch.full(fact_keys.shape, missing_value, dtype=index.sorted_values.dtype, device=fact_keys.device)
    result[matched] = index.sorted_values[safe_positions[matched]]
    return result


def lookup_values(
    dimension_keys: torch.Tensor,
    dimension_values: torch.Tensor,
    fact_keys: torch.Tensor,
    missing_value: int | float = -1,
) -> torch.Tensor:
    """Map fact keys to dimension values with an explicit missing sentinel."""

    return lookup_values_from_index(
        build_lookup_index(dimension_keys, dimension_values),
        fact_keys,
        missing_value=missing_value,
    )


def lookup_row_indices(
    dimension_keys: torch.Tensor, fact_keys: torch.Tensor, missing_value: int = -1
) -> torch.Tensor:
    row_ids = torch.arange(dimension_keys.numel(), dtype=torch.int64, device=dimension_keys.device)
    return lookup_values(dimension_keys, row_ids, fact_keys, missing_value=missing_value)


def composite_key(first: torch.Tensor, second: torch.Tensor, multiplier: int) -> torch.Tensor:
    return first.to(dtype=torch.int64) * multiplier + second.to(dtype=torch.int64)


def string_eq(table: TensorTable, column: str, value: str) -> torch.Tensor:
    vocabulary = table.dictionaries[column]
    matching_ids = [index for index, item in enumerate(vocabulary) if item == value]
    return _isin_ids(table.columns[column], matching_ids)


def string_ne(table: TensorTable, column: str, value: str) -> torch.Tensor:
    return ~string_eq(table, column, value)


def string_in(table: TensorTable, column: str, values: Iterable[str]) -> torch.Tensor:
    accepted = set(values)
    vocabulary = table.dictionaries[column]
    matching_ids = [index for index, item in enumerate(vocabulary) if item in accepted]
    return _isin_ids(table.columns[column], matching_ids)


def string_startswith(table: TensorTable, column: str, prefix: str) -> torch.Tensor:
    vocabulary = table.dictionaries[column]
    matching_ids = [index for index, item in enumerate(vocabulary) if item.startswith(prefix)]
    return _isin_ids(table.columns[column], matching_ids)


def string_contains(table: TensorTable, column: str, needle: str) -> torch.Tensor:
    vocabulary = table.dictionaries[column]
    matching_ids = [index for index, item in enumerate(vocabulary) if needle in item]
    return _isin_ids(table.columns[column], matching_ids)


def string_not_like_special_requests(table: TensorTable, column: str) -> torch.Tensor:
    vocabulary = table.dictionaries[column]
    matching_ids = [
        index
        for index, item in enumerate(vocabulary)
        if not ("special" in item and "requests" in item and item.index("special") < item.rindex("requests"))
    ]
    return _isin_ids(table.columns[column], matching_ids)


def decode(table: TensorTable, column: str, encoded: int) -> str:
    return table.dictionaries[column][int(encoded)]


def yyyymmdd_to_year(values: torch.Tensor) -> torch.Tensor:
    return values.to(dtype=torch.int64) // 10_000


def yyyymmdd_to_iso(value: int) -> str:
    raw = int(value)
    year = raw // 10_000
    month = (raw // 100) % 100
    day = raw % 100
    return f"{year:04d}-{month:02d}-{day:02d}"


def aggregate_sum_by_keys(
    key_columns: Sequence[torch.Tensor], value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids, unique_keys = composite_group_ids([key.to(dtype=torch.int64) for key in key_columns])
    return unique_keys, grouped_sum(value, group_ids, int(unique_keys.shape[0]))


def aggregate_count_by_keys(key_columns: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids, unique_keys = composite_group_ids([key.to(dtype=torch.int64) for key in key_columns])
    return unique_keys, grouped_count(group_ids, int(unique_keys.shape[0]))



def aggregate_sum_by_low_cardinality_keys(
    key_columns: Sequence[torch.Tensor], cardinalities: Sequence[int], value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids, group_count = low_cardinality_group_ids(key_columns, cardinalities)
    return group_ids, grouped_sum_bincount(value, group_ids, group_count)


def aggregate_count_by_low_cardinality_keys(
    key_columns: Sequence[torch.Tensor], cardinalities: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids, group_count = low_cardinality_group_ids(key_columns, cardinalities)
    return group_ids, grouped_count_bincount(group_ids, group_count)


def run_duckdb_sql(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    result = con.execute(sql)
    column_names = [description[0] for description in result.description]
    return [normalize_row(column_names, row) for row in result.fetchall()]


def compare_rows(
    duckdb_rows: Sequence[dict[str, Any]], pytorch_rows: Sequence[dict[str, Any]]
) -> float:
    if len(duckdb_rows) != len(pytorch_rows):
        raise AssertionError(f"row count mismatch: DuckDB={len(duckdb_rows)} PyTorch={len(pytorch_rows)}")
    max_abs_error = 0.0
    for row_index, (duckdb_row, pytorch_row) in enumerate(zip(duckdb_rows, pytorch_rows)):
        if duckdb_row.keys() != pytorch_row.keys():
            raise AssertionError(f"schema mismatch at row {row_index}: {duckdb_row.keys()} != {pytorch_row.keys()}")
        for column_name, duckdb_value in duckdb_row.items():
            pytorch_value = pytorch_row[column_name]
            if isinstance(duckdb_value, float) or isinstance(pytorch_value, float):
                max_abs_error = max(max_abs_error, abs(float(duckdb_value) - float(pytorch_value)))
            elif duckdb_value != pytorch_value:
                raise AssertionError(
                    f"value mismatch at row {row_index} column {column_name}: "
                    f"DuckDB={duckdb_value!r} PyTorch={pytorch_value!r}"
                )
    return max_abs_error


def normalize_row(column_names: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    return {column_name: normalize_value(value) for column_name, value in zip(column_names, row)}


def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value) if isinstance(value, str) else value


def _select_expression(column: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    return column


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


def _isin_ids(values: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    accepted = torch.tensor(tuple(ids), dtype=values.dtype, device=values.device)
    return torch.isin(values, accepted)
