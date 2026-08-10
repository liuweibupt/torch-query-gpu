"""Explicit universal compatibility execution through TensorRecordBatch chunks.

This mode is intentionally separate from the strict TQP physical interpreter.  It
uses DuckDB to execute SQL shapes whose physical operators are not implemented
by the PyTorch backend yet, then converts DuckDB Arrow result batches into the
same TensorRecordBatch/TensorTable substrate used by TQP operators.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator, Sequence

import duckdb
import numpy as np
import torch

from tpch_torch.backend.physical_types import PhysicalTable
from tpch_torch.backend.type_mapping import column_type_from_duckdb_type, encode_decimal_array
from tpch_torch.duckdb_plan_json import describe_output_schema
from tpch_torch.record_batch import BatchMeta, TensorRecordBatch
from tpch_torch.record_batch_storage import ColumnStorage
from tpch_torch.record_batch_types import ColumnType, LogicalDType

DEFAULT_UNIVERSAL_CHUNK_SIZE = 65_536


def execute_universal_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    device: str,
    chunk_size: int = DEFAULT_UNIVERSAL_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Execute any DuckDB-plannable SQL as explicit compatibility mode.

    The query itself is evaluated by DuckDB.  Result chunks are materialized as
    TensorRecordBatch instances and decoded from TensorTable, so downstream code
    sees the same columnar tensor ABI rather than DuckDB row tuples.
    """

    rows: list[dict[str, Any]] = []
    for batch in iter_universal_record_batches(con, sql, device=device, chunk_size=chunk_size):
        rows.extend(_rows_from_batch(batch))
    return rows


def iter_universal_record_batches(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    device: str,
    chunk_size: int = DEFAULT_UNIVERSAL_CHUNK_SIZE,
) -> Iterator[TensorRecordBatch]:
    """Yield DuckDB result chunks encoded as TensorRecordBatch objects."""

    if chunk_size <= 0:
        raise ValueError("universal chunk_size must be positive")
    output_schema = describe_output_schema(con, sql)
    column_types = tuple(_column_type(column.name, column.type_name, column.nullable) for column in output_schema)
    reader = con.execute(sql).fetch_record_batch(rows_per_batch=chunk_size)
    source_offset = 0
    for chunk_index, arrow_batch in enumerate(_record_batches(reader)):
        row_count = int(arrow_batch.num_rows)
        if row_count == 0:
            continue
        yield _record_batch_from_arrow(
            arrow_batch,
            column_types,
            device=device,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            source_offset=source_offset,
        )
        source_offset += row_count


def _record_batch_from_arrow(
    arrow_batch: Any,
    column_types: Sequence[ColumnType],
    *,
    device: str,
    chunk_size: int,
    chunk_index: int,
    source_offset: int,
) -> TensorRecordBatch:
    storages = {}
    types = {}
    names = _arrow_names(arrow_batch, column_types)
    for index, column_type in enumerate(column_types):
        name = names[index]
        storage, logical_type = _storage_from_arrow_column(arrow_batch.column(index), column_type, device)
        storages[name] = storage
        types[name] = logical_type
    batch_device = _storage_batch_device(storages) or torch.device(device)
    return TensorRecordBatch.from_storages(
        columns=storages,
        types=types,
        batch_meta=BatchMeta(
            row_count=int(arrow_batch.num_rows),
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            source_offset=source_offset,
            device=batch_device,
        ),
    )


def _storage_from_arrow_column(column: Any, column_type: ColumnType, device: str) -> tuple[ColumnStorage, ColumnType]:
    values = column.to_numpy(zero_copy_only=False)
    validity = _validity_tensor(column, device)
    logical = column_type.logical_dtype
    if logical == LogicalDType.DECIMAL:
        storage = ColumnStorage.decimal64(_decimal_tensor(values, column_type, validity, device), validity=validity)
        return storage, column_type
    if logical == LogicalDType.DATE:
        return ColumnStorage.fixed(_date_tensor(values, validity, device), validity=validity), column_type
    if logical == LogicalDType.BOOL:
        return ColumnStorage.fixed(_fixed_tensor(values, validity, torch.bool, False, device), validity=validity), column_type
    if logical == LogicalDType.FP32:
        return ColumnStorage.fixed(_fixed_tensor(values, validity, torch.float32, 0.0, device), validity=validity), column_type
    if logical == LogicalDType.FP64:
        return ColumnStorage.fixed(_fixed_tensor(values, validity, torch.float64, 0.0, device), validity=validity), column_type
    if logical == LogicalDType.INT64:
        return ColumnStorage.fixed(_fixed_tensor(values, validity, torch.int64, 0, device), validity=validity), column_type
    return _dictionary_storage(values, validity, column_type.name, device)


def _record_batches(reader: Any) -> Iterator[Any]:
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def _column_type(name: str, duckdb_type: str, nullable: bool | None) -> ColumnType:
    return column_type_from_duckdb_type(name, duckdb_type, nullable=bool(nullable))


def _arrow_names(arrow_batch: Any, column_types: Sequence[ColumnType]) -> tuple[str, ...]:
    arrow_names = tuple(str(name) for name in arrow_batch.schema.names)
    if len(arrow_names) == len(column_types) and len(set(arrow_names)) == len(arrow_names):
        return arrow_names
    return tuple(_unique_name(column_type.name, index, arrow_names[:index]) for index, column_type in enumerate(column_types))


def _unique_name(name: str, index: int, previous: Sequence[str]) -> str:
    if name not in previous:
        return name
    return f"{name}__{index}"


def _validity_tensor(column: Any, device: str) -> torch.Tensor | None:
    if int(column.null_count) == 0:
        return None
    validity = column.is_valid().to_numpy(zero_copy_only=False)
    return torch.as_tensor(validity, dtype=torch.bool, device=device)


def _fixed_tensor(values: Any, validity: torch.Tensor | None, dtype: torch.dtype, fill: Any, device: str) -> torch.Tensor:
    prepared = _fill_invalid(values, validity, fill)
    if np.asarray(prepared).dtype == np.dtype("O"):
        return torch.tensor(_coerced_object_values(prepared, dtype), dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(prepared).copy(), dtype=dtype, device=device)


def _coerced_object_values(values: Any, dtype: torch.dtype) -> list[Any]:
    raw_values = np.asarray(values, dtype=object).tolist()
    if dtype == torch.bool:
        return [bool(value) for value in raw_values]
    if dtype in {torch.int64, torch.int32, torch.int16, torch.int8}:
        return [int(value) for value in raw_values]
    return [float(value) for value in raw_values]


def _decimal_tensor(values: Any, column_type: ColumnType, validity: torch.Tensor | None, device: str) -> torch.Tensor:
    prepared = _fill_invalid(values, validity, Decimal(0))
    return encode_decimal_array(prepared, column_type.to_column_meta(), device)


def _date_tensor(values: Any, validity: torch.Tensor | None, device: str) -> torch.Tensor:
    prepared = _fill_invalid(values, validity, date(1970, 1, 1))
    encoded = [_date_to_yyyymmdd(value) for value in np.asarray(prepared, dtype=object).tolist()]
    return torch.tensor(encoded, dtype=torch.int32, device=device)


def _dictionary_storage(
    values: Any,
    validity: torch.Tensor | None,
    name: str,
    device: str,
) -> tuple[ColumnStorage, ColumnType]:
    prepared = _fill_invalid(values, validity, "")
    strings = tuple(str(value) for value in np.asarray(prepared, dtype=object).tolist())
    vocabulary = tuple(sorted(set(strings))) or ("",)
    ids = {literal: index for index, literal in enumerate(vocabulary)}
    tensor = torch.tensor([ids[value] for value in strings], dtype=torch.int64, device=device)
    return ColumnStorage.dictionary_ids(tensor, vocabulary, validity=validity), ColumnType.string_dict(name, nullable=validity is not None)


def _fill_invalid(values: Any, validity: torch.Tensor | None, fill: Any) -> Any:
    array = np.asarray(values, dtype=object if _needs_object_array(values) else None)
    if validity is None:
        return array
    valid = validity.cpu().numpy().astype(bool, copy=False)
    if array.dtype == np.dtype("O"):
        result = array.copy()
        result[~valid] = fill
        return result
    return np.where(valid, array, fill)


def _needs_object_array(values: Any) -> bool:
    array = np.asarray(values)
    return array.dtype == np.dtype("O") or np.issubdtype(array.dtype, np.datetime64)


def _date_to_yyyymmdd(value: Any) -> int:
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value.astype("datetime64[D]"), unit="D")
        return int(text.replace("-", ""))
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.year * 10_000 + value.month * 100 + value.day
    if isinstance(value, str):
        return int(value[:10].replace("-", ""))
    return int(value)


def _storage_batch_device(storages: dict[str, ColumnStorage]) -> torch.device | None:
    first = next(iter(storages.values()), None)
    return None if first is None else first.device


def _rows_from_batch(batch: TensorRecordBatch) -> list[dict[str, Any]]:
    table = PhysicalTable.from_batch("duckdb_result", batch)
    rows = []
    for row_index in range(table.row_count):
        rows.append({name: table.columns[name].cell(row_index) for name in table.order})
    return rows
