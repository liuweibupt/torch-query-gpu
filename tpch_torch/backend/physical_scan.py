"""Physical scan fetch helpers for DuckDB-backed tensor tables."""

from __future__ import annotations

from typing import Iterator

import duckdb
import torch

from tpch_torch.backend.generic import _encode_generic_column
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.backend.type_mapping import (
    column_meta_from_duckdb_type,
    column_type_from_duckdb_type,
    encode_decimal_array,
)
from tpch_torch.relational import DATE_COLUMNS_EXTENDED
from tpch_torch.record_batch import (
    BatchMeta,
    ColumnMeta,
    ColumnStorage,
    ColumnType,
    LogicalDType,
    TensorRecordBatch,
)

_ROW_ID = "__rowid__"
_DUCKDB_ROW_ID = "rowid"
_TPCH_SORTED_UNIQUE_COLUMNS = {
    ("customer", "c_custkey"),
    ("nation", "n_nationkey"),
    ("orders", "o_orderkey"),
    ("part", "p_partkey"),
    ("region", "r_regionkey"),
    ("supplier", "s_suppkey"),
}


def fetch_physical_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
    *,
    scan_range: tuple[int, int] | None = None,
    chunk_size: int | None = None,
    chunk_index: int | None = None,
) -> PhysicalTable:
    """Fetch a possibly ranged table slice into physical tensor columns."""

    select_list = ", ".join(_select_expression(column) for column in fetched_columns)
    offset, limit = _scan_offset_limit(scan_range)
    columnar = con.execute(f"select {select_list} from {table_name}{limit}").fetchnumpy()
    column_types = _table_column_type_map(con, table_name)
    values: dict[str, PhysicalValue] = {}
    batch_types: dict[str, ColumnType] = {}
    for column in fetched_columns:
        duckdb_type = column_types.get(column, "")
        meta = column_meta_from_duckdb_type(duckdb_type, nullable=True)
        batch_types[column] = column_type_from_duckdb_type(column, duckdb_type, nullable=True)
        tensor, vocabulary, meta = _encode_physical_column(
            columnar[column],
            device,
            column_name=column,
            table_name=table_name,
            meta=meta,
        )
        value = PhysicalValue(
            tensor=tensor,
            dictionary=vocabulary,
            is_date=column in DATE_COLUMNS_EXTENDED,
            meta=meta,
        )
        value = _with_scan_metadata(table_name, column, value)
        values[column] = value
        values[f"{table_name}.{column}"] = value
    row_count = 0 if not fetched_columns else int(next(iter(values.values())).require_tensor().numel())
    _add_rowid_aliases(values, table_name, row_count, device, offset=offset)
    order = order_columns or (_ROW_ID,)
    if not order_columns:
        values[_ROW_ID] = values[_DUCKDB_ROW_ID]
    canonical_names = _scan_canonical_names(fetched_columns, values)
    batch = _scan_batch(
        values,
        batch_types,
        canonical_names,
        row_count,
        device,
        source_offset=offset,
        chunk_size=chunk_size or row_count,
        chunk_index=_scan_chunk_index(offset, chunk_size or row_count, chunk_index),
    )
    aliases = {f"{table_name}.{column}": column for column in fetched_columns if column in values}
    if _DUCKDB_ROW_ID in values and _DUCKDB_ROW_ID in order:
        aliases[f"{table_name}.{_DUCKDB_ROW_ID}"] = _DUCKDB_ROW_ID
    return PhysicalTable(
        table_name,
        {name: values[name] for name in canonical_names},
        order,
        row_count,
        batch,
        aliases,
    )


def _scan_canonical_names(
    fetched_columns: tuple[str, ...],
    values: dict[str, PhysicalValue],
) -> tuple[str, ...]:
    names = list(fetched_columns)
    if _DUCKDB_ROW_ID in values:
        names.append(_DUCKDB_ROW_ID)
    if _ROW_ID in values and _ROW_ID not in names:
        names.append(_ROW_ID)
    return tuple(dict.fromkeys(name for name in names if name in values))


def fetch_physical_table_chunks(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
    *,
    chunk_size: int,
) -> Iterator[PhysicalTable]:
    """Yield physical scan chunks backed by TensorRecordBatch metadata."""

    if chunk_size <= 0:
        raise ValueError("scan chunk_size must be positive")
    total, _ = scan_row_count(con, table_name, None)
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        yield fetch_physical_table(
            con,
            table_name,
            fetched_columns,
            order_columns,
            device,
            scan_range=(start, end),
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )


def _scan_batch(
    values: dict[str, PhysicalValue],
    column_types: dict[str, ColumnType],
    order: tuple[str, ...],
    row_count: int,
    device: str,
    source_offset: int,
    chunk_size: int,
    chunk_index: int,
) -> TensorRecordBatch:
    storages: dict[str, ColumnStorage] = {}
    types: dict[str, ColumnType] = {}
    for name in order:
        value = values[name]
        storages[name] = _storage_from_value(value)
        types[name] = column_types.get(name, ColumnType.int64(name))
    return TensorRecordBatch.from_storages(
        columns=storages,
        types=types,
        batch_meta=BatchMeta(
            row_count=row_count,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            source_offset=source_offset,
            device=torch.device(device),
        ),
    )


def _storage_from_value(value: PhysicalValue) -> ColumnStorage:
    tensor = value.require_tensor()
    if value.dictionary is not None:
        return ColumnStorage.dictionary_ids(tensor, value.dictionary, validity=value.valid)
    if value.meta is not None and value.meta.logical_dtype == LogicalDType.DECIMAL:
        return ColumnStorage.decimal64(tensor, validity=value.valid)
    return ColumnStorage.fixed(tensor, validity=value.valid)


def _scan_chunk_index(offset: int, chunk_size: int, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    if offset <= 0 or chunk_size <= 0:
        return 0
    return offset // chunk_size


def scan_row_count(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    scan_range: tuple[int, int] | None,
) -> tuple[int, int]:
    """Return row count and rowid offset for an optional scan range."""

    if scan_range is None:
        count = int(con.execute(f"select count(*) from {table_name}").fetchone()[0])
        return count, 0
    start, end = scan_range
    return max(end - start, 0), start


def _scan_offset_limit(scan_range: tuple[int, int] | None) -> tuple[int, str]:
    if scan_range is None:
        return 0, ""
    start, end = scan_range
    return start, f" limit {max(end - start, 0)} offset {start}"


def _add_rowid_aliases(
    values: dict[str, PhysicalValue],
    table_name: str,
    row_count: int,
    device: str,
    *,
    offset: int = 0,
) -> None:
    rowids = PhysicalValue(
        torch.arange(offset, offset + row_count, dtype=torch.int64, device=device),
        sorted_non_decreasing=True,
        unique=True,
    )
    values[_DUCKDB_ROW_ID] = rowids
    values[f"{table_name}.{_DUCKDB_ROW_ID}"] = rowids


def _select_expression(column: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    return column


def _with_scan_metadata(table_name: str, column: str, value: PhysicalValue) -> PhysicalValue:
    if (table_name.lower(), column.lower()) not in _TPCH_SORTED_UNIQUE_COLUMNS:
        return value
    tensor = value.require_tensor()
    if not _is_strictly_increasing(tensor):
        return value
    return value.with_metadata(sorted_non_decreasing=True, unique=True)


def _is_strictly_increasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return bool(torch.all(values[1:] > values[:-1]).cpu().item())


def _encode_physical_column(
    values,
    device: str,
    *,
    column_name: str,
    table_name: str,
    meta: ColumnMeta,
) -> tuple[torch.Tensor, tuple[str, ...] | None, ColumnMeta]:
    if meta.logical_dtype == LogicalDType.DECIMAL:
        return encode_decimal_array(values, meta, device), None, meta
    tensor, vocabulary = _encode_generic_column(
        values,
        device,
        column_name=column_name,
        table_name=table_name,
    )
    if vocabulary is not None:
        meta = ColumnMeta.string_dict(vocabulary, nullable=meta.nullable)
    elif meta.logical_dtype == LogicalDType.FP32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor, vocabulary, meta


def _table_column_type_map(con: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    rows = con.execute(f"pragma table_info('{table_name}')").fetchall()
    return {str(row[1]): str(row[2]) for row in rows}
