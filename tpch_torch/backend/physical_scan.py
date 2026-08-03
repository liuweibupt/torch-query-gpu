"""Physical scan fetch helpers for DuckDB-backed tensor tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterator

import duckdb
import numpy as np
import torch

from tpch_torch.backend.generic import _encode_generic_column
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.backend.physical_scan_pushdown import where_clause
from tpch_torch.backend.physical_scan_sql import scan_select_list
from tpch_torch.backend.static_dictionaries import static_string_dictionary
from tpch_torch.backend.type_mapping import (
    column_meta_from_duckdb_type,
    column_type_from_duckdb_type,
    encode_decimal_array,
)
from tpch_torch.errors import UnsupportedPlanError
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
    scan_filters: tuple[str, ...] = (),
) -> PhysicalTable:
    """Fetch a possibly ranged table slice into physical tensor columns."""

    column_types = _table_column_type_map(con, table_name)
    select_list = scan_select_list(table_name, fetched_columns, column_types)
    offset, limit = _scan_offset_limit(scan_range)
    where_sql = where_clause(scan_filters)
    columnar = con.execute(f"select {select_list} from {table_name}{where_sql}{limit}").fetchnumpy()
    return _physical_table_from_columnar(
        columnar,
        column_types,
        table_name,
        fetched_columns,
        order_columns,
        device,
        source_offset=offset,
        chunk_size=chunk_size,
        chunk_index=chunk_index,
    )


def fetch_physical_table_stream(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
    *,
    chunk_size: int,
    scan_filters: tuple[str, ...] = (),
) -> Iterator[PhysicalTable]:
    """Yield table chunks from one DuckDB Arrow RecordBatch stream.

    This is the scan source used by the batch/partitionable pipeline.  It
    executes one DuckDB query and drains the result through Arrow batches,
    avoiding repeated ``LIMIT/OFFSET`` scans for large tables.
    """

    if chunk_size <= 0:
        raise ValueError("scan chunk_size must be positive")
    if not fetched_columns and not scan_filters:
        total, _ = scan_row_count(con, table_name, None)
        for chunk_index, start in enumerate(range(0, total, chunk_size)):
            yield _rowid_only_physical_table(
                table_name,
                start,
                min(start + chunk_size, total),
                chunk_size,
                chunk_index,
                device,
            )
        return
    if not fetched_columns:
        yield from _filtered_rowid_stream(con, table_name, device, chunk_size, scan_filters)
        return
    column_types = _table_column_type_map(con, table_name)
    select_list = scan_select_list(table_name, fetched_columns, column_types)
    where_sql = where_clause(scan_filters)
    reader = con.execute(f"select {select_list} from {table_name}{where_sql}").fetch_record_batch(
        rows_per_batch=chunk_size
    )
    source_offset = 0
    for chunk_index, batch in enumerate(_record_batches(reader)):
        row_count = int(batch.num_rows)
        if row_count == 0:
            continue
        yield _physical_table_from_columnar(
            _record_batch_to_numpy(batch, fetched_columns),
            column_types,
            table_name,
            fetched_columns,
            order_columns,
            device,
            source_offset=source_offset,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )
        source_offset += row_count


def _physical_table_from_columnar(
    columnar: Mapping[str, Any],
    column_types: Mapping[str, str],
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
    *,
    source_offset: int,
    chunk_size: int | None,
    chunk_index: int | None,
) -> PhysicalTable:
    values: dict[str, PhysicalValue] = {}
    batch_types: dict[str, ColumnType] = {}
    for column in fetched_columns:
        duckdb_type = column_types.get(column, "")
        meta = column_meta_from_duckdb_type(duckdb_type, nullable=True)
        tensor, vocabulary, meta = _encode_physical_column(
            columnar[column],
            device,
            column_name=column,
            table_name=table_name,
            meta=meta,
        )
        batch_types[column] = _column_type_for_encoded_column(column, duckdb_type, meta)
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
    _add_rowid_aliases(values, table_name, row_count, device, offset=source_offset)
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
        source_offset=source_offset,
        chunk_size=chunk_size or row_count,
        chunk_index=_scan_chunk_index(source_offset, chunk_size or row_count, chunk_index),
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


def _filtered_rowid_stream(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    device: str,
    chunk_size: int,
    scan_filters: tuple[str, ...],
) -> Iterator[PhysicalTable]:
    reader = con.execute(
        f"select {_DUCKDB_ROW_ID} from {table_name}{where_clause(scan_filters)}"
    ).fetch_record_batch(rows_per_batch=chunk_size)
    emitted_offset = 0
    for chunk_index, batch in enumerate(_record_batches(reader)):
        rowids = torch.as_tensor(
            batch.column(0).to_numpy(zero_copy_only=False),
            dtype=torch.int64,
            device=device,
        )
        yield _rowid_table_from_tensor(
            table_name,
            rowids,
            chunk_size,
            chunk_index,
            emitted_offset,
        )
        emitted_offset += int(rowids.numel())


def _rowid_only_physical_table(
    table_name: str,
    start: int,
    end: int,
    chunk_size: int,
    chunk_index: int,
    device: str,
) -> PhysicalTable:
    rowids = torch.arange(start, end, dtype=torch.int64, device=device)
    return _rowid_table_from_tensor(table_name, rowids, chunk_size, chunk_index, start)


def _rowid_table_from_tensor(
    table_name: str,
    rowid_tensor: torch.Tensor,
    chunk_size: int,
    chunk_index: int,
    source_offset: int,
) -> PhysicalTable:
    rowids = PhysicalValue(rowid_tensor)
    values = {_ROW_ID: rowids, _DUCKDB_ROW_ID: rowids}
    batch = _scan_batch(
        values,
        {_ROW_ID: ColumnType.int64(_ROW_ID), _DUCKDB_ROW_ID: ColumnType.int64(_DUCKDB_ROW_ID)},
        (_ROW_ID,),
        int(rowid_tensor.numel()),
        str(rowid_tensor.device),
        source_offset=source_offset,
        chunk_size=chunk_size,
        chunk_index=chunk_index,
    )
    return PhysicalTable(table_name, {_ROW_ID: rowids}, (_ROW_ID,), int(rowid_tensor.numel()), batch)




def _record_batches(reader: Any) -> Iterator[Any]:
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def _record_batch_to_numpy(batch: Any, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    names = tuple(str(name) for name in batch.schema.names)
    missing = [column for column in columns if column not in names]
    if missing:
        raise UnsupportedPlanError(f"Arrow scan batch is missing column(s): {', '.join(missing)}")
    indexes = {name: index for index, name in enumerate(names)}
    return {
        column: batch.column(indexes[column]).to_numpy(zero_copy_only=False)
        for column in columns
    }


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
    batch_device = _storage_batch_device(storages) or torch.device(device)
    return TensorRecordBatch.from_storages(
        columns=storages,
        types=types,
        batch_meta=BatchMeta(
            row_count=row_count,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            source_offset=source_offset,
            device=batch_device,
        ),
    )


def _storage_from_value(value: PhysicalValue) -> ColumnStorage:
    tensor = value.require_tensor()
    if value.dictionary is not None:
        return ColumnStorage.dictionary_ids(tensor, value.dictionary, validity=value.valid)
    if value.meta is not None and value.meta.logical_dtype == LogicalDType.DECIMAL:
        return ColumnStorage.decimal64(tensor, validity=value.valid)
    return ColumnStorage.fixed(tensor, validity=value.valid)


def _storage_batch_device(storages: Mapping[str, ColumnStorage]) -> torch.device | None:
    first = next(iter(storages.values()), None)
    return None if first is None else first.device


def _column_type_for_encoded_column(
    column: str,
    duckdb_type: str,
    meta: ColumnMeta,
) -> ColumnType:
    if meta.logical_dtype == LogicalDType.UNKNOWN:
        return column_type_from_duckdb_type(column, duckdb_type, nullable=meta.nullable)
    return ColumnType.from_column_meta(column, meta)


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
        if _is_integral_array(values):
            return torch.as_tensor(values, dtype=torch.int64, device=device), None, meta
        return encode_decimal_array(values, meta, device), None, meta
    static_dictionary = static_string_dictionary(table_name, column_name)
    if static_dictionary is not None and _is_integral_array(values):
        tensor = torch.as_tensor(values, dtype=torch.int64, device=device)
        _validate_static_dictionary_ids(tensor, static_dictionary, column_name)
        return tensor, static_dictionary, ColumnMeta.string_dict(static_dictionary, nullable=meta.nullable)
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


def _is_integral_array(values: Any) -> bool:
    return isinstance(values, np.ndarray) and values.dtype.kind in {"i", "u"}


def _validate_static_dictionary_ids(
    tensor: torch.Tensor,
    dictionary: tuple[str, ...],
    column_name: str,
) -> None:
    if tensor.numel() == 0:
        return
    min_value = int(tensor.min().cpu().item())
    max_value = int(tensor.max().cpu().item())
    if min_value < 0 or max_value >= len(dictionary):
        raise UnsupportedPlanError(f"static dictionary encoding produced invalid ids for {column_name}")


def _table_column_type_map(con: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    rows = con.execute(f"pragma table_info('{table_name}')").fetchall()
    return {str(row[1]): str(row[2]) for row in rows}
