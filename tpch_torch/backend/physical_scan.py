"""Physical scan fetch helpers for DuckDB-backed tensor tables."""

from __future__ import annotations

import duckdb
import torch

from tpch_torch.backend.generic import _encode_generic_column
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.relational import DATE_COLUMNS_EXTENDED

_ROW_ID = "__rowid__"
_DUCKDB_ROW_ID = "rowid"


def fetch_physical_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
    *,
    scan_range: tuple[int, int] | None = None,
) -> PhysicalTable:
    """Fetch a possibly ranged table slice into physical tensor columns."""

    select_list = ", ".join(_select_expression(column) for column in fetched_columns)
    offset, limit = _scan_offset_limit(scan_range)
    columnar = con.execute(f"select {select_list} from {table_name}{limit}").fetchnumpy()
    values: dict[str, PhysicalValue] = {}
    for column in fetched_columns:
        tensor, vocabulary = _encode_generic_column(
            columnar[column],
            device,
            column_name=column,
            table_name=table_name,
        )
        value = PhysicalValue(tensor=tensor, dictionary=vocabulary, is_date=column in DATE_COLUMNS_EXTENDED)
        values[column] = value
        values[f"{table_name}.{column}"] = value
    row_count = 0 if not fetched_columns else int(next(iter(values.values())).require_tensor().numel())
    _add_rowid_aliases(values, table_name, row_count, device, offset=offset)
    order = order_columns or (_ROW_ID,)
    if not order_columns:
        values[_ROW_ID] = values[_DUCKDB_ROW_ID]
    return PhysicalTable(table_name, values, order, row_count)


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
    rowids = PhysicalValue(torch.arange(offset, offset + row_count, dtype=torch.int64, device=device))
    values[_DUCKDB_ROW_ID] = rowids
    values[f"{table_name}.{_DUCKDB_ROW_ID}"] = rowids


def _select_expression(column: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    return column
