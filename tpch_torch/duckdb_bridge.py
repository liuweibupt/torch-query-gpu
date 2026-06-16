"""DuckDB integration for TPC-H data, Substrait export, and baselines."""

from __future__ import annotations

import json
import os
import weakref
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import torch

from tpch_torch.storage import TensorTable

LINEITEM_COLUMNS = (
    "l_returnflag",
    "l_linestatus",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_shipdate",
)
LINEITEM_SELECT_FOR_TORCH = """
select
    case l_returnflag
        when 'A' then 0
        when 'N' then 1
        when 'R' then 2
        else error('unexpected l_returnflag value: ' || l_returnflag)
    end::bigint as l_returnflag,
    case l_linestatus
        when 'F' then 0
        when 'O' then 1
        else error('unexpected l_linestatus value: ' || l_linestatus)
    end::bigint as l_linestatus,
    l_quantity::double as l_quantity,
    l_extendedprice::double as l_extendedprice,
    l_discount::double as l_discount,
    l_tax::double as l_tax,
    strftime(l_shipdate, '%Y%m%d')::integer as l_shipdate
from lineitem
""".strip()
LINEITEM_DICTIONARIES = {
    "l_returnflag": ("A", "N", "R"),
    "l_linestatus": ("F", "O"),
}
_LINEITEM_TENSOR_TABLE_CACHE: weakref.WeakKeyDictionary[Any, dict[str, TensorTable]] = weakref.WeakKeyDictionary()


class DuckDBSubstraitError(RuntimeError):
    """Raised when DuckDB cannot export a real Substrait plan."""


class DuckDBTPCHError(RuntimeError):
    """Raised when DuckDB cannot generate TPC-H data."""


def connect_database(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB database. `None` creates an in-memory database."""

    if path is None:
        return duckdb.connect()
    return duckdb.connect(str(path))


def create_lineitem_fixture(
    con: duckdb.DuckDBPyConnection, rows: Iterable[Sequence[Any]]
) -> None:
    """Create a minimal `lineitem` table containing only Q1 columns."""

    clear_lineitem_tensor_table_cache(con)
    con.execute("drop table if exists lineitem")
    con.execute(
        """
        create table lineitem(
            l_returnflag varchar,
            l_linestatus varchar,
            l_quantity double,
            l_extendedprice double,
            l_discount double,
            l_tax double,
            l_shipdate date
        )
        """
    )
    con.executemany(
        "insert into lineitem values (?, ?, ?, ?, ?, ?, ?)",
        list(rows),
    )


def generate_tpch(con: duckdb.DuckDBPyConnection, scale_factor: float = 1.0) -> None:
    """Generate TPC-H tables inside DuckDB using the official DuckDB extension."""

    clear_lineitem_tensor_table_cache(con)
    try:
        con.execute("install tpch")
        con.execute("load tpch")
        con.execute("call dbgen(sf = ?)", [scale_factor])
    except duckdb.Error as exc:
        raise DuckDBTPCHError(f"failed to generate TPC-H data: {exc}") from exc


def export_substrait_json(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    """Export a real Substrait JSON plan using DuckDB's Substrait extension."""

    _load_substrait_extension(con)
    escaped_sql = sql.replace("'", "''")
    try:
        raw_plan = con.execute(f"call get_substrait_json('{escaped_sql}')").fetchone()[0]
    except duckdb.Error as exc:
        raise DuckDBSubstraitError(f"get_substrait_json failed: {exc}") from exc
    return json.loads(raw_plan)


def fetch_lineitem_tensor_table(
    con: duckdb.DuckDBPyConnection, device: str = "cpu"
) -> TensorTable:
    """Fetch Q1 lineitem columns into a columnar `TensorTable`."""

    device_key = _device_cache_key(device)
    cached = _cached_lineitem_tensor_table(con, device_key)
    if cached is not None:
        return cached
    columnar = con.execute(LINEITEM_SELECT_FOR_TORCH).fetchnumpy()
    table = _lineitem_table_from_preencoded_columnar(columnar, device=device)
    _cache_lineitem_tensor_table(con, device_key, table)
    return table


def clear_lineitem_tensor_table_cache(con: Any | None = None) -> None:
    """Clear resident lineitem tensor cache for one connection or all connections."""

    if con is None:
        _LINEITEM_TENSOR_TABLE_CACHE.clear()
        return
    try:
        del _LINEITEM_TENSOR_TABLE_CACHE[con]
    except KeyError:
        return


def _cached_lineitem_tensor_table(con: Any, device_key: str) -> TensorTable | None:
    device_tables = _LINEITEM_TENSOR_TABLE_CACHE.get(con)
    if device_tables is None:
        return None
    return device_tables.get(device_key)


def _cache_lineitem_tensor_table(con: Any, device_key: str, table: TensorTable) -> None:
    device_tables = _LINEITEM_TENSOR_TABLE_CACHE.setdefault(con, {})
    device_tables[device_key] = table


def _device_cache_key(device: str | torch.device) -> str:
    return str(torch.device(device))


def _lineitem_table_from_preencoded_columnar(
    columnar: dict[str, Any], device: str | torch.device
) -> TensorTable:
    columns = {
        "l_returnflag": torch.as_tensor(
            columnar["l_returnflag"], dtype=torch.int64, device=device
        ),
        "l_linestatus": torch.as_tensor(
            columnar["l_linestatus"], dtype=torch.int64, device=device
        ),
        "l_quantity": torch.as_tensor(columnar["l_quantity"], dtype=torch.float64, device=device),
        "l_extendedprice": torch.as_tensor(
            columnar["l_extendedprice"], dtype=torch.float64, device=device
        ),
        "l_discount": torch.as_tensor(columnar["l_discount"], dtype=torch.float64, device=device),
        "l_tax": torch.as_tensor(columnar["l_tax"], dtype=torch.float64, device=device),
        "l_shipdate": torch.as_tensor(columnar["l_shipdate"], dtype=torch.int32, device=device),
    }
    return TensorTable(columns=columns, dictionaries=LINEITEM_DICTIONARIES)


def _load_substrait_extension(con: duckdb.DuckDBPyConnection) -> None:
    extension_path = os.environ.get("TQG_SUBSTRAIT_EXTENSION")
    if extension_path:
        path = Path(extension_path)
        if not path.exists():
            raise DuckDBSubstraitError(f"TQG_SUBSTRAIT_EXTENSION does not exist: {path}")
        try:
            con.load_extension(str(path))
        except duckdb.Error as exc:
            raise DuckDBSubstraitError(
                f"failed to load TQG_SUBSTRAIT_EXTENSION {path}: {exc}"
            ) from exc
        return
    try:
        con.install_extension("substrait", repository="community")
        con.load_extension("substrait")
    except duckdb.Error as exc:
        raise DuckDBSubstraitError(
            "DuckDB Substrait extension is unavailable; expected "
            "INSTALL substrait FROM community; LOAD substrait to work"
        ) from exc
