"""DuckDB integration for TPC-H data, Substrait export, and baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.storage import TensorTable, table_from_columnar

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
    l_returnflag,
    l_linestatus,
    l_quantity::double as l_quantity,
    l_extendedprice::double as l_extendedprice,
    l_discount::double as l_discount,
    l_tax::double as l_tax,
    strftime(l_shipdate, '%Y%m%d')::integer as l_shipdate
from lineitem
""".strip()


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

    try:
        con.execute("install tpch")
        con.execute("load tpch")
        con.execute("call dbgen(sf = ?)", [scale_factor])
    except duckdb.Error as exc:
        raise DuckDBTPCHError(f"failed to generate TPC-H data: {exc}") from exc


def export_substrait_json(
    con: duckdb.DuckDBPyConnection, sql: str = TPC_H_Q1_SQL
) -> dict[str, Any]:
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

    columnar = con.execute(LINEITEM_SELECT_FOR_TORCH).fetchnumpy()
    return table_from_columnar(columnar, device=device)


def run_duckdb_q1(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Run canonical Q1 in DuckDB and return normalized Python rows."""

    result = con.execute(TPC_H_Q1_SQL)
    column_names = [description[0] for description in result.description]
    return [_normalize_result_row(column_names, row) for row in result.fetchall()]


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


def _normalize_result_row(column_names: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column_name, value in zip(column_names, row):
        if column_name == "count_order":
            normalized[column_name] = int(value)
        elif column_name in {"l_returnflag", "l_linestatus"}:
            normalized[column_name] = str(value)
        else:
            normalized[column_name] = float(value)
    return normalized
