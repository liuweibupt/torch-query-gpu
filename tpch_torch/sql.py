"""Canonical SQL text and lookup helpers for supported TPC-H queries."""

from __future__ import annotations

import duckdb

TPC_H_Q1_SQL = """
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from lineitem
where l_shipdate <= date '1998-09-02'
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus
""".strip()


def get_tpch_query(con: duckdb.DuckDBPyConnection, query_id: int) -> str:
    """Return TPC-H SQL text from DuckDB's tpch extension."""

    con.execute("load tpch")
    row = con.execute("select query from tpch_queries() where query_nr = ?", [query_id]).fetchone()
    if row is None:
        raise ValueError(f"unknown TPC-H query id: {query_id}")
    return str(row[0])
