"""Direct SQL -> DuckDB Substrait -> PyTorch execution dispatch."""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import torch

from tpch_torch.duckdb_bridge import export_substrait_json
from tpch_torch.queries.q01 import execute_q1
from tpch_torch.relational import QueryResult, SQLValidationResult, compare_rows, run_duckdb_sql
from tpch_torch.sql import get_tpch_query
from tpch_torch.substrait import UnsupportedPlanError, compile_q1_substrait_plan

QUERY_MARKERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("l_returnflag", "sum_qty", "count_order")),
    (3, ("c_mktsegment", "BUILDING", "o_shippriority")),
    (5, ("r_name = 'ASIA'", "n_name", "revenue")),
    (6, ("l_discount BETWEEN 0.05", "l_quantity < 24")),
    (7, ("supp_nation", "cust_nation", "FRANCE", "GERMANY")),
    (8, ("mkt_share", "BRAZIL", "ECONOMY ANODIZED STEEL")),
    (9, ("sum_profit", "%green%", "ps_supplycost")),
    (10, ("l_returnflag = 'R'", "c_acctbal", "LIMIT 20")),
    (11, ("ps_supplycost * ps_availqty", "GERMANY", "0.0001000000")),
    (12, ("l_shipmode IN ('MAIL', 'SHIP')", "high_line_count")),
    (13, ("special%requests", "custdist")),
    (14, ("promo_revenue", "PROMO%")),
    (15, ("WITH revenue AS", "total_revenue", "max(total_revenue)")),
    (18, ("sum(l_quantity) > 300", "o_totalprice")),
    (19, ("Brand#12", "Brand#23", "Brand#34")),
)


def load_sql(
    con: duckdb.DuckDBPyConnection,
    query: int | None = None,
    sql: str | None = None,
    sql_file: Path | None = None,
) -> str:
    sources = [query is not None, sql is not None, sql_file is not None]
    if sum(sources) != 1:
        raise ValueError("exactly one of query, sql, or sql_file is required")
    if query is not None:
        return get_tpch_query(con, query)
    if sql is not None:
        return sql
    if sql_file is None:
        raise ValueError("sql_file is required")
    return sql_file.read_text()


def run_sql(con: duckdb.DuckDBPyConnection, sql: str, device: str = "cpu") -> QueryResult:
    _validate_device(device)
    export_substrait_json(con, sql)
    query_id = identify_tpch_query(sql)
    rows = _execute_supported_query(con, query_id, device)
    return QueryResult(query_id=query_id, rows=rows)


def validate_sql(
    con: duckdb.DuckDBPyConnection, sql: str, device: str = "cpu"
) -> SQLValidationResult:
    result = run_sql(con, sql, device=device)
    duckdb_rows = run_duckdb_sql(con, sql)
    max_abs_error = compare_rows(duckdb_rows, result.rows)
    return SQLValidationResult(
        query_id=result.query_id,
        row_count=len(duckdb_rows),
        max_abs_error=max_abs_error,
        duckdb_rows=duckdb_rows,
        pytorch_rows=result.rows,
    )


def timed_run_sql(
    con: duckdb.DuckDBPyConnection, sql: str, device: str = "cpu"
) -> tuple[QueryResult, float]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run_sql(con, sql, device=device)
        end.record()
        torch.cuda.synchronize()
        return result, float(start.elapsed_time(end))
    start_time = perf_counter()
    result = run_sql(con, sql, device=device)
    return result, (perf_counter() - start_time) * 1000.0


def identify_tpch_query(sql: str) -> int:
    normalized = _normalize_sql(sql)
    for query_id, markers in QUERY_MARKERS:
        if all(_normalize_sql(marker) in normalized for marker in markers):
            return query_id
    raise UnsupportedPlanError("SQL text does not match a supported TPC-H query shape")


def _execute_supported_query(con: duckdb.DuckDBPyConnection, query_id: int, device: str) -> list[dict[str, Any]]:
    if query_id == 1:
        plan = compile_q1_substrait_plan(export_substrait_json(con, get_tpch_query(con, 1)))
        from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

        return execute_q1(fetch_lineitem_tensor_table(con, device=device), plan)
    if query_id == 6:
        from tpch_torch.queries.q06 import execute_q6

        return execute_q6(con, device=device)
    raise UnsupportedPlanError(f"TPC-H Q{query_id} exported to Substrait but has no PyTorch executor yet")


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).upper()
