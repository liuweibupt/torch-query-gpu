"""Direct SQL -> DuckDB Substrait -> PyTorch execution dispatch."""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import duckdb
import torch

from tpch_torch.duckdb_bridge import DuckDBSubstraitError, export_substrait_json
from tpch_torch.planner import export_duckdb_logical_plan
from tpch_torch.queries.q01 import execute_q1
from tpch_torch.relational import QueryResult, SQLValidationResult, compare_rows, run_duckdb_sql
from tpch_torch.sql import get_tpch_query
from tpch_torch.substrait import UnsupportedPlanError, compile_q1_substrait_plan
from tpch_torch.substrait import (
    Q1_GROUP_KEYS,
    Q1_ORDER_KEYS,
    Q1_REQUIRED_COLUMNS,
    Q1_SHIPDATE_CUTOFF_YYYYMMDD,
    Q1Plan,
)

QUERY_MARKERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("l_returnflag", "sum_qty", "count_order")),
    (2, ("s_acctbal", "p_type LIKE '%BRASS'", "min(ps_supplycost)")),
    (3, ("c_mktsegment", "BUILDING", "o_shippriority")),
    (4, ("o_orderpriority", "l_commitdate < l_receiptdate", "order_count")),
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
    (16, ("count(DISTINCT ps_suppkey)", "Brand#45", "Customer%Complaints")),
    (17, ("avg_yearly", "Brand#23", "MED BOX")),
    (18, ("sum(l_quantity) > 300", "o_totalprice")),
    (19, ("Brand#12", "Brand#23", "Brand#34")),
    (20, ("forest%", "0.5 * sum(l_quantity)", "CANADA")),
    (21, ("numwait", "SAUDI ARABIA", "l1.l_receiptdate > l1.l_commitdate")),
    (22, ("cntrycode", "substring(c_phone FROM 1 FOR 2)", "numcust")),
)
SUPPORTED_EXECUTOR_QUERIES: frozenset[int] = frozenset(query_id for query_id, _ in QUERY_MARKERS)
PlanSource = Literal["substrait", "duckdb-logical", "auto"]


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
    return run_sql_with_plan_source(con, sql, device=device, plan_source="substrait")


def run_sql_with_plan_source(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    plan_source: PlanSource = "substrait",
) -> QueryResult:
    _validate_device(device)
    plan_json = _admit_plan(con, sql, plan_source)
    query_id = identify_tpch_query(sql)
    rows = _execute_supported_query(con, query_id, plan_json, device)
    return QueryResult(query_id=query_id, rows=rows)


def validate_sql(
    con: duckdb.DuckDBPyConnection, sql: str, device: str = "cpu"
) -> SQLValidationResult:
    return validate_sql_with_plan_source(con, sql, device=device, plan_source="substrait")


def validate_sql_with_plan_source(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    plan_source: PlanSource = "substrait",
) -> SQLValidationResult:
    result = run_sql_with_plan_source(con, sql, device=device, plan_source=plan_source)
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
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    plan_source: PlanSource = "substrait",
) -> tuple[QueryResult, float]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run_sql_with_plan_source(con, sql, device=device, plan_source=plan_source)
        end.record()
        torch.cuda.synchronize()
        return result, float(start.elapsed_time(end))
    start_time = perf_counter()
    result = run_sql_with_plan_source(con, sql, device=device, plan_source=plan_source)
    return result, (perf_counter() - start_time) * 1000.0


def _admit_plan(con: duckdb.DuckDBPyConnection, sql: str, plan_source: PlanSource) -> dict[str, Any]:
    if plan_source == "substrait":
        return export_substrait_json(con, sql)
    if plan_source == "duckdb-logical":
        export_duckdb_logical_plan(con, sql)
        return {}
    if plan_source == "auto":
        try:
            return export_substrait_json(con, sql)
        except DuckDBSubstraitError:
            export_duckdb_logical_plan(con, sql)
            return {}
    raise ValueError(f"unknown plan_source: {plan_source}")


def identify_tpch_query(sql: str) -> int:
    normalized = _normalize_sql(sql)
    for query_id, markers in QUERY_MARKERS:
        if all(_normalize_sql(marker) in normalized for marker in markers):
            return query_id
    raise UnsupportedPlanError("SQL text does not match a supported TPC-H query shape")


def is_query_executor_supported(query_id: int) -> bool:
    return query_id in SUPPORTED_EXECUTOR_QUERIES


def _execute_supported_query(
    con: duckdb.DuckDBPyConnection, query_id: int, plan_json: dict[str, Any], device: str
) -> list[dict[str, Any]]:
    if query_id == 1:
        plan = _compile_q1_plan(plan_json)
        from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

        return execute_q1(fetch_lineitem_tensor_table(con, device=device), plan)
    if query_id == 6:
        from tpch_torch.queries.q06 import execute_q6

        return execute_q6(con, device=device)
    executor_by_query = {
        2: "q02",
        3: "q03",
        4: "q04",
        5: "q05",
        7: "q07",
        8: "q08",
        9: "q09",
        10: "q10",
        11: "q11",
        12: "q12",
        13: "q13",
        14: "q14",
        15: "q15",
        16: "q16",
        17: "q17",
        18: "q18",
        19: "q19",
        20: "q20",
        21: "q21",
        22: "q22",
    }
    module_name = executor_by_query.get(query_id)
    if module_name is None:
        raise UnsupportedPlanError(f"TPC-H Q{query_id} exported to Substrait but has no PyTorch executor yet")
    module = __import__(f"tpch_torch.queries.{module_name}", fromlist=[f"execute_q{query_id}"])
    return getattr(module, f"execute_q{query_id}")(con, device=device)


def _compile_q1_plan(plan_json: dict[str, Any]) -> Q1Plan:
    if plan_json:
        return compile_q1_substrait_plan(plan_json)
    return Q1Plan(
        table_name="lineitem",
        shipdate_cutoff_yyyymmdd=Q1_SHIPDATE_CUTOFF_YYYYMMDD,
        required_columns=Q1_REQUIRED_COLUMNS,
        group_keys=Q1_GROUP_KEYS,
        order_keys=Q1_ORDER_KEYS,
    )


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).upper()
