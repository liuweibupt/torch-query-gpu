"""Strict DuckDB Substrait frontend for TQP plans."""

from __future__ import annotations

import duckdb

from tpch_torch.duckdb_bridge import export_substrait_json
from tpch_torch.ir import TQPPlan
from tpch_torch.runner import identify_tpch_query


def compile_substrait_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    """Compile SQL through DuckDB's real Substrait exporter into a TQP plan."""

    return TQPPlan(
        query_id=identify_tpch_query(sql),
        source_sql=sql,
        frontend="substrait",
        plan_json=export_substrait_json(con, sql),
    )
