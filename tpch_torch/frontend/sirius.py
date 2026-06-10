"""Sirius-like DuckDB frontend for TQP plans."""

from __future__ import annotations

import duckdb

from tpch_torch.ir import DuckDBPlanMetadata, TQPPlan
from tpch_torch.planner import export_duckdb_logical_plan
from tpch_torch.query_catalog import identify_tpch_query


def compile_sirius_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    """Compile SQL through DuckDB planner admission into a TQP plan."""

    duckdb_plan = export_duckdb_logical_plan(con, sql)
    return TQPPlan(
        query_id=identify_tpch_query(sql),
        source_sql=sql,
        frontend="sirius",
        duckdb_metadata=DuckDBPlanMetadata(
            logical_plan=duckdb_plan.logical_plan,
            logical_opt=duckdb_plan.logical_opt,
            physical_plan=duckdb_plan.physical_plan,
        ),
    )
