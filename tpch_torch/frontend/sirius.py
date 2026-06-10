"""Sirius-like DuckDB frontend for TQP plans."""

from __future__ import annotations

import duckdb

from tpch_torch.generic_sql import parse_generic_sql
from tpch_torch.ir import DuckDBPlanMetadata, TQPPlan
from tpch_torch.planner import export_duckdb_logical_plan
from tpch_torch.query_catalog import identify_tpch_query
from tpch_torch.substrait import UnsupportedPlanError


def compile_sirius_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    """Compile SQL through DuckDB planner admission into a TQP plan."""

    duckdb_plan = export_duckdb_logical_plan(con, sql)
    generic_plan = None
    try:
        query_id = identify_tpch_query(sql)
    except UnsupportedPlanError:
        query_id = None
        generic_plan = parse_generic_sql(sql)
    return TQPPlan(
        query_id=query_id,
        source_sql=sql,
        frontend="sirius",
        duckdb_metadata=DuckDBPlanMetadata(
            logical_plan=duckdb_plan.logical_plan,
            logical_opt=duckdb_plan.logical_opt,
            physical_plan=duckdb_plan.physical_plan,
        ),
        generic_plan=generic_plan,
    )
