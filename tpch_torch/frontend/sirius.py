"""Sirius-like DuckDB frontend for TQP plans."""

from __future__ import annotations

import duckdb

from tpch_torch.duckdb_plan_json import (
    describe_scan_table_schemas,
    describe_output_schema,
    export_duckdb_physical_plan_json,
    lower_duckdb_json_to_operator_graph,
)
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.frontend.duckdb_ast import select_expressions_by_alias
from tpch_torch.generic_sql import parse_generic_sql
from tpch_torch.ir import DuckDBPlanMetadata, TQPPlan
from tpch_torch.planner import export_duckdb_logical_plan
from tpch_torch.query_catalog import identify_tpch_query


def compile_sirius_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    """Compile SQL through DuckDB planner admission into a TQP plan."""

    duckdb_plan = export_duckdb_logical_plan(con, sql)
    generic_plan = None
    generic_error = None
    try:
        query_id = identify_tpch_query(sql)
    except UnsupportedPlanError:
        query_id = None
        try:
            generic_plan = parse_generic_sql(sql)
        except (UnsupportedPlanError, ValueError, TypeError) as exc:
            generic_error = f"{type(exc).__name__}: {exc}"
    physical_plan_json = export_duckdb_physical_plan_json(con, sql)
    operator_graph = lower_duckdb_json_to_operator_graph(
        sql,
        query_id,
        physical_plan_json,
        output_schema=describe_output_schema(con, sql),
        select_aliases=select_expressions_by_alias(con, sql),
        table_schemas=describe_scan_table_schemas(con, physical_plan_json),
    )
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
        generic_error=generic_error,
        operator_graph=operator_graph,
    )
