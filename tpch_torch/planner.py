"""DuckDB logical-plan admission inspired by Sirius."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


@dataclass(frozen=True)
class DuckDBLogicalPlan:
    """Textual DuckDB plans captured from EXPLAIN output."""

    logical_plan: str
    logical_opt: str
    physical_plan: str


class DuckDBPlannerError(RuntimeError):
    """Raised when DuckDB cannot parse or plan the original SQL."""


def export_duckdb_logical_plan(con: object, sql: str) -> DuckDBLogicalPlan:
    """Ask DuckDB to parse/plan original SQL and return textual EXPLAIN sections."""

    try:
        con.execute("PRAGMA explain_output='all'")
        rows = con.execute(f"EXPLAIN {sql}").fetchall()
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB EXPLAIN failed: {exc}") from exc
    sections = {str(name): str(plan) for name, plan in rows}
    return DuckDBLogicalPlan(
        logical_plan=sections.get("logical_plan", ""),
        logical_opt=sections.get("logical_opt", ""),
        physical_plan=sections.get("physical_plan", ""),
    )
