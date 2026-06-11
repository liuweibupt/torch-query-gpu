"""Internal TQP plan IR shared by frontends and backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tpch_torch.operator_graph import TQPOperatorGraph

FrontendName = Literal["sirius", "substrait"]


@dataclass(frozen=True)
class DuckDBPlanMetadata:
    """DuckDB logical/physical plan text captured at frontend admission time."""

    logical_plan: str = ""
    logical_opt: str = ""
    physical_plan: str = ""


@dataclass(frozen=True)
class TQPPlan:
    """Internal query plan passed from a TQP frontend to an execution backend."""

    query_id: int | None
    source_sql: str
    frontend: FrontendName
    duckdb_metadata: DuckDBPlanMetadata | None = None
    plan_json: dict[str, Any] | None = None
    generic_plan: Any | None = None
    generic_error: str | None = None
    operator_graph: TQPOperatorGraph | None = None
