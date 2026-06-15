"""Graph-lowered fused physical operators."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.operator_graph import TQPOperatorGraph


def try_execute_fused_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    device: str,
) -> list[dict[str, Any]] | None:
    """Return fused rows for recognized physical graphs, otherwise None."""

    return None
