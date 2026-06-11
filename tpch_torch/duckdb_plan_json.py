"""DuckDB JSON physical-plan export and lowering into TQP operator graphs."""

from __future__ import annotations

import json
from typing import Any

import duckdb

from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.planner import DuckDBPlannerError

_PLAN_JSON_COLUMN_INDEX = 1


def export_duckdb_physical_plan_json(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Return DuckDB's structured physical plan for the original SQL."""

    try:
        rows = con.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB JSON EXPLAIN failed: {exc}") from exc
    if not rows:
        raise DuckDBPlannerError("DuckDB JSON EXPLAIN returned no rows")
    raw_plan = rows[0][_PLAN_JSON_COLUMN_INDEX]
    loaded = json.loads(str(raw_plan))
    if not isinstance(loaded, list):
        raise DuckDBPlannerError("DuckDB JSON EXPLAIN did not return a plan list")
    return loaded


def lower_duckdb_json_to_operator_graph(
    source_sql: str,
    query_id: int | None,
    plan_json: list[dict[str, Any]],
) -> TQPOperatorGraph:
    """Lower DuckDB JSON physical-plan nodes to the repository's graph IR."""

    nodes: list[TQPOperatorNode] = []

    def lower_node(raw_node: dict[str, Any], path: tuple[int, ...]) -> str:
        node_id = _node_id(path)
        child_ids = tuple(
            lower_node(child, (*path, child_index))
            for child_index, child in enumerate(raw_node.get("children") or ())
        )
        name = str(raw_node.get("name", "UNKNOWN")).strip()
        nodes.append(
            TQPOperatorNode(
                node_id=node_id,
                kind=_operator_kind(name),
                name=name,
                children=child_ids,
                metadata=dict(raw_node.get("extra_info") or {}),
            )
        )
        return node_id

    if len(plan_json) != 1:
        raise DuckDBPlannerError(f"expected one DuckDB root plan, got {len(plan_json)}")
    root_id = lower_node(plan_json[0], (0,))
    return TQPOperatorGraph(
        source_sql=source_sql,
        query_id=query_id,
        root_id=root_id,
        nodes=tuple(nodes),
    )


def _node_id(path: tuple[int, ...]) -> str:
    return "n" + "_".join(str(part) for part in path)


def _operator_kind(name: str) -> OperatorKind:
    normalized = name.upper().strip()
    if normalized.endswith("SCAN") or normalized in {"SEQ_SCAN", "COLUMN_DATA_SCAN"}:
        if normalized in {"CTE_SCAN"}:
            return OperatorKind.CTE
        if normalized in {"DELIM_SCAN"}:
            return OperatorKind.DELIM
        return OperatorKind.SCAN
    if "FILTER" in normalized:
        return OperatorKind.FILTER
    if "PROJECTION" in normalized:
        return OperatorKind.PROJECT
    if "GROUP_BY" in normalized or normalized == "UNGROUPED_AGGREGATE":
        return OperatorKind.AGGREGATE
    if "JOIN" in normalized or normalized == "NESTED_LOOP_JOIN":
        return OperatorKind.JOIN
    if normalized == "ORDER_BY":
        return OperatorKind.SORT
    if normalized == "TOP_N":
        return OperatorKind.LIMIT
    if normalized == "CTE":
        return OperatorKind.CTE
    if normalized in {"DUMMY_SCAN"}:
        return OperatorKind.SCAN
    return OperatorKind.UNKNOWN
