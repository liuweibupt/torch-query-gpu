"""DuckDB JSON physical-plan export and lowering into TQP operator graphs."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import duckdb

from tpch_torch.operator_graph import OperatorKind, TQPOutputColumn, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.operator_slot_binding import bind_node_slots
from tpch_torch.planner import DuckDBPlannerError

_PLAN_JSON_COLUMN_INDEX = 1

_CANONICAL_SEQUENCE_KEYS = {
    "Projections": "projections",
    "Filters": "filters",
    "Aggregates": "aggregates",
    "Groups": "groups",
    "Order By": "order_by",
    "Conditions": "conditions",
    "Expressions": "expressions",
}

_CANONICAL_SCALAR_KEYS = {
    "Table": "table",
    "Type": "scan_type",
    "Join Type": "join_type",
    "Delim Index": "delim_index",
    "Table Index": "table_index",
    "CTE Index": "cte_index",
    "Top": "top",
    "Limit": "limit",
    "Expression": "expression",
}


def export_duckdb_physical_plan_json(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Return DuckDB's structured physical plan for the original SQL."""

    try:
        con.execute("PRAGMA explain_output='physical_only'")
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


def describe_output_schema(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[TQPOutputColumn, ...]:
    """Return DuckDB-bound output names/types, mirroring Sirius prepared schema usage."""

    try:
        rows = con.execute(f"DESCRIBE {sql}").fetchall()
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB DESCRIBE failed: {exc}") from exc
    return tuple(TQPOutputColumn(str(row[0]), str(row[1]), _nullable(row[2])) for row in rows)


def describe_output_columns(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    """Return output names from DuckDB binding instead of backend SQL text parsing."""

    return tuple(column.name for column in describe_output_schema(con, sql))


def describe_scan_table_schemas(
    con: duckdb.DuckDBPyConnection,
    plan_json: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[TQPOutputColumn, ...]]:
    """Return DuckDB catalog schemas for base tables present in a physical plan."""

    return {
        table_name: _describe_table_schema(con, table_name)
        for table_name in sorted(_scan_table_names(plan_json))
    }


def lower_duckdb_json_to_operator_graph(
    source_sql: str,
    query_id: int | None,
    plan_json: list[dict[str, Any]],
    *,
    output_schema: Sequence[TQPOutputColumn] = (),
    select_aliases: Mapping[str, str] | None = None,
    table_schemas: Mapping[str, Sequence[TQPOutputColumn]] | None = None,
) -> TQPOperatorGraph:
    """Lower DuckDB JSON physical-plan nodes to the repository's graph IR."""

    nodes: list[TQPOperatorNode] = []
    schema = tuple(output_schema)
    slots_by_node: dict[str, tuple[Any, ...]] = {}

    def lower_node(raw_node: dict[str, Any], path: tuple[int, ...]) -> str:
        node_id = _node_id(path)
        child_ids = tuple(
            lower_node(child, (*path, child_index))
            for child_index, child in enumerate(raw_node.get("children") or ())
        )
        name = str(raw_node.get("name", "UNKNOWN")).strip()
        kind = _operator_kind(name)
        raw_metadata = _normalized_metadata(
            raw_node.get("extra_info") or {},
            is_root=path == (0,),
            output_schema=schema,
        )
        _attach_scan_schema(raw_metadata, kind, table_schemas or {})
        metadata, output_slots = bind_node_slots(
            node_id=node_id,
            kind=kind,
            metadata=raw_metadata,
            child_slots=tuple(slots_by_node[child_id] for child_id in child_ids),
            output_schema=schema,
            select_aliases=select_aliases or {},
            is_root=path == (0,),
        )
        nodes.append(
            TQPOperatorNode(
                node_id=node_id,
                kind=kind,
                name=name,
                children=child_ids,
                metadata=metadata,
                output_slots=output_slots,
            )
        )
        slots_by_node[node_id] = output_slots
        return node_id

    if len(plan_json) != 1:
        raise DuckDBPlannerError(f"expected one DuckDB root plan, got {len(plan_json)}")
    root_id = lower_node(plan_json[0], (0,))
    return TQPOperatorGraph(
        source_sql=source_sql,
        query_id=query_id,
        root_id=root_id,
        nodes=tuple(nodes),
        output_schema=schema,
        select_aliases=select_aliases or {},
    )


def _nullable(value: Any) -> bool | None:
    text = str(value).strip().upper()
    if text == "YES":
        return True
    if text == "NO":
        return False
    return None


def _describe_table_schema(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
) -> tuple[TQPOutputColumn, ...]:
    try:
        rows = con.execute("select * from pragma_table_info(?)", [table_name]).fetchall()
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB table schema lookup failed for {table_name}: {exc}") from exc
    return tuple(TQPOutputColumn(str(row[1]), str(row[2]), not bool(row[3])) for row in rows)


def _scan_table_names(plan_json: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for node in plan_json:
        extra_info = node.get("extra_info") or {}
        table_name = extra_info.get("Table")
        if table_name:
            names.add(str(table_name).strip())
        names.update(_scan_table_names(tuple(node.get("children") or ())))
    return names


def _attach_scan_schema(
    metadata: dict[str, Any],
    kind: OperatorKind,
    table_schemas: Mapping[str, Sequence[TQPOutputColumn]],
) -> None:
    if kind != OperatorKind.SCAN:
        return
    table_name = _metadata_scalar(metadata.get("table") or metadata.get("Table") or "")
    if not table_name or table_name not in table_schemas:
        return
    metadata["scan_output_types"] = {
        column.name: column.type_name
        for column in table_schemas[table_name]
    }
    metadata["scan_output_nullable"] = {
        column.name: column.nullable
        for column in table_schemas[table_name]
    }


def _normalized_metadata(
    extra_info: dict[str, Any],
    *,
    is_root: bool,
    output_schema: tuple[TQPOutputColumn, ...],
) -> dict[str, Any]:
    metadata = dict(extra_info)
    for raw_key, canonical_key in _CANONICAL_SEQUENCE_KEYS.items():
        if raw_key in extra_info:
            metadata[canonical_key] = _metadata_tuple(extra_info[raw_key])
    for raw_key, canonical_key in _CANONICAL_SCALAR_KEYS.items():
        if raw_key in extra_info:
            metadata[canonical_key] = _metadata_scalar(extra_info[raw_key])
    if "Estimated Cardinality" in extra_info:
        metadata["estimated_cardinality"] = _metadata_int(extra_info["Estimated Cardinality"])
    if "Projections" in extra_info:
        metadata["projection_count"] = len(metadata["projections"])
    if is_root and output_schema:
        metadata["output_names"] = tuple(column.name for column in output_schema)
        metadata["output_types"] = tuple(column.type_name for column in output_schema)
    return metadata


def _metadata_tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _metadata_scalar(value: Any) -> str:
    return str(value).strip()


def _metadata_int(value: Any) -> int | None:
    text = str(value).strip()
    if not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


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
    if "GROUP_BY" in normalized or normalized in {"AGGREGATE", "UNGROUPED_AGGREGATE"}:
        return OperatorKind.AGGREGATE
    if "JOIN" in normalized or normalized == "NESTED_LOOP_JOIN":
        return OperatorKind.JOIN
    if normalized == "ORDER_BY":
        return OperatorKind.SORT
    if normalized == "TOP_N":
        return OperatorKind.LIMIT
    if normalized == "CTE":
        return OperatorKind.CTE
    if normalized == "UNION":
        return OperatorKind.SET
    if normalized == "WINDOW":
        return OperatorKind.WINDOW
    if normalized in {"DUMMY_SCAN"}:
        return OperatorKind.SCAN
    return OperatorKind.UNKNOWN
