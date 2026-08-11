"""Framework-level SQL admission and strict coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import duckdb

from tpch_torch.backend.physical_metadata import metadata_list, metadata_string
from tpch_torch.backend.physical_window import validate_window_projection_support
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

_STRICT_NODE_KINDS = frozenset(
    {
        OperatorKind.SCAN,
        OperatorKind.FILTER,
        OperatorKind.PROJECT,
        OperatorKind.AGGREGATE,
        OperatorKind.JOIN,
        OperatorKind.SORT,
        OperatorKind.LIMIT,
        OperatorKind.CTE,
        OperatorKind.DELIM,
        OperatorKind.SET,
        OperatorKind.WINDOW,
    }
)
_STRICT_JOIN_TYPES = frozenset(
    {"INNER", "LEFT", "RIGHT", "SEMI", "ANTI", "RIGHT_SEMI", "RIGHT_ANTI", "MARK"}
)


@dataclass(frozen=True)
class StrictCoverageGap:
    """One static reason a graph may need universal compatibility execution."""

    node_id: str
    node_name: str
    node_kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_kind": self.node_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StrictCoverageReport:
    """Static strict-path coverage summary for a lowered TQP operator graph."""

    strict_admissible: bool
    node_count: int
    gaps: tuple[StrictCoverageGap, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_admissible": self.strict_admissible,
            "node_count": self.node_count,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


@dataclass(frozen=True)
class SQLAdmission:
    """DuckDB-bound SQL plan plus static TQP/PyTorch coverage metadata."""

    plan: TQPPlan
    strict_coverage: StrictCoverageReport

    @property
    def graph(self) -> TQPOperatorGraph:
        if self.plan.operator_graph is None:
            raise ValueError("SQL admission requires a TQP operator graph")
        return self.plan.operator_graph

    def to_dict(self) -> dict[str, Any]:
        graph = self.graph
        return {
            "frontend": self.plan.frontend,
            "query_id": self.plan.query_id,
            "root_id": graph.root_id,
            "root_name": graph.root.name,
            "output_schema": [
                {"name": column.name, "type": column.type_name, "nullable": column.nullable}
                for column in graph.output_schema
            ],
            "strict_coverage": self.strict_coverage.to_dict(),
            "nodes": [_node_dict(node) for node in graph.nodes],
        }


def admit_sql(con: duckdb.DuckDBPyConnection, sql: str) -> SQLAdmission:
    """Parse/bind/plan arbitrary DuckDB SQL into a TQP graph and coverage report."""

    plan = compile_sirius_plan(con, sql)
    if plan.operator_graph is None:
        raise ValueError("Sirius-like frontend did not produce an operator graph")
    return SQLAdmission(plan, analyze_strict_coverage(plan.operator_graph))


def analyze_strict_coverage(graph: TQPOperatorGraph) -> StrictCoverageReport:
    """Return a static coverage estimate before executing any tensor operator."""

    gaps = tuple(gap for node in graph.nodes for gap in _node_gaps(node))
    return StrictCoverageReport(
        strict_admissible=not gaps,
        node_count=len(graph.nodes),
        gaps=gaps,
    )


def _node_gaps(node: TQPOperatorNode) -> tuple[StrictCoverageGap, ...]:
    if node.kind not in _STRICT_NODE_KINDS:
        return (_gap(node, f"unsupported DuckDB physical node: {node.name}"),)
    if node.kind == OperatorKind.SCAN:
        return _scan_gaps(node)
    if node.kind == OperatorKind.JOIN:
        return _join_gaps(node)
    if node.kind == OperatorKind.SET:
        return _set_gaps(node)
    if node.kind == OperatorKind.WINDOW:
        return _window_gaps(node)
    return ()


def _scan_gaps(node: TQPOperatorNode) -> tuple[StrictCoverageGap, ...]:
    normalized = node.name.strip().upper()
    if normalized in {"DUMMY_SCAN", "COLUMN_DATA_SCAN"}:
        return ()
    if metadata_string(node, "Table"):
        return ()
    return (_gap(node, "scan node is missing table metadata"),)


def _join_gaps(node: TQPOperatorNode) -> tuple[StrictCoverageGap, ...]:
    normalized = node.name.strip().upper()
    if normalized == "RIGHT_DELIM_JOIN":
        return ()
    join_type = (metadata_string(node, "Join Type") or "").upper()
    if join_type not in _STRICT_JOIN_TYPES:
        return (_gap(node, f"physical join type is not supported yet: {join_type}"),)
    if join_type == "MARK":
        return ()
    if metadata_list(node, "Conditions"):
        return ()
    return (_gap(node, "join node is missing conditions"),)


def _set_gaps(node: TQPOperatorNode) -> tuple[StrictCoverageGap, ...]:
    if node.name.strip().upper() == "UNION":
        return ()
    return (_gap(node, f"unsupported set-operation node: {node.name}"),)


def _window_gaps(node: TQPOperatorNode) -> tuple[StrictCoverageGap, ...]:
    reasons = validate_window_projection_support(metadata_list(node, "Projections"))
    return tuple(_gap(node, reason) for reason in reasons)


def _gap(node: TQPOperatorNode, reason: str) -> StrictCoverageGap:
    return StrictCoverageGap(node.node_id, node.name, str(node.kind), reason)


def _node_dict(node: TQPOperatorNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "name": node.name,
        "kind": str(node.kind),
        "children": list(node.children),
        "metadata_keys": sorted(str(key) for key in node.metadata),
        "output_slots": [_slot_dict(slot) for slot in node.output_slots],
    }


def _slot_dict(slot: Any) -> Mapping[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "name": slot.name,
        "type_name": slot.type_name,
        "aliases": list(slot.aliases),
    }
