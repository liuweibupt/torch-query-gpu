"""Explicit TQP operator graph IR lowered from SQL planner output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class OperatorKind(StrEnum):
    """Backend-visible operator categories."""

    SCAN = "scan"
    FILTER = "filter"
    PROJECT = "project"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SORT = "sort"
    LIMIT = "limit"
    CTE = "cte"
    DELIM = "delim"
    COMPILED_TPCH = "compiled_tpch"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TQPOutputColumn:
    """DuckDB-bound output schema column carried across the TQP boundary."""

    name: str
    type_name: str
    nullable: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "type_name", str(self.type_name))


@dataclass(frozen=True)
class TQPOperatorNode:
    """A single immutable operator node in a TQP graph."""

    node_id: str
    kind: OperatorKind
    name: str
    children: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class TQPOperatorGraph:
    """Immutable operator graph passed from frontend to PyTorch backend."""

    source_sql: str
    query_id: int | None
    root_id: str
    nodes: tuple[TQPOperatorNode, ...]
    output_schema: tuple[TQPOutputColumn, ...] = ()
    select_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "output_schema", tuple(self.output_schema))
        object.__setattr__(self, "select_aliases", MappingProxyType(dict(self.select_aliases)))
        if self.root_id not in {node.node_id for node in self.nodes}:
            raise ValueError(f"root node is missing: {self.root_id}")

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.output_schema)

    @property
    def output_types(self) -> tuple[str, ...]:
        return tuple(column.type_name for column in self.output_schema)

    @property
    def root(self) -> TQPOperatorNode:
        return self.node_by_id(self.root_id)

    def node_by_id(self, node_id: str) -> TQPOperatorNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)
