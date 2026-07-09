"""Explicit scan-chunk execution for safe physical PyTorch plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from tpch_torch.backend.physical_partitionable import row_ranges
from tpch_torch.backend.physical_scan import scan_row_count
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

_SAFE_NODE_KINDS = frozenset({OperatorKind.SCAN, OperatorKind.FILTER, OperatorKind.PROJECT})


@dataclass(frozen=True)
class ScanChunkConfig:
    """Explicit opt-in scan chunking for one scan/filter/project table path."""

    table: str
    chunk_size: int

    def __post_init__(self) -> None:
        if not self.table.strip():
            raise ValueError("scan chunk table must be non-empty")
        if self.chunk_size <= 0:
            raise ValueError("scan chunk_size must be positive")


def execute_chunked_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    config: ScanChunkConfig,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Execute a safe physical graph one scan chunk at a time."""

    from tpch_torch.backend.physical import PhysicalPlanExecutor

    table = _analyze_chunked_scan_graph(graph, config)
    rows: list[dict[str, Any]] = []
    total, _ = scan_row_count(con, table, None)
    for start, end in row_ranges(total, config.chunk_size):
        executor = PhysicalPlanExecutor(
            con,
            graph,
            device=device,
            scan_ranges={table: (start, end)},
            scan_chunk_sizes={table: config.chunk_size},
            enable_fusion=False,
        )
        rows.extend(executor.execute())
    return rows


def _analyze_chunked_scan_graph(graph: TQPOperatorGraph, config: ScanChunkConfig) -> str:
    requested = config.table.strip().lower()
    _reject_global_or_unsafe_nodes(graph)
    scan_tables = _scan_tables(graph)
    if scan_tables != (requested,):
        raise UnsupportedPlanError(
            "scan chunk execution currently supports exactly one matching scan table; "
            f"requested={requested!r} plan_tables={scan_tables!r}"
        )
    return requested


def _reject_global_or_unsafe_nodes(graph: TQPOperatorGraph) -> None:
    unsafe = sorted({node.kind.value for node in graph.nodes if node.kind not in _SAFE_NODE_KINDS})
    if not unsafe:
        return
    if OperatorKind.AGGREGATE.value in unsafe:
        raise UnsupportedPlanError(
            "scan chunk execution does not merge aggregate state; use PartitionConfig for aggregate fragments"
        )
    if OperatorKind.JOIN.value in unsafe:
        raise UnsupportedPlanError("scan chunk execution does not support join plans yet")
    raise UnsupportedPlanError(
        "scan chunk execution currently supports scan/filter/project only; "
        f"unsupported={unsafe}"
    )


def _scan_tables(graph: TQPOperatorGraph) -> tuple[str, ...]:
    tables: list[str] = []
    for node in graph.nodes:
        if node.kind != OperatorKind.SCAN:
            continue
        table = _metadata_string(node, "Table")
        if table is not None:
            tables.append(table.lower())
    return tuple(dict.fromkeys(tables))


def _metadata_list(node: TQPOperatorNode, key: str) -> tuple[str, ...]:
    value = node.metadata.get(key)
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _metadata_string(node: TQPOperatorNode, key: str) -> str | None:
    values = _metadata_list(node, key)
    return values[0] if values else None
