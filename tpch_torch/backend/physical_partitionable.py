"""CoddSpeed-style partitionable execution for physical PyTorch plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import duckdb

from tpch_torch.backend.physical_metadata import metadata_list as _metadata_list, metadata_string as _metadata_string
from tpch_torch.backend.physical_partitionable_final import (
    FinalAggregateColumn,
    FinalAggregatePlan,
    merge_partitioned_aggregate_tables,
)
from tpch_torch.backend.physical_types import PhysicalTable
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

SUPPORTED_PARTITION_AGGREGATES = frozenset({"sum", "count", "count_star", "min", "max", "avg"})
_UNSUPPORTED_NODE_KINDS = frozenset({OperatorKind.JOIN, OperatorKind.CTE, OperatorKind.DELIM, OperatorKind.LIMIT})


@dataclass(frozen=True)
class PartitionConfig:
    """Explicit opt-in partitioning configuration for one scan table."""

    table: str
    chunk_size: int

    def __post_init__(self) -> None:
        if not self.table.strip():
            raise ValueError("partition table must be non-empty")
        if self.chunk_size <= 0:
            raise ValueError("partition chunk_size must be positive")


@dataclass(frozen=True)
class _AggregateColumn:
    name: str
    function: str


@dataclass(frozen=True)
class _PartitionAnalysis:
    table: str
    group_columns: tuple[str, ...]
    aggregates: tuple[_AggregateColumn, ...]
    count_column: str | None
    sort_by_group_keys: bool


def row_ranges(row_count: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    """Return non-overlapping half-open row ranges covering `row_count`."""

    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return tuple((start, min(start + chunk_size, row_count)) for start in range(0, row_count, chunk_size))


def execute_partitionable_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    config: PartitionConfig,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Execute a supported aggregate graph through local/final batch aggregation."""

    analysis = _analyze_partitionable_graph(con, graph, config)
    partial_tables = _execute_partitionable_batch_pipeline(
        con,
        graph,
        analysis.table,
        config.chunk_size,
        device,
    )
    merged = merge_partitioned_aggregate_tables(partial_tables, _final_aggregate_plan(analysis))
    return _rows_from_table(merged)


def _execute_partitionable_batch_pipeline(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    table: str,
    chunk_size: int,
    device: str,
) -> tuple[PhysicalTable, ...]:
    from tpch_torch.backend.physical_pipeline import iter_batch_pipeline

    return tuple(iter_batch_pipeline(con, graph, table=table, chunk_size=chunk_size, device=device))


def _final_aggregate_plan(analysis: _PartitionAnalysis) -> FinalAggregatePlan:
    return FinalAggregatePlan(
        group_columns=analysis.group_columns,
        aggregates=tuple(
            FinalAggregateColumn(column.name, column.function)
            for column in analysis.aggregates
        ),
        count_column=analysis.count_column,
        sort_by_group_keys=analysis.sort_by_group_keys,
    )


def _analyze_partitionable_graph(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    config: PartitionConfig,
) -> _PartitionAnalysis:
    table = config.table.strip().lower()
    _reject_unsupported_node_kinds(graph)
    scan_tables = _scan_tables(graph)
    if scan_tables != (table,):
        raise UnsupportedPlanError(
            "partitionable execution currently supports exactly one scan table; "
            f"requested={table!r} plan_tables={scan_tables!r}"
        )
    aggregate_node = _single_aggregate_node(graph)
    aliases = graph.output_names or _describe_aliases(con, graph.source_sql)
    group_columns = aliases[: len(_metadata_list(aggregate_node, "Groups"))]
    aggregates = _aggregate_columns(aggregate_node, aliases, len(group_columns))
    count_column = _count_column(aggregates)
    if any(column.function == "avg" for column in aggregates) and count_column is None:
        raise UnsupportedPlanError("partitionable AVG requires a COUNT aggregate in the fragment output")
    sort_by_group_keys = _sort_by_group_keys_only(graph, group_columns)
    return _PartitionAnalysis(table, group_columns, aggregates, count_column, sort_by_group_keys)


def _reject_unsupported_node_kinds(graph: TQPOperatorGraph) -> None:
    unsupported = sorted({node.kind.value for node in graph.nodes if node.kind in _UNSUPPORTED_NODE_KINDS})
    if unsupported:
        raise UnsupportedPlanError(
            "partitionable execution currently supports scan/filter/project/aggregate/sort only; "
            f"unsupported={unsupported}"
        )


def _scan_tables(graph: TQPOperatorGraph) -> tuple[str, ...]:
    tables = []
    for node in graph.nodes:
        if node.kind != OperatorKind.SCAN:
            continue
        table = _metadata_string(node, "Table")
        if table is not None:
            tables.append(table.lower())
    return tuple(dict.fromkeys(tables))


def _single_aggregate_node(graph: TQPOperatorGraph) -> TQPOperatorNode:
    aggregate_nodes = [node for node in graph.nodes if node.kind == OperatorKind.AGGREGATE]
    if len(aggregate_nodes) != 1:
        raise UnsupportedPlanError(
            "partitionable execution currently requires exactly one aggregate node; "
            f"found={len(aggregate_nodes)}"
        )
    return aggregate_nodes[0]


def _aggregate_columns(
    aggregate_node: TQPOperatorNode,
    output_aliases: Sequence[str],
    group_count: int,
) -> tuple[_AggregateColumn, ...]:
    aggregate_exprs = _metadata_list(aggregate_node, "Aggregates")
    aggregate_aliases = output_aliases[group_count:]
    if len(aggregate_aliases) != len(aggregate_exprs):
        raise UnsupportedPlanError("partitionable aggregate output aliases do not match aggregate expressions")
    return tuple(
        _AggregateColumn(alias, _aggregate_function(expression))
        for alias, expression in zip(aggregate_aliases, aggregate_exprs)
    )


def _aggregate_function(expression: str) -> str:
    if expression.strip().lower() == "count_star()":
        return "count_star"
    match = re.fullmatch(r'(?:"?)(sum_no_overflow|sum|avg|min|max|count)(?:"?)\(.*\)', expression.strip(), re.I)
    if match is None:
        raise UnsupportedPlanError(f"unsupported partitionable aggregate expression: {expression}")
    function = match.group(1).lower()
    return "sum" if function == "sum_no_overflow" else function


def _count_column(aggregates: Iterable[_AggregateColumn]) -> str | None:
    for column in aggregates:
        if column.function in {"count", "count_star"}:
            return column.name
    return None


def _sort_by_group_keys_only(graph: TQPOperatorGraph, group_columns: Sequence[str]) -> bool:
    sort_nodes = [node for node in graph.nodes if node.kind == OperatorKind.SORT]
    if not sort_nodes:
        return False
    group_tail_names = {column.rsplit(".", 1)[-1].lower() for column in group_columns}
    for node in sort_nodes:
        for item in _metadata_list(node, "Order By"):
            if _is_descending_order(item):
                raise UnsupportedPlanError(
                    "partitionable execution currently supports ascending ORDER BY over group keys only"
                )
            order_expr = _order_expression_without_direction(item)
            if order_expr.rsplit(".", 1)[-1].lower() not in group_tail_names:
                raise UnsupportedPlanError(
                    "partitionable execution only supports ORDER BY over group keys"
                )
    return True


def _is_descending_order(item: str) -> bool:
    return re.search(r"\s+DESC(\s+NULLS\s+(FIRST|LAST))?\s*$", item.strip(), re.I) is not None


def _order_expression_without_direction(item: str) -> str:
    return re.sub(r"\s+(ASC|DESC)(\s+NULLS\s+(FIRST|LAST))?\s*$", "", item.strip(), flags=re.I)


def _describe_aliases(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(str(row[0]) for row in rows)


def _rows_from_table(table: PhysicalTable) -> list[dict[str, Any]]:
    import tpch_torch.backend.physical as physical

    return physical._rows_from_table(table)
