"""CoddSpeed-style partitionable execution for physical PyTorch plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import duckdb

from tpch_torch.backend.physical_metadata import metadata_list as _metadata_list, metadata_string as _metadata_string
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
    partial_rows = _execute_partitionable_batch_pipeline(
        con,
        graph,
        analysis.table,
        config.chunk_size,
        device,
    )
    return _merge_partial_rows(partial_rows, analysis)


def _execute_partitionable_batch_pipeline(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    table: str,
    chunk_size: int,
    device: str,
) -> list[dict[str, Any]]:
    from tpch_torch.backend.physical_pipeline import execute_batch_pipeline

    return execute_batch_pipeline(con, graph, table=table, chunk_size=chunk_size, device=device)


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


def _merge_partial_rows(rows: Sequence[dict[str, Any]], analysis: _PartitionAnalysis) -> list[dict[str, Any]]:
    if not analysis.group_columns:
        return [_finalize_group((), rows, analysis)]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[column] for column in analysis.group_columns), []).append(row)
    result = [_finalize_group(key, group_rows, analysis) for key, group_rows in grouped.items()]
    if analysis.sort_by_group_keys:
        result.sort(key=lambda row: tuple(row[column] for column in analysis.group_columns))
    return result


def _finalize_group(
    key: tuple[Any, ...],
    rows: Sequence[dict[str, Any]],
    analysis: _PartitionAnalysis,
) -> dict[str, Any]:
    output = {column: value for column, value in zip(analysis.group_columns, key)}
    for aggregate in analysis.aggregates:
        output[aggregate.name] = _merge_aggregate_value(aggregate, rows, analysis.count_column)
    return output


def _merge_aggregate_value(
    aggregate: _AggregateColumn,
    rows: Sequence[dict[str, Any]],
    count_column: str | None,
) -> Any:
    if aggregate.function in {"sum", "count", "count_star"}:
        return _sum_values(rows, aggregate.name, aggregate.function in {"count", "count_star"})
    if aggregate.function == "min":
        return _extreme_value(rows, aggregate.name, min)
    if aggregate.function == "max":
        return _extreme_value(rows, aggregate.name, max)
    if aggregate.function == "avg":
        return _weighted_average(rows, aggregate.name, count_column)
    raise UnsupportedPlanError(f"unsupported partition aggregate merge: {aggregate.function}")


def _sum_values(rows: Sequence[dict[str, Any]], column: str, zero_when_empty: bool) -> Any:
    values = [row[column] for row in rows if row[column] is not None]
    if not values:
        return 0 if zero_when_empty else None
    return sum(values)


def _extreme_value(rows: Sequence[dict[str, Any]], column: str, reducer) -> Any:
    values = [row[column] for row in rows if row[column] is not None]
    return None if not values else reducer(values)


def _weighted_average(rows: Sequence[dict[str, Any]], column: str, count_column: str | None) -> float | None:
    if count_column is None:
        raise UnsupportedPlanError("partitionable AVG merge requires count_column")
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row[column]
        count = row[count_column]
        if value is None or count is None:
            continue
        numerator += float(value) * float(count)
        denominator += float(count)
    return None if denominator == 0.0 else numerator / denominator


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


def _table_row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(con.execute(f"select count(*) from {table}").fetchone()[0])


def _describe_aliases(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(str(row[0]) for row in rows)
