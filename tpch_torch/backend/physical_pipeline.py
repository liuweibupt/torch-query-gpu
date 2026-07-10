"""Batch-oriented physical pipeline operators for safe local plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence

import duckdb
import torch

from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_expr import expression_sort_key_name, strip_order_direction
from tpch_torch.backend.physical_pipeline_aggregate import LocalAggregateBatchOperator
from tpch_torch.backend.physical_projection import (
    aggregate_order_alias,
    matching_expression_alias,
    order_alias_value,
    normalize_projection_expressions,
    projection_output_name,
    projection_value_expression,
    resolve_alias_projections,
)
from tpch_torch.backend.physical_projection_binding import parent_bound_projection_expression
from tpch_torch.backend.physical_required import parents_by_child, required_columns_from_parents
from tpch_torch.backend.physical_sql import select_expressions_by_alias
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.record_batch import BatchMeta, ColumnStorage, ColumnType, TensorRecordBatch

_ROW_ID = "__rowid__"


class BatchOperator(Protocol):
    """Pull-based tensor batch operator."""

    def next_batch(self) -> PhysicalTable | None:
        """Return the next tensor-backed batch, or None when exhausted."""


@dataclass(frozen=True)
class PipelineContext:
    con: duckdb.DuckDBPyConnection
    graph: TQPOperatorGraph
    device: str
    chunk_size: int
    select_aliases: dict[str, str]
    parents: dict[str, tuple[str, ...]]


@dataclass
class ScanBatchOperator:
    """Scan one physical table as TensorRecordBatch-backed chunks."""

    context: PipelineContext
    node: TQPOperatorNode
    table_name: str
    fetched_columns: tuple[str, ...]
    projected_columns: tuple[str, ...]
    filters: tuple[str, ...]
    _chunks: Iterator[PhysicalTable] | None = None

    def next_batch(self) -> PhysicalTable | None:
        if self._chunks is None:
            self._chunks = self._iter_scan_chunks()
        for table in self._chunks:
            filtered = _apply_scan_filters(table, self.filters)
            return filtered
        return None

    def _iter_scan_chunks(self) -> Iterator[PhysicalTable]:
        import tpch_torch.backend.physical as physical

        total, _ = physical.scan_row_count(self.context.con, self.table_name, None)
        for chunk_index, start in enumerate(range(0, total, self.context.chunk_size)):
            end = min(start + self.context.chunk_size, total)
            if not self.fetched_columns:
                yield _rowid_only_table(
                    self.table_name,
                    start,
                    end,
                    self.context.chunk_size,
                    chunk_index,
                    self.context.device,
                )
                continue
            yield physical.fetch_physical_table(
                self.context.con,
                self.table_name,
                self.fetched_columns,
                self.projected_columns,
                self.context.device,
                scan_range=(start, end),
                chunk_size=self.context.chunk_size,
                chunk_index=chunk_index,
            )


@dataclass(frozen=True)
class FilterBatchOperator:
    child: BatchOperator
    expression: str

    def next_batch(self) -> PhysicalTable | None:
        batch = self.child.next_batch()
        if batch is None:
            return None
        value = evaluate_expression(batch, self.expression)
        return batch.filter(_filter_mask(value))


@dataclass(frozen=True)
class ProjectBatchOperator:
    context: PipelineContext
    node: TQPOperatorNode
    child: BatchOperator

    def next_batch(self) -> PhysicalTable | None:
        batch = self.child.next_batch()
        if batch is None:
            return None
        return _project_batch(self.context, self.node, batch)


@dataclass(frozen=True)
class SortBatchOperator:
    context: PipelineContext
    node: TQPOperatorNode
    child: BatchOperator

    def next_batch(self) -> PhysicalTable | None:
        batch = self.child.next_batch()
        if batch is None:
            return None
        order_items = _metadata_list(self.node, "Order By")
        return batch if not order_items else _sort_table(self.context, batch, order_items)


def execute_batch_pipeline(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    *,
    table: str,
    chunk_size: int,
    device: str,
) -> list[dict[str, Any]]:
    """Execute a scan/filter/project graph via pull-based batch operators."""

    operator = build_batch_pipeline(con, graph, table=table, chunk_size=chunk_size, device=device)
    aliases = _describe_aliases(con, graph.source_sql)
    rows: list[dict[str, Any]] = []
    while True:
        batch = operator.next_batch()
        if batch is None:
            return rows
        table_batch = _trim_to_output_arity(batch, len(aliases))
        rows.extend(_rows_from_table(_rename_for_output(table_batch, aliases)))


def build_batch_pipeline(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    *,
    table: str,
    chunk_size: int,
    device: str,
) -> BatchOperator:
    """Build a pull-based batch pipeline for safe scan/filter/project plans."""

    context = PipelineContext(
        con=con,
        graph=graph,
        device=device,
        chunk_size=chunk_size,
        select_aliases=select_expressions_by_alias(graph.source_sql),
        parents=parents_by_child(graph),
    )
    return _build_operator(context, graph.root_id, table.lower())


def _build_operator(context: PipelineContext, node_id: str, table: str) -> BatchOperator:
    node = context.graph.node_by_id(node_id)
    if node.kind == OperatorKind.SCAN:
        return _scan_operator(context, node, table)
    if node.kind == OperatorKind.FILTER:
        return FilterBatchOperator(_single_child_operator(context, node, table), _required_string(node, "Expression"))
    if node.kind == OperatorKind.PROJECT:
        return ProjectBatchOperator(context, node, _single_child_operator(context, node, table))
    if node.kind == OperatorKind.AGGREGATE:
        return LocalAggregateBatchOperator(_single_child_operator(context, node, table), node)
    if node.kind == OperatorKind.SORT:
        return SortBatchOperator(context, node, _single_child_operator(context, node, table))
    raise UnsupportedPlanError(f"batch pipeline does not support node: {node.name}")


def _scan_operator(context: PipelineContext, node: TQPOperatorNode, requested_table: str) -> ScanBatchOperator:
    table_name = _required_string(node, "Table")
    if table_name.lower() != requested_table:
        raise UnsupportedPlanError(f"batch pipeline scan table mismatch: {table_name} != {requested_table}")
    projected_columns = _metadata_list(node, "Projections")
    filters = _metadata_list(node, "Filters")
    if not projected_columns:
        projected_columns = _required_scan_columns(context, table_name, node.node_id)
    fetched_columns = tuple(dict.fromkeys((*projected_columns, *_filter_columns(context.con, table_name, filters))))
    return ScanBatchOperator(context, node, table_name, fetched_columns, tuple(projected_columns), filters)


def _rowid_only_table(
    table_name: str,
    start: int,
    end: int,
    chunk_size: int,
    chunk_index: int,
    device: str,
) -> PhysicalTable:
    rowids = torch.arange(start, end, dtype=torch.int64, device=device)
    batch = TensorRecordBatch.from_storages(
        columns={_ROW_ID: ColumnStorage.fixed(rowids)},
        types={_ROW_ID: ColumnType.int64(_ROW_ID)},
        batch_meta=BatchMeta(
            row_count=int(rowids.numel()),
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            source_offset=start,
            device=torch.device(device),
        ),
    )
    return PhysicalTable.from_batch(table_name, batch, order=(_ROW_ID,))


def _single_child_operator(context: PipelineContext, node: TQPOperatorNode, table: str) -> BatchOperator:
    if len(node.children) != 1:
        raise UnsupportedPlanError(f"{node.name} expects one child")
    return _build_operator(context, node.children[0], table)


def _project_batch(context: PipelineContext, node: TQPOperatorNode, child: PhysicalTable) -> PhysicalTable:
    expressions = normalize_projection_expressions(_metadata_list(node, "Projections"))
    if not expressions:
        return child
    expressions = resolve_alias_projections(context.select_aliases, child, expressions)
    parent_required = required_columns_from_parents(context.graph, context.parents, node.node_id, _metadata_list)
    items = []
    for index, expression in enumerate(expressions):
        value_expression = parent_bound_projection_expression(
            child,
            expression,
            len(expressions),
            parent_required,
        ) or projection_value_expression(context.select_aliases, child, expression)
        value = evaluate_expression(child, value_expression)
        if value.is_literal:
            value = _materialize_literal(value, child)
        name, aliases = projection_output_name(child, expression, index, value, context.select_aliases)
        items.append((name, value, aliases))
    return PhysicalTable.projected("projection", items, child.row_count)


def _sort_table(
    context: PipelineContext,
    table: PhysicalTable,
    order_items: Sequence[str],
) -> PhysicalTable:
    result = table
    for raw_item in reversed(tuple(order_items)):
        expr, descending = strip_order_direction(raw_item)
        key_name = expression_sort_key_name(expr)
        key = _sort_value(context, result, expr, key_name).require_tensor()
        order = torch.argsort(key, descending=descending, stable=True)
        result = result.gather(order)
    return result


def _sort_value(
    context: PipelineContext,
    table: PhysicalTable,
    expression: str,
    key_name: str,
) -> PhysicalValue:
    try:
        return table.value_named(key_name)
    except KeyError:
        alias = aggregate_order_alias(context.select_aliases, table, expression)
        if alias is not None:
            return table.value_named(alias)
        expression_alias = matching_expression_alias(table, expression)
        if expression_alias is not None:
            return table.value_named(expression_alias)
        alias_value = order_alias_value(context.select_aliases, table, key_name)
        if alias_value is not None:
            return evaluate_expression(table, alias_value)
        return evaluate_expression(table, expression)


def _apply_scan_filters(table: PhysicalTable, filters: Sequence[str]) -> PhysicalTable:
    result = table
    for raw_filter in filters:
        filter_text = raw_filter.strip()
        if filter_text.lower().startswith("optional:"):
            continue
        result = result.filter(_filter_mask(evaluate_expression(result, filter_text)))
    return result


def _filter_mask(value: PhysicalValue) -> torch.Tensor:
    mask = value.require_tensor()
    if value.valid is None:
        return mask
    return mask & value.valid.to(device=mask.device)


def _materialize_literal(value: PhysicalValue, table: PhysicalTable) -> PhysicalValue:
    import tpch_torch.backend.physical as physical

    return physical._materialize_literal(value, table)


def _rename_for_output(table: PhysicalTable, aliases: Sequence[str]) -> PhysicalTable:
    import tpch_torch.backend.physical as physical

    return physical._rename_for_output(table, aliases)


def _trim_to_output_arity(table: PhysicalTable, output_arity: int) -> PhysicalTable:
    import tpch_torch.backend.physical as physical

    return physical._trim_to_output_arity(table, output_arity)


def _rows_from_table(table: PhysicalTable) -> list[dict[str, Any]]:
    import tpch_torch.backend.physical as physical

    return physical._rows_from_table(table)


def _describe_aliases(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in con.execute(f"DESCRIBE {sql}").fetchall())


def _filter_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    filters: Sequence[str],
) -> tuple[str, ...]:
    import tpch_torch.backend.physical as physical

    return physical._filter_columns(con, table_name, filters)


def _required_scan_columns(context: PipelineContext, table_name: str, node_id: str) -> tuple[str, ...]:
    import tpch_torch.backend.physical as physical

    return physical._required_scan_columns(context.con, table_name, context.graph, context.parents, node_id)


def _metadata_list(node: TQPOperatorNode, key: str) -> tuple[str, ...]:
    value = node.metadata.get(key)
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _required_string(node: TQPOperatorNode, key: str) -> str:
    values = _metadata_list(node, key)
    if not values:
        raise UnsupportedPlanError(f"{node.name} is missing {key} metadata")
    return values[0]
