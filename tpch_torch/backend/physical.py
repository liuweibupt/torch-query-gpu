"""DuckDB JSON physical-plan execution with PyTorch tensors."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import duckdb
import torch

from tpch_torch.backend.physical_expr import (
    evaluate_expression,
    expression_sort_key_name,
    strip_order_direction,
)
from tpch_torch.backend import physical_fusion
from tpch_torch.backend.physical_aggregate import aggregate_specs, execute_grouped_aggregate, execute_ungrouped_aggregate
from tpch_torch.backend.physical_delim import build_delim_table, execute_delim_join_result
from tpch_torch.backend.physical_join import (
    inner_join_indices as _inner_join_indices,
    try_execute_scalar_nested_loop_join,
)
from tpch_torch.backend.physical_join_exec import execute_join_node as _execute_join_node, join_conditions
from tpch_torch.backend.physical_metadata import metadata_list as _metadata_list, metadata_string as _metadata_string
from tpch_torch.backend.physical_mark import execute_literal_mark_join, execute_mark_join
from tpch_torch.backend.physical_projection import (
    aggregate_order_alias,
    matching_expression_alias,
    normalize_projection_expressions,
    order_alias_value,
    projection_output_name,
    projection_value_expression,
    resolve_alias_projections,
)
from tpch_torch.backend.physical_projection_binding import parent_bound_projection_expression
from tpch_torch.backend.physical_required import parents_by_child, required_columns_from_parents
from tpch_torch.backend.physical_sql import select_expressions_by_alias
from tpch_torch.backend.physical_scan import (
    fetch_physical_table,
    fetch_physical_table_stream,
    scan_row_count,
)
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue, table_device
from tpch_torch.backend.physical_union import execute_union_node
from tpch_torch.backend.physical_window import execute_window_node
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

_ROW_ID = "__rowid__"
_aggregate_specs = aggregate_specs

class PhysicalPlanExecutor:
    """Interpret supported DuckDB physical-plan nodes using tensor operators."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        graph: TQPOperatorGraph,
        device: str = "cpu",
        *,
        scan_ranges: Mapping[str, tuple[int, int]] | None = None,
        scan_chunk_sizes: Mapping[str, int] | None = None,
        enable_fusion: bool = True,
    ):
        self._con = con
        self._graph = graph
        self._device = device
        self._scan_ranges = {key.lower(): value for key, value in (scan_ranges or {}).items()}
        self._scan_chunk_sizes = {key.lower(): value for key, value in (scan_chunk_sizes or {}).items()}
        self._enable_fusion = enable_fusion
        self._select_aliases = dict(graph.select_aliases) or select_expressions_by_alias(graph.source_sql)
        self._parents = parents_by_child(graph)
        self._delim_tables: dict[str, PhysicalTable] = {}
        self._cte_tables: dict[str, PhysicalTable] = {}

    def execute(self) -> list[dict[str, Any]]:
        if self._enable_fusion:
            fused_rows = physical_fusion.try_execute_fused_physical_plan(
                self._con, self._graph, self._device, self._scan_ranges
            )
            if fused_rows is not None:
                return fused_rows
        table = self._execute_node(self._graph.root_id)
        aliases = _output_aliases(self._con, self._graph)
        table = _trim_to_output_arity(table, len(aliases))
        return _rows_from_table(_rename_for_output(table, aliases))

    def _execute_node(self, node_id: str) -> PhysicalTable:
        node = self._graph.node_by_id(node_id)
        normalized = node.name.strip().upper()
        if node.kind == OperatorKind.SCAN:
            return self._execute_scan(node)
        if node.kind == OperatorKind.FILTER:
            return self._execute_filter(node)
        if node.kind == OperatorKind.PROJECT:
            return self._execute_projection(node)
        if node.kind == OperatorKind.JOIN:
            if normalized == "RIGHT_DELIM_JOIN":
                return self._execute_delim_join(node)
            return self._execute_join(node)
        if node.kind == OperatorKind.AGGREGATE:
            return self._execute_aggregate(node)
        if node.kind == OperatorKind.SORT:
            return self._execute_sort(node)
        if node.kind == OperatorKind.LIMIT or normalized == "LIMIT":
            return self._execute_limit(node)
        if node.kind == OperatorKind.DELIM:
            return self._execute_delim_scan(node)
        if node.kind == OperatorKind.CTE:
            return self._execute_cte(node)
        if node.kind == OperatorKind.SET:
            return self._execute_set(node)
        if node.kind == OperatorKind.WINDOW:
            return self._execute_window(node)
        if normalized == "EMPTY_RESULT":
            return PhysicalTable("empty", {}, (), 0)
        raise UnsupportedPlanError(f"unsupported DuckDB physical node: {node.name}")

    def _execute_scan(self, node: TQPOperatorNode) -> PhysicalTable:
        normalized = node.name.strip().upper()
        if normalized == "DUMMY_SCAN":
            one = torch.ones(1, dtype=torch.int64, device=self._device)
            return PhysicalTable("dummy", {_ROW_ID: PhysicalValue(one)}, (_ROW_ID,), 1)
        table_name = _metadata_string(node, "Table")
        if table_name is None:
            raise UnsupportedPlanError(f"scan node is missing table metadata: {node.name}")
        projected_columns = _metadata_list(node, "Projections")
        filters = _metadata_list(node, "Filters")
        if not projected_columns:
            projected_columns = _required_scan_columns(self._con, table_name, self._graph, self._parents, node.node_id)
        fetched_columns = tuple(
            dict.fromkeys((*projected_columns, *_filter_columns(self._con, table_name, filters)))
        )
        scan_range = self._scan_ranges.get(table_name.lower())
        if not fetched_columns:
            count, offset = scan_row_count(self._con, table_name, scan_range)
            row_id = torch.arange(offset, offset + count, dtype=torch.int64, device=self._device)
            return PhysicalTable(table_name, {_ROW_ID: PhysicalValue(row_id)}, (_ROW_ID,), count)
        table = fetch_physical_table(
            self._con,
            table_name,
            fetched_columns,
            tuple(projected_columns),
            self._device,
            scan_range=scan_range,
            chunk_size=self._scan_chunk_sizes.get(table_name.lower()),
        )
        return _apply_scan_filters(table, filters) if filters else table

    def _execute_filter(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        expression = _metadata_string(node, "Expression")
        if expression is None:
            raise UnsupportedPlanError("filter node is missing Expression metadata")
        return child.filter(self._filter_mask(evaluate_expression(child, expression)))

    def _execute_projection(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        expressions = normalize_projection_expressions(_metadata_list(node, "Projections"))
        if not expressions:
            return child
        expressions = resolve_alias_projections(self._select_aliases, child, expressions)
        parent_required = required_columns_from_parents(self._graph, self._parents, node.node_id, _metadata_list)
        items = []
        for index, expression in enumerate(expressions):
            value_expression = parent_bound_projection_expression(
                child,
                expression,
                len(expressions),
                parent_required,
            ) or projection_value_expression(self._select_aliases, child, expression)
            value = evaluate_expression(child, value_expression)
            if value.is_literal:
                value = _materialize_literal(value, child)
            name, aliases = projection_output_name(child, expression, index, value, self._select_aliases)
            if _is_scalar_subquery_guard_projection(expression):
                aliases = tuple(dict.fromkeys((*aliases, "SUBQUERY")))
            items.append((name, value, aliases))
        return PhysicalTable.projected("projection", items, child.row_count)

    def _execute_join(self, node: TQPOperatorNode) -> PhysicalTable:
        if len(node.children) != 2:
            raise UnsupportedPlanError("physical hash join expects two children")
        left = self._execute_node(node.children[0])
        if (_metadata_string(node, "Join Type") or "").upper() == "MARK":
            right_node = self._graph.node_by_id(node.children[1])
            if right_node.name.strip().upper() == "COLUMN_DATA_SCAN":
                return execute_literal_mark_join(node, left, self._graph.source_sql)
            return execute_mark_join(node, left, self._execute_node(node.children[1]))
        right = self._execute_node(node.children[1])
        if node.name.strip().upper() == "NESTED_LOOP_JOIN":
            scalar_join = try_execute_scalar_nested_loop_join(left, right, _metadata_string(node, "Conditions") or "")
            if scalar_join is not None:
                return scalar_join
        return _execute_join_node(
            node,
            left,
            right,
            self._graph.source_sql,
            required_columns_from_parents(self._graph, self._parents, node.node_id, _metadata_list),
        )

    def _execute_delim_join(self, node: TQPOperatorNode) -> PhysicalTable:
        if len(node.children) != 3:
            raise UnsupportedPlanError("RIGHT_DELIM_JOIN expects three children")
        outer = self._execute_node(node.children[0])
        delim_index = _metadata_string(node, "Delim Index")
        if delim_index is not None:
            self._delim_tables[delim_index] = build_delim_table(outer, join_conditions(node))
        subquery = self._execute_node(node.children[1])
        return execute_delim_join_result(
            node,
            outer,
            subquery,
            self._graph.source_sql,
            required_columns_from_parents(self._graph, self._parents, node.node_id, _metadata_list),
        )

    def _execute_delim_scan(self, node: TQPOperatorNode) -> PhysicalTable:
        delim_index = _metadata_string(node, "Delim Index")
        if delim_index is None or delim_index not in self._delim_tables:
            raise UnsupportedPlanError(f"DELIM_SCAN has no materialized delimiter table: {delim_index}")
        return self._delim_tables[delim_index]

    def _execute_cte(self, node: TQPOperatorNode) -> PhysicalTable:
        normalized = node.name.strip().upper()
        if normalized == "CTE_SCAN":
            cte_index = _metadata_string(node, "CTE Index")
            if cte_index is None or cte_index not in self._cte_tables:
                raise UnsupportedPlanError(f"CTE_SCAN has no materialized table: {cte_index}")
            return self._cte_tables[cte_index]
        if normalized == "CTE":
            if len(node.children) != 2:
                raise UnsupportedPlanError("CTE expects materializer and consumer children")
            cte_index = _metadata_string(node, "Table Index")
            if cte_index is None:
                raise UnsupportedPlanError("CTE node is missing Table Index metadata")
            self._cte_tables[cte_index] = self._execute_node(node.children[0])
            return self._execute_node(node.children[1])
        raise UnsupportedPlanError(f"unsupported DuckDB CTE node: {node.name}")

    def _execute_set(self, node: TQPOperatorNode) -> PhysicalTable:
        if node.name.strip().upper() != "UNION":
            raise UnsupportedPlanError(f"unsupported set-operation node: {node.name}")
        return execute_union_node(tuple(self._execute_node(child) for child in node.children))

    def _execute_window(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        return execute_window_node(child, _metadata_list(node, "Projections"))

    def _execute_aggregate(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        specs = aggregate_specs(node, child)
        group_exprs = _metadata_list(node, "Groups")
        if group_exprs:
            return execute_grouped_aggregate(child, group_exprs, specs)
        return execute_ungrouped_aggregate(child, specs)

    def _execute_sort(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        order_items = _metadata_list(node, "Order By")
        if not order_items:
            return child
        return self._sort_table(child, order_items)

    def _execute_limit(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        top = _metadata_string(node, "Top") or _metadata_string(node, "Limit")
        if top is None:
            return child
        limit = int(top)
        order_items = _metadata_list(node, "Order By")
        table = self._limit_ordered_table(child, order_items, limit) if order_items else child
        indices = torch.arange(min(limit, table.row_count), dtype=torch.int64, device=table_device(table))
        return table.gather(indices)

    def _limit_ordered_table(
        self,
        table: PhysicalTable,
        order_items: Sequence[str],
        limit: int,
    ) -> PhysicalTable:
        topk_table = self._try_topk_limit(table, order_items, limit)
        if topk_table is not None:
            return topk_table
        return self._sort_table(table, order_items)

    def _try_topk_limit(
        self,
        table: PhysicalTable,
        order_items: Sequence[str],
        limit: int,
    ) -> PhysicalTable | None:
        if len(order_items) != 1 or limit <= 0 or limit >= table.row_count:
            return None
        expr, descending = strip_order_direction(order_items[0])
        key_name = expression_sort_key_name(expr)
        key = self._sort_value(table, expr, key_name).require_tensor()
        if _has_duplicate_values(key):
            return None
        _, indices = torch.topk(key, k=limit, largest=descending, sorted=True)
        return table.gather(indices)

    def _sort_table(self, table: PhysicalTable, order_items: Sequence[str]) -> PhysicalTable:
        result = table
        single_order = len(order_items) == 1
        for raw_item in reversed(tuple(order_items)):
            expr, descending = strip_order_direction(raw_item)
            key_name = expression_sort_key_name(expr)
            key_value = self._sort_value(result, expr, key_name)
            key = key_value.require_tensor()
            key_was_unique = key_value.unique
            order = torch.argsort(key, descending=descending, stable=True)
            result = result.gather(order)
            if single_order and not descending:
                result = _mark_value_metadata(
                    result,
                    key_name,
                    sorted_non_decreasing=True,
                    unique=key_was_unique,
                )
        return result

    def _sort_value(self, table: PhysicalTable, expression: str, key_name: str) -> PhysicalValue:
        try:
            return table.value_named(key_name)
        except KeyError:
            alias = aggregate_order_alias(self._select_aliases, table, expression)
            if alias is not None:
                return table.value_named(alias)
            expression_alias = matching_expression_alias(table, expression)
            if expression_alias is not None:
                return table.value_named(expression_alias)
            alias_value = order_alias_value(self._select_aliases, table, key_name)
            if alias_value is not None:
                return evaluate_expression(table, alias_value)
            return evaluate_expression(table, expression)

    def _single_child(self, node: TQPOperatorNode) -> PhysicalTable:
        if len(node.children) != 1:
            raise UnsupportedPlanError(f"{node.name} expects one child")
        return self._execute_node(node.children[0])

    @staticmethod
    def _filter_mask(value: PhysicalValue) -> torch.Tensor:
        mask = value.require_tensor()
        if value.valid is None:
            return mask
        return mask & value.valid.to(device=mask.device)


def execute_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Execute a supported DuckDB JSON physical graph with PyTorch tensors."""

    return PhysicalPlanExecutor(con, graph, device=device).execute()


def _has_duplicate_values(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return False
    sorted_values = torch.sort(values).values
    return bool(torch.any(sorted_values[1:] == sorted_values[:-1]).cpu().item())


def _mark_value_metadata(
    table: PhysicalTable,
    name: str,
    *,
    sorted_non_decreasing: bool,
    unique: bool,
) -> PhysicalTable:
    try:
        value = table.value_named(name)
    except KeyError:
        return table
    replacement = value.with_metadata(sorted_non_decreasing=sorted_non_decreasing, unique=unique)
    columns = {
        column: replacement if candidate is value else candidate
        for column, candidate in table.columns.items()
    }
    return PhysicalTable(table.name, columns, table.order, table.row_count)


def _apply_scan_filters(table: PhysicalTable, filters: Sequence[str]) -> PhysicalTable:
    result = table
    for raw_filter in filters:
        filter_text = raw_filter.strip()
        if filter_text.lower().startswith("optional:"):
            continue
        result = result.filter(PhysicalPlanExecutor._filter_mask(evaluate_expression(result, filter_text)))
    return result


def _materialize_literal(value: PhysicalValue, table: PhysicalTable) -> PhysicalValue:
    literal = value.literal
    device = table_device(table)
    if isinstance(literal, bool):
        tensor = torch.full((table.row_count,), literal, dtype=torch.bool, device=device)
    elif isinstance(literal, int):
        tensor = torch.full((table.row_count,), literal, dtype=torch.int64, device=device)
    elif isinstance(literal, float):
        tensor = torch.full((table.row_count,), literal, dtype=torch.float64, device=device)
    else:
        raise UnsupportedPlanError(f"cannot project non-numeric literal: {literal!r}")
    return PhysicalValue(tensor)


def _rename_for_output(table: PhysicalTable, aliases: Sequence[str]) -> PhysicalTable:
    if len(aliases) != len(table.order):
        return table
    items = [(alias, table.columns[name], (name, alias)) for alias, name in zip(aliases, table.order)]
    return PhysicalTable.projected("output", items, table.row_count)


def _trim_to_output_arity(table: PhysicalTable, output_arity: int) -> PhysicalTable:
    if output_arity >= len(table.order):
        return table
    items = [(name, table.columns[name], (name,)) for name in table.order[:output_arity]]
    return PhysicalTable.projected(table.name, items, table.row_count)


def _rows_from_table(table: PhysicalTable) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(table.row_count):
        row = {name: table.columns[name].cell(row_index) for name in table.order}
        rows.append(row)
    return rows


def _describe_aliases(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(str(row[0]) for row in rows)


def _output_aliases(con: duckdb.DuckDBPyConnection, graph: TQPOperatorGraph) -> tuple[str, ...]:
    return graph.output_names or _describe_aliases(con, graph.source_sql)


def _filter_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    filters: Sequence[str],
) -> tuple[str, ...]:
    if not filters:
        return ()
    available = _table_columns(con, table_name)
    referenced: list[str] = []
    for raw_filter in filters:
        if raw_filter.lower().startswith("optional:"):
            continue
        tokens = set(re.findall(r"[A-Za-z_][\w]*", raw_filter))
        referenced.extend(column for column in available if column in tokens)
    return tuple(dict.fromkeys(referenced))


def _required_scan_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    graph: TQPOperatorGraph,
    parents: dict[str, tuple[str, ...]],
    node_id: str,
) -> tuple[str, ...]:
    available = _table_columns(con, table_name)
    required = required_columns_from_parents(graph, parents, node_id, _metadata_list)
    tokens = set()
    for expression in required:
        tokens.update(re.findall(r"[A-Za-z_][\w]*", expression))
    return tuple(column for column in available if column in tokens)


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in con.execute(f"pragma table_info('{table_name}')").fetchall())


def _is_scalar_subquery_guard_projection(expression: str) -> bool:
    return "scalar subqueries can only return a single row" in expression
