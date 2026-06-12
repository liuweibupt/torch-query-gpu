"""DuckDB JSON physical-plan execution with PyTorch tensors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import duckdb
import torch

from tpch_torch.backend.physical_expr import (
    aggregate_output_aliases,
    evaluate_expression,
    expression_sort_key_name,
    projection_name,
    strip_order_direction,
)
from tpch_torch.backend.physical_join import inner_join_indices
from tpch_torch.backend.physical_sql import replace_aggregate_calls_with_refs, select_expressions_by_alias
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue, table_device
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.backend.generic import _encode_generic_column
from tpch_torch.relational import DATE_COLUMNS_EXTENDED

_ROW_ID = "__rowid__"


@dataclass(frozen=True)
class _AggregateSpec:
    function: str
    argument: str | None
    aliases: tuple[str, ...]


class PhysicalPlanExecutor:
    """Interpret supported DuckDB physical-plan nodes using tensor operators."""

    def __init__(self, con: duckdb.DuckDBPyConnection, graph: TQPOperatorGraph, device: str = "cpu"):
        self._con = con
        self._graph = graph
        self._device = device
        self._select_aliases = select_expressions_by_alias(graph.source_sql)

    def execute(self) -> list[dict[str, Any]]:
        table = self._execute_node(self._graph.root_id)
        aliases = _describe_aliases(self._con, self._graph.source_sql)
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
            return self._execute_join(node)
        if node.kind == OperatorKind.AGGREGATE:
            return self._execute_aggregate(node)
        if node.kind == OperatorKind.SORT:
            return self._execute_sort(node)
        if node.kind == OperatorKind.LIMIT or normalized == "LIMIT":
            return self._execute_limit(node)
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
        fetched_columns = tuple(
            dict.fromkeys((*projected_columns, *_filter_columns(self._con, table_name, filters)))
        )
        if not fetched_columns:
            count = int(self._con.execute(f"select count(*) from {table_name}").fetchone()[0])
            row_id = torch.arange(count, dtype=torch.int64, device=self._device)
            return PhysicalTable(table_name, {_ROW_ID: PhysicalValue(row_id)}, (_ROW_ID,), count)
        table = _fetch_physical_table(
            self._con,
            table_name,
            fetched_columns,
            tuple(projected_columns),
            self._device,
        )
        return _apply_scan_filters(table, filters) if filters else table

    def _execute_filter(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        expression = _metadata_string(node, "Expression")
        if expression is None:
            raise UnsupportedPlanError("filter node is missing Expression metadata")
        return child.filter(evaluate_expression(child, expression).require_tensor())

    def _execute_projection(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        expressions = _metadata_list(node, "Projections")
        if not expressions:
            return child
        expressions = _resolve_alias_projections(self._select_aliases, expressions)
        items = []
        for index, expression in enumerate(expressions):
            value = evaluate_expression(child, expression)
            if value.is_literal:
                value = _materialize_literal(value, child)
            name, aliases = projection_name(child, expression, index)
            items.append((name, value, aliases))
        return PhysicalTable.projected("projection", items, child.row_count)

    def _execute_join(self, node: TQPOperatorNode) -> PhysicalTable:
        join_type = (_metadata_string(node, "Join Type") or "").upper()
        if join_type != "INNER":
            raise UnsupportedPlanError(f"physical join type is not supported yet: {join_type}")
        if len(node.children) != 2:
            raise UnsupportedPlanError("physical hash join expects two children")
        left = self._execute_node(node.children[0])
        right = self._execute_node(node.children[1])
        left_expr, right_expr = _join_condition(node)
        left_key = evaluate_expression(left, left_expr).require_tensor()
        right_key = evaluate_expression(right, right_expr).require_tensor()
        left_rows, right_rows = _inner_join_indices(left_key, right_key)
        return _combine_join_tables(left, right, left_rows, right_rows, left_expr, right_expr)

    def _execute_aggregate(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        specs = _aggregate_specs(node, child)
        group_exprs = _metadata_list(node, "Groups")
        if group_exprs:
            return _execute_grouped_aggregate(child, group_exprs, specs)
        return _execute_ungrouped_aggregate(child, specs)

    def _execute_sort(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        order_items = _metadata_list(node, "Order By")
        if not order_items:
            return child
        return _sort_table(child, order_items)

    def _execute_limit(self, node: TQPOperatorNode) -> PhysicalTable:
        child = self._single_child(node)
        top = _metadata_string(node, "Top") or _metadata_string(node, "Limit")
        if top is None:
            return child
        limit = int(top)
        order_items = _metadata_list(node, "Order By")
        table = _sort_table(child, order_items) if order_items else child
        indices = torch.arange(min(limit, table.row_count), dtype=torch.int64, device=table_device(table))
        return table.gather(indices)

    def _single_child(self, node: TQPOperatorNode) -> PhysicalTable:
        if len(node.children) != 1:
            raise UnsupportedPlanError(f"{node.name} expects one child")
        return self._execute_node(node.children[0])


def execute_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Execute a supported DuckDB JSON physical graph with PyTorch tensors."""

    return PhysicalPlanExecutor(con, graph, device=device).execute()


def _fetch_physical_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    fetched_columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    device: str,
) -> PhysicalTable:
    select_list = ", ".join(_select_expression(column) for column in fetched_columns)
    columnar = con.execute(f"select {select_list} from {table_name}").fetchnumpy()
    values: dict[str, PhysicalValue] = {}
    for column in fetched_columns:
        tensor, vocabulary = _encode_generic_column(
            columnar[column],
            device,
            column_name=column,
            table_name=table_name,
        )
        value = PhysicalValue(tensor=tensor, dictionary=vocabulary, is_date=column in DATE_COLUMNS_EXTENDED)
        values[column] = value
        values[f"{table_name}.{column}"] = value
    row_count = 0 if not fetched_columns else int(next(iter(values.values())).require_tensor().numel())
    order = order_columns or (_ROW_ID,)
    if not order_columns:
        values[_ROW_ID] = PhysicalValue(torch.arange(row_count, dtype=torch.int64, device=device))
    return PhysicalTable(table_name, values, order, row_count)


def _apply_scan_filters(table: PhysicalTable, filters: Sequence[str]) -> PhysicalTable:
    result = table
    for raw_filter in filters:
        filter_text = raw_filter.strip()
        if filter_text.lower().startswith("optional:"):
            continue
        result = result.filter(evaluate_expression(result, filter_text).require_tensor())
    return result


def _execute_grouped_aggregate(
    child: PhysicalTable,
    group_exprs: Sequence[str],
    specs: Sequence[_AggregateSpec],
) -> PhysicalTable:
    key_values = [evaluate_expression(child, expression) for expression in group_exprs]
    key_tensors = [value.require_tensor().to(dtype=torch.int64) for value in key_values]
    stacked = torch.stack(key_tensors, dim=1)
    unique_keys, inverse = torch.unique(stacked, dim=0, sorted=True, return_inverse=True)
    row_count = int(unique_keys.shape[0])
    items: list[tuple[str, PhysicalValue, Sequence[str]]] = []
    for index, (expression, value) in enumerate(zip(group_exprs, key_values)):
        name, aliases = projection_name(child, expression, index)
        key_tensor = unique_keys[:, index].to(dtype=value.require_tensor().dtype)
        items.append((name, PhysicalValue(key_tensor, value.dictionary, value.is_date), aliases))
    for spec in specs:
        value = _evaluate_group_aggregate(child, inverse, row_count, spec)
        items.append((spec.aliases[0], value, spec.aliases))
    return PhysicalTable.projected("aggregate", items, row_count)


def _execute_ungrouped_aggregate(child: PhysicalTable, specs: Sequence[_AggregateSpec]) -> PhysicalTable:
    items = [(spec.aliases[0], _evaluate_scalar_aggregate(child, spec), spec.aliases) for spec in specs]
    return PhysicalTable.projected("aggregate", items, 1)


def _evaluate_group_aggregate(
    child: PhysicalTable,
    group_ids: torch.Tensor,
    group_count: int,
    spec: _AggregateSpec,
) -> PhysicalValue:
    if spec.function == "count_star":
        ones = torch.ones(group_ids.numel(), dtype=torch.int64, device=group_ids.device)
        return PhysicalValue(_scatter_sum(ones, group_ids, group_count))
    values = _aggregate_argument(child, spec).require_tensor()
    if spec.function == "sum":
        return PhysicalValue(_scatter_sum(values, group_ids, group_count))
    if spec.function == "min":
        return PhysicalValue(_scatter_reduce(values, group_ids, group_count, "amin"))
    if spec.function == "max":
        return PhysicalValue(_scatter_reduce(values, group_ids, group_count, "amax"))
    if spec.function == "avg":
        sums = _scatter_sum(values.to(dtype=torch.float64), group_ids, group_count)
        counts = _scatter_sum(torch.ones_like(values, dtype=torch.float64), group_ids, group_count)
        return PhysicalValue(sums / counts)
    raise UnsupportedPlanError(f"unsupported grouped aggregate: {spec.function}")


def _evaluate_scalar_aggregate(child: PhysicalTable, spec: _AggregateSpec) -> PhysicalValue:
    if spec.function == "count_star":
        tensor = torch.tensor([child.row_count], dtype=torch.int64, device=table_device(child))
        return PhysicalValue(tensor)
    values = _aggregate_argument(child, spec).require_tensor()
    if values.numel() == 0:
        tensor = torch.tensor([float("nan")], dtype=torch.float64, device=table_device(child))
    elif spec.function == "sum":
        tensor = values.sum().reshape(1)
    elif spec.function == "min":
        tensor = values.min().reshape(1)
    elif spec.function == "max":
        tensor = values.max().reshape(1)
    elif spec.function == "avg":
        tensor = values.to(dtype=torch.float64).mean().reshape(1)
    else:
        raise UnsupportedPlanError(f"unsupported scalar aggregate: {spec.function}")
    return PhysicalValue(tensor)


def _aggregate_argument(child: PhysicalTable, spec: _AggregateSpec) -> PhysicalValue:
    if spec.argument is None:
        raise UnsupportedPlanError(f"aggregate requires an argument: {spec.function}")
    return evaluate_expression(child, spec.argument)


def _sort_table(table: PhysicalTable, order_items: Sequence[str]) -> PhysicalTable:
    result = table
    for raw_item in reversed(tuple(order_items)):
        expr, descending = strip_order_direction(raw_item)
        key_name = expression_sort_key_name(expr)
        key = _sort_value(result, expr, key_name).require_tensor()
        order = torch.argsort(key, descending=descending, stable=True)
        result = result.gather(order)
    return result


def _sort_value(table: PhysicalTable, expression: str, key_name: str) -> PhysicalValue:
    try:
        return table.value_named(key_name)
    except KeyError:
        return evaluate_expression(table, expression)


def _resolve_alias_projections(
    select_aliases: dict[str, str],
    expressions: Sequence[str],
) -> tuple[str, ...]:
    resolved = []
    for expression in expressions:
        source = select_aliases.get(expression)
        resolved.append(
            replace_aggregate_calls_with_refs(source)
            if source is not None
            else expression
        )
    return tuple(resolved)


def _inner_join_indices(left_key: torch.Tensor, right_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return inner_join_indices(left_key, right_key)


def _combine_join_tables(
    left: PhysicalTable,
    right: PhysicalTable,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
    left_key: str,
    right_key: str,
) -> PhysicalTable:
    items: list[tuple[str, PhysicalValue, Sequence[str]]] = []
    for name in left.order:
        if _same_column(name, left_key):
            continue
        value = left.columns[name].gather(left_rows)
        items.append((name, value, _join_aliases(left, name, left_key, right_key)))
    for name in right.order:
        if _same_column(name, right_key):
            continue
        value = right.columns[name].gather(right_rows)
        items.append((name, value, _join_aliases(right, name, right_key, left_key)))
    return PhysicalTable.projected("join", items, int(left_rows.numel()))


def _aggregate_specs(node: TQPOperatorNode, child: PhysicalTable) -> tuple[_AggregateSpec, ...]:
    specs = []
    for raw in _metadata_list(node, "Aggregates"):
        if raw.lower() == "count_star()":
            specs.append(_AggregateSpec("count_star", None, ("count_star()", "count(*)")))
            continue
        match = re.fullmatch(r"(sum|avg|min|max|count)\((.*)\)", raw.strip(), re.I)
        if match is None:
            raise UnsupportedPlanError(f"unsupported aggregate expression: {raw}")
        function = match.group(1).lower()
        argument = match.group(2).strip()
        child_name = _child_name(child, argument)
        specs.append(_AggregateSpec(function, argument, aggregate_output_aliases(function, argument, child_name)))
    return tuple(specs)


def _child_name(child: PhysicalTable, argument: str) -> str | None:
    stripped = argument.strip()
    if not stripped.startswith("#"):
        return stripped
    index = int(stripped[1:])
    if index >= len(child.order):
        return None
    return child.order[index]


def _join_condition(node: TQPOperatorNode) -> tuple[str, str]:
    conditions = _metadata_list(node, "Conditions")
    if len(conditions) != 1 or "=" not in conditions[0]:
        raise UnsupportedPlanError(f"unsupported join condition: {conditions}")
    left, right = conditions[0].split("=", 1)
    return left.strip(), right.strip()


def _join_aliases(table: PhysicalTable, column: str, own_key: str, other_key: str) -> tuple[str, ...]:
    aliases = [column, f"{table.name}.{column}"]
    if column == own_key:
        aliases.append(other_key)
    return tuple(dict.fromkeys(aliases))


def _same_column(left: str, right: str) -> bool:
    return left == right or left.rsplit(".", 1)[-1] == right.rsplit(".", 1)[-1]


def _scatter_sum(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    return result.index_add(0, group_ids.to(dtype=torch.int64), values)


def _scatter_reduce(values: torch.Tensor, group_ids: torch.Tensor, group_count: int, reduce: str) -> torch.Tensor:
    fill_value = float("inf") if reduce == "amin" else float("-inf")
    result = torch.full((group_count,), fill_value, dtype=values.dtype, device=values.device)
    return result.scatter_reduce(0, group_ids.to(dtype=torch.int64), values, reduce=reduce, include_self=True)


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


def _rows_from_table(table: PhysicalTable) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(table.row_count):
        row = {name: table.columns[name].cell(row_index) for name in table.order}
        rows.append(row)
    return rows


def _describe_aliases(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(str(row[0]) for row in rows)


def _select_expression(column: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    return column


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


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in con.execute(f"pragma table_info('{table_name}')").fetchall())


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
