"""MARK join execution for DuckDB IN / NOT IN physical plans."""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch

from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_join import join_indices_for_conditions
from tpch_torch.backend.physical_join_exec import join_conditions
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import TQPOperatorNode
from tpch_torch.operators import membership_mask


def execute_mark_join(node: TQPOperatorNode, left: PhysicalTable, right: PhysicalTable) -> PhysicalTable:
    """Append a boolean SUBQUERY marker for left rows that match right rows."""

    left_rows, _ = join_indices_for_conditions(left, right, join_conditions(node))
    marker = torch.zeros(left.row_count, dtype=torch.bool, device=_table_device(left))
    if left_rows.numel() > 0:
        marker[left_rows.to(dtype=torch.int64)] = True
    return _append_marker(left, marker)


def execute_literal_mark_join(node: TQPOperatorNode, left: PhysicalTable, source_sql: str) -> PhysicalTable:
    """Append a boolean SUBQUERY marker for a literal-list MARK join."""

    left_expr = _mark_left_expression(node)
    value = evaluate_expression(left, left_expr)
    literals = _matching_literal_list(source_sql, value)
    if literals is None:
        raise UnsupportedPlanError(f"could not recover MARK literal list for: {left_expr}")
    return _append_marker(left, _literal_membership(value, literals))


def _mark_left_expression(node: TQPOperatorNode) -> str:
    conditions = join_conditions(node)
    if len(conditions) != 1:
        raise UnsupportedPlanError("literal MARK join expects one condition")
    return conditions[0][0]


def _matching_literal_list(sql: str, value: PhysicalValue) -> tuple[Any, ...] | None:
    lists = _literal_lists(sql)
    if value.dictionary is not None:
        return next((items for items in lists if all(isinstance(item, str) for item in items)), None)
    return next((items for items in lists if all(isinstance(item, (int, float)) for item in items)), None)


def _literal_lists(sql: str) -> tuple[tuple[Any, ...], ...]:
    lists = []
    for match in re.finditer(r"\bIN\s*\(([^()]+)\)", sql, re.I):
        raw = match.group(1)
        if re.search(r"\bSELECT\b", raw, re.I):
            continue
        items = tuple(_parse_literal(item.strip()) for item in raw.split(","))
        if all(item is not _NO_LITERAL for item in items):
            lists.append(items)
    return tuple(lists)


def _parse_literal(raw: str) -> Any:
    if re.fullmatch(r"'[^']*'", raw):
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return _NO_LITERAL


def _literal_membership(value: PhysicalValue, literals: Sequence[Any]) -> torch.Tensor:
    tensor = value.require_tensor()
    if value.dictionary is None:
        return membership_mask(tensor, literals)
    accepted = [value.dictionary.index(str(item)) for item in literals if str(item) in value.dictionary]
    if not accepted:
        return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    return membership_mask(tensor, accepted)


def _append_marker(table: PhysicalTable, marker: torch.Tensor) -> PhysicalTable:
    columns = dict(table.columns)
    columns["SUBQUERY"] = PhysicalValue(marker)
    order = (*table.order, "SUBQUERY")
    return PhysicalTable("mark", columns, order, table.row_count)


def _table_device(table: PhysicalTable) -> torch.device:
    for value in table.columns.values():
        if value.tensor is not None:
            return value.tensor.device
    return torch.device("cpu")


_NO_LITERAL = object()
