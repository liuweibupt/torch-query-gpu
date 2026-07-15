"""Physical join node execution primitives."""

from __future__ import annotations

from typing import Sequence

import torch

from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_metadata import metadata_list as _metadata_list, metadata_string as _metadata_string
from tpch_torch.backend.physical_join import (
    anti_join_indices,
    combine_join_tables,
    join_indices_for_conditions,
    outer_join_indices,
    semi_join_indices,
    semi_join_table,
)
from tpch_torch.backend.physical_types import PhysicalTable
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import TQPOperatorNode


def execute_join_node(
    node: TQPOperatorNode,
    left: PhysicalTable,
    right: PhysicalTable,
    source_sql: str,
    required_columns: Sequence[str],
) -> PhysicalTable:
    """Execute one supported DuckDB physical join node with PyTorch tensors."""

    join_type = _join_type(node)
    conditions = join_conditions(node)
    wrapped = _execute_dummy_dependent_wrapper(join_type, left, right)
    if wrapped is not None:
        return wrapped
    if join_type == "SEMI":
        rows = semi_join_indices(left, right, conditions)
        return semi_join_table(left, rows, tuple(left for left, _ in conditions), source_sql, required_columns)
    if join_type == "ANTI":
        rows = anti_join_indices(left, right, conditions)
        return semi_join_table(left, rows, tuple(left for left, _ in conditions), source_sql, required_columns)
    if join_type == "RIGHT_SEMI":
        swapped = _swap_conditions(conditions)
        rows = semi_join_indices(right, left, swapped)
        return semi_join_table(right, rows, tuple(left for left, _ in swapped), source_sql, required_columns)
    if join_type == "RIGHT_ANTI":
        swapped = _swap_conditions(conditions)
        rows = anti_join_indices(right, left, swapped)
        return semi_join_table(right, rows, tuple(left for left, _ in swapped), source_sql, required_columns)
    left_rows, right_rows = join_indices_for_conditions(left, right, conditions)
    left_rows, right_rows = _apply_residual_conditions(left, right, left_rows, right_rows, residual_conditions(node))
    return _join_rows(join_type, left, right, left_rows, right_rows, conditions, source_sql, required_columns)


def join_conditions(node: TQPOperatorNode) -> tuple[tuple[str, str], ...]:
    conditions = _metadata_list(node, "Conditions")
    parsed = []
    for condition in conditions:
        if _is_residual_condition(condition):
            continue
        left, right = _split_join_equality(condition)
        parsed.append((left, right))
    if not parsed:
        raise UnsupportedPlanError(f"unsupported join condition: {conditions}")
    return tuple(parsed)


def residual_conditions(node: TQPOperatorNode) -> tuple[str, ...]:
    """Return non-equi predicates that should filter joined candidate rows."""

    return tuple(condition for condition in _metadata_list(node, "Conditions") if _is_residual_condition(condition))


def _join_type(node: TQPOperatorNode) -> str:
    join_type = (_metadata_string(node, "Join Type") or "").upper()
    supported = {"INNER", "LEFT", "RIGHT", "SEMI", "ANTI", "RIGHT_SEMI", "RIGHT_ANTI"}
    if join_type not in supported:
        raise UnsupportedPlanError(f"physical join type is not supported yet: {join_type}")
    return join_type


def _execute_dummy_dependent_wrapper(
    join_type: str,
    left: PhysicalTable,
    right: PhysicalTable,
) -> PhysicalTable | None:
    if join_type not in {"RIGHT", "RIGHT_SEMI", "RIGHT_ANTI"}:
        return None
    if _is_dummy_table(right):
        return left
    if _is_dummy_table(left):
        return right if join_type == "SEMI" else right.gather(_empty_indices(right))
    return None


def _is_dummy_table(table: PhysicalTable) -> bool:
    return table.name == "dummy" and table.order == ("__rowid__",)


def _empty_indices(table: PhysicalTable) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=_table_device(table))


def _table_device(table: PhysicalTable) -> torch.device:
    for value in table.columns.values():
        if value.tensor is not None:
            return value.tensor.device
    return torch.device("cpu")


def _join_rows(
    join_type: str,
    left: PhysicalTable,
    right: PhysicalTable,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
    conditions: Sequence[tuple[str, str]],
    source_sql: str,
    required_columns: Sequence[str],
) -> PhysicalTable:
    left_valid = None
    right_valid = None
    if join_type == "LEFT":
        left_rows, right_rows, right_valid = outer_join_indices(left.row_count, left_rows, right_rows)
    if join_type == "RIGHT":
        right_rows, left_rows, left_valid = outer_join_indices(right.row_count, right_rows, left_rows)
    return combine_join_tables(
        left,
        right,
        left_rows,
        right_rows,
        tuple(left_expr for left_expr, _ in conditions),
        tuple(right_expr for _, right_expr in conditions),
        source_sql,
        required_columns,
        left_valid,
        right_valid,
    )


def _swap_conditions(conditions: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((right, left) for left, right in conditions)


def _split_join_equality(condition: str) -> tuple[str, str]:
    if " IS NOT DISTINCT FROM " in condition:
        left, right = condition.split(" IS NOT DISTINCT FROM ", 1)
        return left.strip(), right.strip()
    if "=" not in condition:
        raise UnsupportedPlanError(f"unsupported join condition: {condition}")
    left, right = condition.split("=", 1)
    return _strip_not_distinct(left), _strip_not_distinct(right)


def _strip_not_distinct(expression: str) -> str:
    return expression.replace("IS NOT DISTINCT FROM", "").strip()


def _is_residual_condition(condition: str) -> bool:
    return "!=" in condition or "<>" in condition


def _apply_residual_conditions(
    left: PhysicalTable,
    right: PhysicalTable,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
    residuals: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not residuals or left_rows.numel() == 0:
        return left_rows, right_rows
    left_joined = left.gather(left_rows)
    right_joined = right.gather(right_rows)
    mask = torch.ones(left_rows.shape, dtype=torch.bool, device=left_rows.device)
    for condition in residuals:
        mask = mask & _evaluate_residual(left_joined, right_joined, condition)
    return left_rows[mask], right_rows[mask]


def _evaluate_residual(left: PhysicalTable, right: PhysicalTable, condition: str) -> torch.Tensor:
    for operator in ("!=", "<>"):
        if operator not in condition:
            continue
        left_expr, right_expr = condition.split(operator, 1)
        left_tensor = evaluate_expression(left, left_expr.strip()).require_tensor()
        right_tensor = evaluate_expression(right, right_expr.strip()).require_tensor()
        return left_tensor != right_tensor
    raise UnsupportedPlanError(f"unsupported join residual condition: {condition}")
