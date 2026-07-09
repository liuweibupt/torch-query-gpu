"""Tensor join kernels for DuckDB physical-plan execution."""

from __future__ import annotations

import re
from typing import Sequence

import torch

from tpch_torch.backend.physical_aliases import qualified_aliases_for_join_side
from tpch_torch.backend.physical_decimal_expr import decimal_comparison_tensors
from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_join_aliases import (
    existing_aliases as _existing_aliases,
    has_positional_reference as _has_positional_reference,
    matches_any_key as _matches_any_key,
    same_column as _same_column,
    unqualified as _unqualified,
)
from tpch_torch.backend.physical_key_ops import (
    comparable_key_tensors,
    comparable_value_tensors,
    pairwise_equal_values,
)
from tpch_torch.backend.physical_membership import membership_join_indices
from tpch_torch.backend.physical_projection import matching_aggregate_alias
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.backend.physical_sql_regions import output_requires_column
from tpch_torch.errors import UnsupportedPlanError


def inner_join_indices(left_key: torch.Tensor, right_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching row indices for an inner equi-join without host key materialization."""

    _validate_join_keys(left_key, right_key)
    device = left_key.device
    if left_key.numel() == 0 or right_key.numel() == 0:
        return _empty_indices(device)

    left_values, right_values = comparable_key_tensors(left_key, right_key)
    right_order, sorted_right_values = _sorted_build_keys(right_values)
    starts = torch.searchsorted(sorted_right_values, left_values, right=False)
    ends = torch.searchsorted(sorted_right_values, left_values, right=True)
    match_counts = ends - starts
    if _is_strictly_increasing(sorted_right_values):
        return _unique_build_join_indices(starts, match_counts, right_order)
    match_count = int(match_counts.sum().cpu().item())
    if match_count == 0:
        return _empty_indices(device)

    left_rows = torch.repeat_interleave(
        torch.arange(left_values.numel(), dtype=torch.int64, device=device),
        match_counts,
    )
    right_positions = _matching_sorted_positions(starts, match_counts, match_count)
    right_rows = right_positions if right_order is None else right_order[right_positions]
    return left_rows, right_rows


def inner_join_indices_for_values(
    left_value: PhysicalValue,
    right_value: PhysicalValue,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inner equi-join indices while honoring known build-key metadata."""

    left_key, right_key = comparable_value_tensors(left_value, right_value)
    if right_value.sorted_non_decreasing and right_value.unique:
        return _sorted_unique_build_join_indices(left_key, right_key)
    return inner_join_indices(left_key, right_key)


def join_indices_for_conditions(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching row indices for one or more equi-join conditions."""

    if len(conditions) == 1:
        left_expr, right_expr = conditions[0]
        return inner_join_indices_for_values(
            evaluate_expression(left, left_expr),
            evaluate_expression(right, right_expr),
        )
    return _join_indices_for_multiple_conditions(left, right, conditions)


def semi_join_indices(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> torch.Tensor:
    """Return left row indices that have at least one match on the right side."""

    return membership_join_indices(left, right, conditions, matched=True)


def anti_join_indices(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> torch.Tensor:
    """Return left row indices that have no match on the right side."""

    return membership_join_indices(left, right, conditions, matched=False)


def semi_join_table(
    table: PhysicalTable,
    rows: torch.Tensor,
    keys: Sequence[str],
    source_sql: str,
    required_columns: Sequence[str],
) -> PhysicalTable:
    """Return SEMI/ANTI preserved rows with DuckDB-style key pruning."""

    items = []
    for name in table.order:
        if _drop_semi_key(table, name, keys, source_sql, required_columns):
            continue
        value = table.columns[name]
        items.append((name, value.gather(rows), _existing_aliases(table, value)))
    return PhysicalTable.projected("semi_join", items, int(rows.numel()))


def outer_join_indices(
    preserved_row_count: int,
    preserved_rows: torch.Tensor,
    optional_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append unmatched preserved rows and mark optional-side validity."""

    device = preserved_rows.device
    matched = _matched_preserved_mask(preserved_row_count, preserved_rows, device)
    unmatched_rows = torch.nonzero(~matched).flatten().to(dtype=torch.int64)
    if unmatched_rows.numel() == 0:
        return preserved_rows, optional_rows, torch.ones_like(preserved_rows, dtype=torch.bool)
    combined_preserved = torch.cat((preserved_rows, unmatched_rows))
    filler_optional = torch.zeros_like(unmatched_rows)
    combined_optional = torch.cat((optional_rows, filler_optional))
    optional_valid = torch.cat(
        (
            torch.ones_like(preserved_rows, dtype=torch.bool),
            torch.zeros_like(unmatched_rows, dtype=torch.bool),
        )
    )
    return combined_preserved, combined_optional, optional_valid


def try_execute_scalar_nested_loop_join(
    left: PhysicalTable,
    right: PhysicalTable,
    condition: str,
) -> PhysicalTable | None:
    """Execute DuckDB scalar-subquery nested-loop joins as a left-side filter."""

    comparison = _split_scalar_subquery_condition(condition)
    if comparison is None:
        return None
    left_expr, operator = comparison
    if right.row_count != 1 or len(right.order) != 1:
        raise UnsupportedPlanError("scalar SUBQUERY join expects one row and one column")
    left_value = _scalar_left_value(left, left_expr)
    right_value = _single_row_value(right.value_at(0))
    return left.filter(_compare_scalar_values(left_value, operator, right_value))


def _scalar_left_value(table: PhysicalTable, expression: str) -> PhysicalValue:
    alias = matching_aggregate_alias(table, expression)
    if alias is not None:
        return table.value_named(alias)
    return evaluate_expression(table, expression)


def _unique_build_join_indices(
    starts: torch.Tensor,
    match_counts: torch.Tensor,
    right_order: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched = match_counts > 0
    left_rows = torch.nonzero(matched).flatten().to(dtype=torch.int64)
    right_positions = starts[matched].to(dtype=torch.int64)
    right_rows = right_positions if right_order is None else right_order[right_positions]
    return left_rows, right_rows


def _sorted_unique_build_join_indices(
    left_key: torch.Tensor,
    right_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_join_keys(left_key, right_key)
    device = left_key.device
    if left_key.numel() == 0 or right_key.numel() == 0:
        return _empty_indices(device)

    left_values, right_values = comparable_key_tensors(left_key, right_key)
    right_values = right_values.contiguous()
    positions = torch.searchsorted(right_values, left_values, right=False).to(dtype=torch.int64)
    in_bounds = positions < right_values.numel()
    safe_positions = torch.where(in_bounds, positions, torch.zeros_like(positions))
    matched = in_bounds & (right_values[safe_positions] == left_values)
    left_rows = torch.nonzero(matched).flatten().to(dtype=torch.int64)
    return left_rows, positions[matched]


def _sorted_build_keys(right_values: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    if _is_sorted_non_decreasing(right_values):
        return None, right_values.contiguous()
    right_order = torch.argsort(right_values, stable=True)
    return right_order, right_values[right_order].contiguous()


def _is_sorted_non_decreasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return not bool(torch.any(values[1:] < values[:-1]).cpu().item())


def _is_strictly_increasing(values: torch.Tensor) -> bool:
    if values.numel() <= 1:
        return True
    return bool(torch.all(values[1:] > values[:-1]).cpu().item())


def _matching_sorted_positions(
    starts: torch.Tensor,
    match_counts: torch.Tensor,
    match_count: int,
) -> torch.Tensor:
    segment_offsets = torch.cumsum(match_counts, dim=0) - match_counts
    repeated_starts = torch.repeat_interleave(starts, match_counts)
    repeated_offsets = torch.repeat_interleave(segment_offsets, match_counts)
    local_offsets = torch.arange(match_count, dtype=torch.int64, device=starts.device)
    return repeated_starts + local_offsets - repeated_offsets


def _validate_join_keys(left_key: torch.Tensor, right_key: torch.Tensor) -> None:
    if left_key.ndim != 1 or right_key.ndim != 1:
        raise ValueError("physical join keys must be 1-D tensors")
    if left_key.device != right_key.device:
        raise ValueError("physical join keys must be on the same device")


def _empty_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    return empty, empty.clone()


def combine_join_tables(
    left: PhysicalTable,
    right: PhysicalTable,
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    source_sql: str,
    required_columns: Sequence[str] = (),
    left_valid: torch.Tensor | None = None,
    right_valid: torch.Tensor | None = None,
    right_optional_order: str = "default",
) -> PhysicalTable:
    """Build the physical join output while preserving DuckDB's pruned column order."""

    items: list[tuple[str, PhysicalValue, Sequence[str]]] = []
    delayed: list[tuple[str, PhysicalValue, Sequence[str]]] = []
    equivalent_keys = left_valid is None and right_valid is None
    left_items = _join_side_items(
        left,
        left_rows,
        left_keys,
        right_keys,
        source_sql,
        required_columns,
        left_valid,
        equivalent_keys,
    )
    right_items = _join_side_items(
        right,
        right_rows,
        right_keys,
        left_keys,
        source_sql,
        required_columns,
        right_valid,
        equivalent_keys,
    )
    if left_valid is None and right_optional_order == "outer_first":
        items.extend(left_items[0])
        items.extend(left_items[1])
        items.extend(right_items[0])
    else:
        items.extend(left_items[0])
        items.extend(right_items[0])
    if left_valid is None and right_optional_order == "outer_first":
        delayed.extend(right_items[1])
    elif left_valid is None:
        delayed.extend(left_items[1])
        delayed.extend(right_items[1])
    else:
        delayed.extend(right_items[1])
        delayed.extend(left_items[1])
    items.extend(delayed)
    table = PhysicalTable.projected("join", items, int(left_rows.numel()))
    if left_valid is None and right_valid is None:
        return _refresh_inner_join_key_aliases(table, left_keys, right_keys)
    return table


def right_join_has_no_unmatched_rows(right_rows: torch.Tensor, right_row_count: int) -> bool:
    """Return whether an inner match vector covers every preserved right row."""

    if right_row_count == 0:
        return True
    if right_rows.numel() == 0:
        return False
    counts = torch.bincount(right_rows.to(dtype=torch.int64), minlength=right_row_count)
    return bool(torch.all(counts[:right_row_count] > 0).cpu().item())


def _matched_preserved_mask(row_count: int, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    matched = torch.zeros(row_count, dtype=torch.bool, device=device)
    if rows.numel() > 0:
        matched[rows.to(dtype=torch.int64)] = True
    return matched


def _split_scalar_subquery_condition(condition: str) -> tuple[str, str] | None:
    for operator in (">=", "<=", "!=", "<>", ">", "<"):
        suffix = f" {operator} SUBQUERY"
        if condition.endswith(suffix):
            return condition[: -len(suffix)].strip(), operator
    return None


def _single_row_value(value: PhysicalValue) -> PhysicalValue:
    tensor = value.require_tensor()
    index = torch.zeros(1, dtype=torch.int64, device=tensor.device)
    return value.gather(index)


def _compare_scalar_values(
    left_value: PhysicalValue,
    operator: str,
    right_value: PhysicalValue,
) -> torch.Tensor:
    decimal_tensors = decimal_comparison_tensors(left_value, right_value)
    left_tensor, right_tensor = decimal_tensors or comparable_key_tensors(
        left_value.require_tensor(),
        right_value.require_tensor(),
    )
    compared = _compare_with_scalar(left_tensor, operator, right_tensor)
    return _apply_scalar_compare_validity(compared, left_value, right_value)


def _apply_scalar_compare_validity(
    compared: torch.Tensor,
    left_value: PhysicalValue,
    right_value: PhysicalValue,
) -> torch.Tensor:
    if left_value.valid is not None:
        compared = compared & left_value.valid.to(device=compared.device)
    if right_value.valid is not None:
        compared = compared & right_value.valid.to(device=compared.device)
    return compared


def _compare_with_scalar(values: torch.Tensor, operator: str, scalar: torch.Tensor) -> torch.Tensor:
    if operator == ">":
        return values > scalar
    if operator == ">=":
        return values >= scalar
    if operator == "<":
        return values < scalar
    if operator == "<=":
        return values <= scalar
    if operator in {"!=", "<>"}:
        return values != scalar
    raise UnsupportedPlanError(f"unsupported scalar SUBQUERY comparison: {operator}")


def _join_side_items(
    table: PhysicalTable,
    rows: torch.Tensor,
    own_keys: Sequence[str],
    other_keys: Sequence[str],
    source_sql: str,
    required_columns: Sequence[str],
    valid: torch.Tensor | None = None,
    equivalent_keys: bool = True,
) -> tuple[list[tuple[str, PhysicalValue, Sequence[str]]], list[tuple[str, PhysicalValue, Sequence[str]]]]:
    items = []
    delayed = []
    needs_internal_keys = _has_positional_reference(required_columns)
    for name in table.order:
        output_key = output_requires_column(source_sql, table, name)
        keep_key = output_key or needs_internal_keys or _matches_any_key(name, required_columns)
        if _matches_any_key(name, own_keys) and not keep_key:
            continue
        source_value = table.columns[name]
        value = source_value.gather(rows) if valid is None else source_value.gather_optional(rows, valid)
        aliases = (
            *_existing_aliases(table, source_value),
            *_join_aliases(table, name, own_keys, other_keys, source_sql, equivalent_keys),
        )
        item = (name, value, aliases)
        if _matches_any_key(name, own_keys) and not output_key:
            delayed.append(item)
        else:
            items.append(item)
    return items, delayed


def _drop_semi_key(
    table: PhysicalTable,
    name: str,
    keys: Sequence[str],
    source_sql: str,
    required_columns: Sequence[str],
) -> bool:
    if not _matches_any_key(name, keys):
        return False
    if output_requires_column(source_sql, table, name):
        return False
    return not _matches_any_key(name, required_columns)


def _refresh_inner_join_key_aliases(
    table: PhysicalTable,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> PhysicalTable:
    columns = dict(table.columns)
    aliases = dict(getattr(table, "aliases", {}) or {})
    for left_key, right_key in zip(left_keys, right_keys):
        _assign_equivalent_key_alias(columns, aliases, table, left_key, right_key)
        _assign_equivalent_key_alias(columns, aliases, table, right_key, left_key)
    return PhysicalTable(table.name, columns, table.order, table.row_count, table.batch, aliases)


def _assign_equivalent_key_alias(
    columns: dict[str, PhysicalValue],
    aliases: dict[str, str],
    table: PhysicalTable,
    source_key: str,
    alias_key: str,
) -> None:
    source_name = _matching_order_name(table, source_key)
    if source_name is None:
        return
    alias_name = _unqualified(alias_key)
    columns[alias_name] = table.columns[source_name]
    aliases.pop(alias_name, None)


def _matching_order_name(table: PhysicalTable, key: str) -> str | None:
    for name in table.order:
        if _same_column(name, key):
            return name
    return None


def _join_indices_for_multiple_conditions(
    left: PhysicalTable,
    right: PhysicalTable,
    conditions: Sequence[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_left, first_right = conditions[0]
    left_rows, right_rows = inner_join_indices_for_values(
        evaluate_expression(left, first_left),
        evaluate_expression(right, first_right),
    )
    for left_expr, right_expr in conditions[1:]:
        if left_rows.numel() == 0:
            return left_rows, right_rows
        matched = pairwise_equal_values(
            evaluate_expression(left, left_expr),
            evaluate_expression(right, right_expr),
            left_rows,
            right_rows,
        )
        left_rows = left_rows[matched]
        right_rows = right_rows[matched]
    return left_rows, right_rows


def _join_aliases(
    table: PhysicalTable,
    column: str,
    own_keys: Sequence[str],
    other_keys: Sequence[str],
    source_sql: str = "",
    equivalent_keys: bool = True,
) -> tuple[str, ...]:
    aliases = [column, f"{table.name}.{column}"]
    if equivalent_keys:
        aliases.extend(other for own, other in zip(own_keys, other_keys) if _same_column(column, own))
    aliases.extend(qualified_aliases_for_join_side(source_sql, table.name, column, own_keys, other_keys))
    return tuple(dict.fromkeys(aliases))
