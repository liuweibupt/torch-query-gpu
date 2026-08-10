"""Window operator helpers for DuckDB physical WINDOW nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch

from tpch_torch.backend.physical_aggregate import group_key_mapping
from tpch_torch.backend.physical_expr import evaluate_expression, strip_order_direction
from tpch_torch.backend.physical_expr_parse import split_args as _split_args
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue, table_device
from tpch_torch.errors import UnsupportedPlanError

_RANKING_FUNCTIONS = {"row_number", "rank", "dense_rank", "rank_dense"}
_AGGREGATE_FUNCTIONS = {"sum", "count", "avg", "min", "max"}


@dataclass(frozen=True)
class WindowCall:
    function: str
    argument: str
    partitions: tuple[str, ...]
    order_by: tuple[str, ...]


def execute_window_node(child: PhysicalTable, projections: Sequence[str]) -> PhysicalTable:
    """Append supported DuckDB window projections to a tensor table."""

    items = [(name, child.columns[name], (name,)) for name in child.order]
    for projection in projections:
        call = _parse_window_call(projection)
        value = _evaluate_window_call(child, call)
        items.append((projection, value, (projection,)))
    return PhysicalTable.projected("window", items, child.row_count)


def _parse_window_call(projection: str) -> WindowCall:
    match = re.fullmatch(r'"?([A-Za-z_][\w]*)"?\s*\((.*)\)\s+OVER\s*\((.*)\)', projection.strip(), re.I | re.S)
    if match is None:
        raise UnsupportedPlanError(f"unsupported WINDOW projection: {projection}")
    function = match.group(1).lower()
    if function not in _RANKING_FUNCTIONS | _AGGREGATE_FUNCTIONS:
        raise UnsupportedPlanError(f"unsupported WINDOW function: {function}")
    partitions, order_by = _parse_window_spec(match.group(3).strip())
    argument = match.group(2).strip()
    if function in _RANKING_FUNCTIONS and argument:
        raise UnsupportedPlanError(f"{function} window function expects no arguments")
    return WindowCall("dense_rank" if function == "rank_dense" else function, argument, partitions, order_by)


def _parse_window_spec(spec: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not spec:
        return (), ()
    upper = spec.upper()
    if upper.startswith("ORDER BY "):
        return (), _split_args(spec[len("ORDER BY ") :])
    if not upper.startswith("PARTITION BY "):
        raise UnsupportedPlanError(f"unsupported WINDOW specification: {spec}")
    body = spec[len("PARTITION BY ") :]
    order_index = _find_top_level_order_by(body)
    if order_index < 0:
        return _split_args(body), ()
    partitions = _split_args(body[:order_index])
    order_by = _split_args(body[order_index + len(" ORDER BY ") :])
    return partitions, order_by


def _find_top_level_order_by(text: str) -> int:
    depth = 0
    in_quote = False
    upper = text.upper()
    for index, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0 and upper.startswith(" ORDER BY ", index):
            return index
    return -1


def _evaluate_window_call(table: PhysicalTable, call: WindowCall) -> PhysicalValue:
    if call.function in _RANKING_FUNCTIONS:
        return _evaluate_ranking_window(table, call)
    if call.order_by:
        raise UnsupportedPlanError("aggregate WINDOW with ORDER BY frame is not supported yet")
    return _evaluate_aggregate_window(table, call)


def _evaluate_ranking_window(table: PhysicalTable, call: WindowCall) -> PhysicalValue:
    order = _window_order(table, call.partitions, call.order_by)
    sorted_positions = torch.arange(table.row_count, dtype=torch.int64, device=table_device(table))
    partitions_changed = _changed_at_order(table, order, call.partitions)
    partition_ids = torch.cumsum(partitions_changed.to(dtype=torch.int64), dim=0) - 1
    partition_starts = torch.nonzero(partitions_changed).flatten().to(dtype=torch.int64)
    row_number = sorted_positions - partition_starts[partition_ids] + 1
    if call.function == "row_number":
        return PhysicalValue(_scatter_to_input_order(row_number, order))
    peer_changed = partitions_changed | _changed_at_order(table, order, call.order_by)
    peer_ids = torch.cumsum(peer_changed.to(dtype=torch.int64), dim=0) - 1
    peer_starts = torch.nonzero(peer_changed).flatten().to(dtype=torch.int64)
    if call.function == "rank":
        ranks = peer_starts[peer_ids] - partition_starts[partition_ids] + 1
    else:
        first_peer_in_partition = peer_ids[partition_starts[partition_ids]]
        ranks = peer_ids - first_peer_in_partition + 1
    return PhysicalValue(_scatter_to_input_order(ranks, order))


def _window_order(table: PhysicalTable, partitions: Sequence[str], order_by: Sequence[str]) -> torch.Tensor:
    order = torch.arange(table.row_count, dtype=torch.int64, device=table_device(table))
    for item in reversed(tuple(_partition_order_items(partitions)) + tuple(order_by)):
        expression, descending = strip_order_direction(item)
        value = evaluate_expression(table, expression)
        _reject_null_order_key(value, expression)
        key = value.require_tensor().index_select(0, order)
        order = order[torch.argsort(key, descending=descending, stable=True)]
    return order


def _partition_order_items(partitions: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{expression} ASC" for expression in partitions)


def _changed_at_order(table: PhysicalTable, order: torch.Tensor, expressions: Sequence[str]) -> torch.Tensor:
    changed = torch.zeros(order.numel(), dtype=torch.bool, device=order.device)
    if order.numel() == 0:
        return changed
    changed[0] = True
    for expression in expressions:
        value = evaluate_expression(table, strip_order_direction(expression)[0])
        _reject_null_order_key(value, expression)
        ordered = value.require_tensor().index_select(0, order)
        changed[1:] |= ordered[1:] != ordered[:-1]
    return changed


def _reject_null_order_key(value: PhysicalValue, expression: str) -> None:
    if value.valid is not None and not bool(torch.all(value.valid).cpu().item()):
        raise UnsupportedPlanError(f"WINDOW ordering with NULL key is not supported yet: {expression}")


def _reject_null_partition_keys(table: PhysicalTable, partitions: Sequence[str]) -> None:
    for expression in partitions:
        value = evaluate_expression(table, expression)
        if value.valid is not None and not bool(torch.all(value.valid).cpu().item()):
            raise UnsupportedPlanError(f"WINDOW partitioning with NULL key is not supported yet: {expression}")


def _scatter_to_input_order(values_in_window_order: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(values_in_window_order)
    result[order] = values_in_window_order
    return result


def _evaluate_aggregate_window(table: PhysicalTable, call: WindowCall) -> PhysicalValue:
    argument = _window_argument(table, call)
    if not call.partitions:
        return _broadcast_scalar_aggregate(table, call.function, argument)
    _reject_null_partition_keys(table, call.partitions)
    mapping = group_key_mapping(table, call.partitions)
    values = _aggregate_group_values(table, call.function, argument, mapping.inverse, int(mapping.unique_keys.shape[0]))
    return PhysicalValue(values.tensor.index_select(0, mapping.inverse), valid=_gather_valid(values, mapping.inverse), meta=values.meta)


def _window_argument(table: PhysicalTable, call: WindowCall) -> PhysicalValue | None:
    if call.function == "count" and call.argument in {"", "*"}:
        return None
    if call.argument in {"", "*"}:
        raise UnsupportedPlanError(f"WINDOW {call.function} requires an argument")
    return evaluate_expression(table, call.argument)


def _broadcast_scalar_aggregate(table: PhysicalTable, function: str, argument: PhysicalValue | None) -> PhysicalValue:
    scalar = _aggregate_all_rows(table, function, argument)
    tensor = scalar.require_tensor().expand(table.row_count).clone()
    valid = None if scalar.valid is None else scalar.valid.expand(table.row_count).clone()
    return PhysicalValue(tensor, valid=valid, meta=scalar.meta)


def _aggregate_group_values(
    table: PhysicalTable,
    function: str,
    argument: PhysicalValue | None,
    group_ids: torch.Tensor,
    group_count: int,
) -> PhysicalValue:
    device = table_device(table)
    if function == "count" and argument is None:
        ones = torch.ones(group_ids.numel(), dtype=torch.int64, device=device)
        return PhysicalValue(_scatter_sum(ones, group_ids, group_count))
    if argument is None:
        raise UnsupportedPlanError(f"WINDOW {function} requires an argument")
    values = argument.require_tensor()
    valid = argument.valid if argument.valid is not None else torch.ones(values.shape, dtype=torch.bool, device=device)
    if function == "count":
        return PhysicalValue(_scatter_sum(valid.to(dtype=torch.int64), group_ids, group_count))
    if function == "sum":
        return PhysicalValue(_scatter_sum(torch.where(valid, values, torch.zeros_like(values)), group_ids, group_count), valid=_valid_counts(valid, group_ids, group_count) > 0, meta=argument.meta)
    if function == "avg":
        sums = _scatter_sum(torch.where(valid, values.to(dtype=torch.float64), torch.zeros_like(values, dtype=torch.float64)), group_ids, group_count)
        counts = _scatter_sum(valid.to(dtype=torch.float64), group_ids, group_count)
        return PhysicalValue(sums / torch.clamp(counts, min=1.0), valid=counts > 0)
    if function in {"min", "max"}:
        return _group_min_max(values, valid, group_ids, group_count, function, argument.meta)
    raise UnsupportedPlanError(f"unsupported aggregate WINDOW function: {function}")


def _aggregate_all_rows(table: PhysicalTable, function: str, argument: PhysicalValue | None) -> PhysicalValue:
    group_ids = torch.zeros(table.row_count, dtype=torch.int64, device=table_device(table))
    return _aggregate_group_values(table, function, argument, group_ids, 1)


def _scatter_sum(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    return result.index_add(0, group_ids.to(dtype=torch.int64), values)


def _valid_counts(valid: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    return _scatter_sum(valid.to(dtype=torch.int64), group_ids, group_count)


def _group_min_max(
    values: torch.Tensor,
    valid: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
    function: str,
    meta,
) -> PhysicalValue:
    reduce = "amin" if function == "min" else "amax"
    fill = _reduce_fill_value(values.dtype, reduce)
    result = torch.full((group_count,), fill, dtype=values.dtype, device=values.device)
    safe = torch.where(valid, values, torch.full_like(values, fill))
    reduced = result.scatter_reduce(0, group_ids.to(dtype=torch.int64), safe, reduce=reduce, include_self=True)
    return PhysicalValue(reduced, valid=_valid_counts(valid, group_ids, group_count) > 0, meta=meta)


def _reduce_fill_value(dtype: torch.dtype, reduce: str) -> int | float:
    if dtype.is_floating_point:
        return float("inf") if reduce == "amin" else float("-inf")
    info = torch.iinfo(dtype)
    return info.max if reduce == "amin" else info.min


def _gather_valid(value: PhysicalValue, inverse: torch.Tensor) -> torch.Tensor | None:
    return None if value.valid is None else value.valid.index_select(0, inverse)
