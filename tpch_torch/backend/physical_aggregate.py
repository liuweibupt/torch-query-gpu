"""Aggregate execution helpers for DuckDB physical plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch

from tpch_torch.backend.physical_expr import aggregate_output_aliases, evaluate_expression, projection_name
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue, table_device
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operator_graph import TQPOperatorNode

_AGGREGATE_FUNCTION_ALIASES = {"sum_no_overflow": "sum"}


@dataclass(frozen=True)
class AggregateSpec:
    function: str
    argument: str | None
    aliases: tuple[str, ...]
    distinct: bool = False


def execute_grouped_aggregate(
    child: PhysicalTable,
    group_exprs: Sequence[str],
    specs: Sequence[AggregateSpec],
) -> PhysicalTable:
    key_values = [evaluate_expression(child, expression) for expression in group_exprs]
    key_dtype = _group_key_dtype(key_values)
    key_tensors = [value.require_tensor().to(dtype=key_dtype) for value in key_values]
    stacked = torch.stack(key_tensors, dim=1)
    unique_keys, inverse, keys_sorted = _unique_group_keys(stacked, key_values)
    row_count = int(unique_keys.shape[0])
    items: list[tuple[str, PhysicalValue, Sequence[str]]] = []
    for index, (expression, value) in enumerate(zip(group_exprs, key_values)):
        name, aliases = projection_name(child, expression, index)
        key_tensor = unique_keys[:, index].to(dtype=value.require_tensor().dtype)
        items.append(
            (name, _group_key_value(key_tensor, value, len(group_exprs), keys_sorted), aliases)
        )
    for spec in specs:
        value = _evaluate_group_aggregate(child, inverse, row_count, spec)
        items.append((spec.aliases[0], value, spec.aliases))
    return PhysicalTable.projected("aggregate", items, row_count)


def _group_key_dtype(values: Sequence[PhysicalValue]) -> torch.dtype:
    if any(value.require_tensor().dtype.is_floating_point for value in values):
        return torch.float64
    return torch.int64


def _unique_group_keys(
    stacked_keys: torch.Tensor,
    key_values: Sequence[PhysicalValue],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if _keys_known_sorted(key_values) or _is_lexicographically_non_decreasing(stacked_keys):
        unique_keys, inverse = torch.unique_consecutive(stacked_keys, dim=0, return_inverse=True)
        return unique_keys, inverse, True
    unique_keys, inverse = torch.unique(stacked_keys, dim=0, sorted=True, return_inverse=True)
    return unique_keys, inverse, True


def _keys_known_sorted(key_values: Sequence[PhysicalValue]) -> bool:
    return len(key_values) == 1 and key_values[0].sorted_non_decreasing


def _group_key_value(
    key_tensor: torch.Tensor,
    source: PhysicalValue,
    group_key_count: int,
    keys_sorted: bool,
) -> PhysicalValue:
    return PhysicalValue(
        key_tensor,
        source.dictionary,
        source.is_date,
        sorted_non_decreasing=group_key_count == 1 and keys_sorted,
        unique=group_key_count == 1,
    )


def _is_lexicographically_non_decreasing(stacked_keys: torch.Tensor) -> bool:
    if stacked_keys.shape[0] <= 1:
        return True
    previous = stacked_keys[:-1]
    current = stacked_keys[1:]
    changed = current != previous
    equal_rows = ~torch.any(changed, dim=1)
    first_changed = changed.to(dtype=torch.int64).argmax(dim=1)
    current_first = current.gather(1, first_changed.reshape(-1, 1)).flatten()
    previous_first = previous.gather(1, first_changed.reshape(-1, 1)).flatten()
    return bool(torch.all(equal_rows | (current_first > previous_first)).cpu().item())


def execute_ungrouped_aggregate(child: PhysicalTable, specs: Sequence[AggregateSpec]) -> PhysicalTable:
    items = [(spec.aliases[0], _evaluate_scalar_aggregate(child, spec), spec.aliases) for spec in specs]
    return PhysicalTable.projected("aggregate", items, 1)


def _evaluate_group_aggregate(
    child: PhysicalTable,
    group_ids: torch.Tensor,
    group_count: int,
    spec: AggregateSpec,
) -> PhysicalValue:
    if spec.function == "count_star":
        ones = torch.ones(group_ids.numel(), dtype=torch.int64, device=group_ids.device)
        return PhysicalValue(_scatter_sum(ones, group_ids, group_count))
    argument = _aggregate_argument(child, spec)
    values = argument.require_tensor()
    if spec.function == "count":
        if spec.distinct:
            return PhysicalValue(_scatter_count_distinct(values, group_ids, group_count))
        valid = _validity_or_ones(argument, values)
        return PhysicalValue(_scatter_sum(valid.to(dtype=torch.int64), group_ids, group_count))
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
    if spec.function == "first":
        return PhysicalValue(_scatter_reduce(values, group_ids, group_count, "amin"))
    raise UnsupportedPlanError(f"unsupported grouped aggregate: {spec.function}")


def _evaluate_scalar_aggregate(child: PhysicalTable, spec: AggregateSpec) -> PhysicalValue:
    if spec.function == "count_star":
        tensor = torch.tensor([child.row_count], dtype=torch.int64, device=table_device(child))
        return PhysicalValue(tensor)
    argument = _aggregate_argument(child, spec)
    values = argument.require_tensor()
    if spec.function == "count" and spec.distinct:
        tensor = torch.unique(values).numel()
        tensor = torch.tensor([tensor], dtype=torch.int64, device=table_device(child))
    elif spec.function == "count":
        tensor = _validity_or_ones(argument, values).sum().reshape(1)
    elif values.numel() == 0:
        tensor = torch.tensor([0.0], dtype=torch.float64, device=table_device(child))
        valid = torch.zeros(1, dtype=torch.bool, device=table_device(child))
        return PhysicalValue(tensor=tensor, valid=valid)
    elif spec.function == "sum":
        tensor = values.sum().reshape(1)
    elif spec.function == "min":
        tensor = values.min().reshape(1)
    elif spec.function == "max":
        tensor = values.max().reshape(1)
    elif spec.function == "avg":
        tensor = values.to(dtype=torch.float64).mean().reshape(1)
    elif spec.function == "first":
        tensor = values[:1]
    else:
        raise UnsupportedPlanError(f"unsupported scalar aggregate: {spec.function}")
    return PhysicalValue(tensor)


def _aggregate_argument(child: PhysicalTable, spec: AggregateSpec) -> PhysicalValue:
    if spec.argument is None:
        raise UnsupportedPlanError(f"aggregate requires an argument: {spec.function}")
    return evaluate_expression(child, spec.argument)


def _validity_or_ones(value: PhysicalValue, tensor: torch.Tensor) -> torch.Tensor:
    if value.valid is not None:
        return value.valid
    return torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)


def _metadata_list(node: TQPOperatorNode, key: str) -> tuple[str, ...]:
    value = node.metadata.get(key)
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def aggregate_specs(node: TQPOperatorNode, child: PhysicalTable) -> tuple[AggregateSpec, ...]:
    specs = []
    for raw in _metadata_list(node, "Aggregates"):
        if raw.lower() == "count_star()":
            specs.append(AggregateSpec("count_star", None, ("count_star()", "count(*)")))
            continue
        match = re.fullmatch(r'(?:"?)(sum_no_overflow|sum|avg|min|max|count|first)(?:"?)\((.*)\)', raw.strip(), re.I)
        if match is None:
            raise UnsupportedPlanError(f"unsupported aggregate expression: {raw}")
        function = _canonical_aggregate_function(match.group(1))
        argument = match.group(2).strip()
        distinct = argument.upper().startswith("DISTINCT ")
        argument = argument[len("DISTINCT ") :].strip() if distinct else argument
        child_name = _child_name(child, argument)
        specs.append(AggregateSpec(function, argument, aggregate_output_aliases(function, argument, child_name), distinct))
    return tuple(specs)


_aggregate_specs = aggregate_specs


def _canonical_aggregate_function(function: str) -> str:
    lowered = function.lower()
    return _AGGREGATE_FUNCTION_ALIASES.get(lowered, lowered)


def _child_name(child: PhysicalTable, argument: str) -> str | None:
    stripped = argument.strip()
    if not stripped.startswith("#"):
        return stripped
    index = int(stripped[1:])
    if index >= len(child.order):
        return None
    return child.order[index]


def _scatter_sum(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    return result.index_add(0, group_ids.to(dtype=torch.int64), values)


def _scatter_reduce(values: torch.Tensor, group_ids: torch.Tensor, group_count: int, reduce: str) -> torch.Tensor:
    fill_value = float("inf") if reduce == "amin" else float("-inf")
    result = torch.full((group_count,), fill_value, dtype=values.dtype, device=values.device)
    return result.scatter_reduce(0, group_ids.to(dtype=torch.int64), values, reduce=reduce, include_self=True)


def _scatter_count_distinct(values: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    pairs = torch.stack((group_ids.to(dtype=torch.int64), values.to(dtype=torch.int64)), dim=1)
    unique_pairs = torch.unique(pairs, dim=0, sorted=True)
    ones = torch.ones(unique_pairs.shape[0], dtype=torch.int64, device=values.device)
    return _scatter_sum(ones, unique_pairs[:, 0], group_count)
