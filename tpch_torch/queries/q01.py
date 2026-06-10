"""TPC-H Q1 execution on PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tpch_torch.storage import TensorTable
from tpch_torch.substrait import Q1Plan

Q1_RESULT_COLUMNS = (
    "l_returnflag",
    "l_linestatus",
    "sum_qty",
    "sum_base_price",
    "sum_disc_price",
    "sum_charge",
    "avg_qty",
    "avg_price",
    "avg_disc",
    "count_order",
)


def execute_q1(table: TensorTable, plan: Q1Plan) -> list[dict[str, Any]]:
    """Execute the validated TPC-H Q1 plan against a columnar tensor table."""

    table.require_columns(plan.required_columns)
    filtered = _filter_q1(table, plan)
    if _filtered_row_count(filtered) == 0:
        return []

    grouping = _build_low_cardinality_grouping(table, filtered)
    aggregates = _aggregate_q1(filtered, grouping.group_ids, grouping.total_group_count)
    non_empty_group_ids, aggregates = _compact_groups(aggregates)
    unique_keys = _decode_group_ids(non_empty_group_ids, grouping.status_count)
    return _format_rows(table, unique_keys, aggregates)


@dataclass(frozen=True)
class _Q1Grouping:
    group_ids: torch.Tensor
    status_count: int
    total_group_count: int


def _filter_q1(table: TensorTable, plan: Q1Plan) -> dict[str, torch.Tensor]:
    mask = table.columns["l_shipdate"] <= plan.shipdate_cutoff_yyyymmdd
    selected_rows = torch.nonzero(mask).flatten()
    return {
        name: table.columns[name].index_select(0, selected_rows)
        for name in plan.required_columns
        if name != "l_shipdate"
    }


def _filtered_row_count(columns: dict[str, torch.Tensor]) -> int:
    first_column = next(iter(columns.values()), None)
    if first_column is None:
        return 0
    return int(first_column.numel())


def _build_low_cardinality_grouping(
    table: TensorTable, columns: dict[str, torch.Tensor]
) -> _Q1Grouping:
    status_count = len(table.dictionaries["l_linestatus"])
    flag_count = len(table.dictionaries["l_returnflag"])
    group_ids = (columns["l_returnflag"].to(dtype=torch.int64) * status_count) + columns[
        "l_linestatus"
    ].to(dtype=torch.int64)
    return _Q1Grouping(
        group_ids=group_ids,
        status_count=status_count,
        total_group_count=flag_count * status_count,
    )


def _aggregate_q1(
    columns: dict[str, torch.Tensor], group_ids: torch.Tensor, group_count: int
) -> dict[str, torch.Tensor]:
    quantity = columns["l_quantity"]
    extendedprice = columns["l_extendedprice"]
    discount = columns["l_discount"]
    tax = columns["l_tax"]

    discounted_price = extendedprice * (1.0 - discount)
    charge = discounted_price * (1.0 + tax)
    count_order = torch.bincount(group_ids, minlength=group_count)
    count_as_float = count_order.to(dtype=quantity.dtype)

    sum_qty = torch.bincount(group_ids, weights=quantity, minlength=group_count)
    sum_base_price = torch.bincount(group_ids, weights=extendedprice, minlength=group_count)
    sum_discount = torch.bincount(group_ids, weights=discount, minlength=group_count)

    return {
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": torch.bincount(
            group_ids, weights=discounted_price, minlength=group_count
        ),
        "sum_charge": torch.bincount(group_ids, weights=charge, minlength=group_count),
        "avg_qty": sum_qty / count_as_float,
        "avg_price": sum_base_price / count_as_float,
        "avg_disc": sum_discount / count_as_float,
        "count_order": count_order,
    }


def _compact_groups(
    aggregates: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    count_order = aggregates["count_order"]
    non_empty_group_ids = torch.nonzero(count_order > 0).flatten()
    compacted = {name: tensor[non_empty_group_ids] for name, tensor in aggregates.items()}
    return non_empty_group_ids, compacted


def _decode_group_ids(group_ids: torch.Tensor, status_count: int) -> torch.Tensor:
    return torch.stack((group_ids // status_count, group_ids % status_count), dim=1)


def _format_rows(
    table: TensorTable, unique_keys: torch.Tensor, aggregates: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    rows = []
    host_keys = unique_keys.cpu()
    host_aggregates = {name: tensor.cpu() for name, tensor in aggregates.items()}
    for index in range(int(host_keys.shape[0])):
        row = {
            "l_returnflag": table.decode_value("l_returnflag", int(host_keys[index, 0])),
            "l_linestatus": table.decode_value("l_linestatus", int(host_keys[index, 1])),
        }
        for name in Q1_RESULT_COLUMNS[2:]:
            value = host_aggregates[name][index].item()
            row[name] = int(value) if name == "count_order" else float(value)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["l_returnflag"], row["l_linestatus"]))
