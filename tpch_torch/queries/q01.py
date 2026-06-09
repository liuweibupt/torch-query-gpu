"""TPC-H Q1 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import torch

from tpch_torch.operators import composite_group_ids, grouped_count, grouped_sum
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
    filtered = _filter_q1(table, plan.shipdate_cutoff_yyyymmdd)
    if filtered["l_shipdate"].numel() == 0:
        return []

    group_ids, unique_keys = composite_group_ids(
        [filtered["l_returnflag"], filtered["l_linestatus"]]
    )
    group_count = int(unique_keys.shape[0])
    aggregates = _aggregate_q1(filtered, group_ids, group_count)
    return _format_rows(table, unique_keys, aggregates)


def _filter_q1(table: TensorTable, shipdate_cutoff_yyyymmdd: int) -> dict[str, torch.Tensor]:
    mask = table.columns["l_shipdate"] <= shipdate_cutoff_yyyymmdd
    return {name: tensor[mask] for name, tensor in table.columns.items()}


def _aggregate_q1(
    columns: dict[str, torch.Tensor], group_ids: torch.Tensor, group_count: int
) -> dict[str, torch.Tensor]:
    quantity = columns["l_quantity"]
    extendedprice = columns["l_extendedprice"]
    discount = columns["l_discount"]
    tax = columns["l_tax"]

    discounted_price = extendedprice * (1.0 - discount)
    charge = discounted_price * (1.0 + tax)
    count_order = grouped_count(group_ids, group_count)
    count_as_float = count_order.to(dtype=quantity.dtype)

    sum_qty = grouped_sum(quantity, group_ids, group_count)
    sum_base_price = grouped_sum(extendedprice, group_ids, group_count)
    sum_discount = grouped_sum(discount, group_ids, group_count)

    return {
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": grouped_sum(discounted_price, group_ids, group_count),
        "sum_charge": grouped_sum(charge, group_ids, group_count),
        "avg_qty": sum_qty / count_as_float,
        "avg_price": sum_base_price / count_as_float,
        "avg_disc": sum_discount / count_as_float,
        "count_order": count_order,
    }


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
