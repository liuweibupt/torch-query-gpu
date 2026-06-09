"""TPC-H Q3 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import aggregate_sum_by_keys, fetch_tensor_table, lookup_row_indices, lookup_values, string_eq, yyyymmdd_to_iso


def execute_q3(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_mktsegment"], device)

    order_rows = lookup_row_indices(orders.columns["o_orderkey"], lineitem.columns["l_orderkey"])
    order_ok = order_rows >= 0
    order_custkeys = orders.columns["o_custkey"][order_rows.clamp_min(0)]
    customer_segment = lookup_values(customer.columns["c_custkey"], customer.columns["c_mktsegment"], order_custkeys)
    building_id = int(customer.columns["c_mktsegment"][string_eq(customer, "c_mktsegment", "BUILDING")][0])
    mask = (
        order_ok
        & (customer_segment == building_id)
        & (orders.columns["o_orderdate"][order_rows.clamp_min(0)] < 19950315)
        & (lineitem.columns["l_shipdate"] > 19950315)
    )
    revenue = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    keys, sums = aggregate_sum_by_keys(
        [
            lineitem.columns["l_orderkey"][mask],
            orders.columns["o_orderdate"][order_rows.clamp_min(0)][mask],
            orders.columns["o_shippriority"][order_rows.clamp_min(0)][mask],
        ],
        revenue,
    )
    rows = [
        {
            "l_orderkey": int(keys[i, 0].item()),
            "revenue": float(sums[i].cpu().item()),
            "o_orderdate": yyyymmdd_to_iso(int(keys[i, 1].item())),
            "o_shippriority": int(keys[i, 2].item()),
        }
        for i in range(int(keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: (-row["revenue"], row["o_orderdate"]))[:10]
