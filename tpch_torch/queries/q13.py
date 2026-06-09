"""TPC-H Q13 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import aggregate_count_by_keys, fetch_tensor_table, lookup_values, string_not_like_special_requests


def execute_q13(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    customer = fetch_tensor_table(con, "customer", ["c_custkey"], device)
    orders = fetch_tensor_table(con, "orders", ["o_custkey", "o_orderkey", "o_comment"], device)

    valid_orders = string_not_like_special_requests(orders, "o_comment")
    order_keys, counts = aggregate_count_by_keys([orders.columns["o_custkey"][valid_orders]])
    customer_counts = lookup_values(order_keys[:, 0], counts, customer.columns["c_custkey"], missing_value=0)
    count_keys, distributions = aggregate_count_by_keys([customer_counts])
    rows = [
        {"c_count": int(count_keys[i, 0].item()), "custdist": int(distributions[i].cpu().item())}
        for i in range(int(count_keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: (-row["custdist"], -row["c_count"]))
