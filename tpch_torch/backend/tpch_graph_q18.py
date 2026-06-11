"""TPC-H Q18 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph_nodes import SemiJoinNode, aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values, yyyymmdd_to_iso


def execute_q18_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_quantity"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_name"], device)

    order_keys, qty_sums = aggregate_sum_by_keys([lineitem.columns["l_orderkey"]], lineitem.columns["l_quantity"])
    large_orderkeys = order_keys[:, 0][qty_sums > 300.0]
    order_mask = SemiJoinNode(orders.columns["o_orderkey"], large_orderkeys).execute()
    qty_by_order = lookup_values(order_keys[:, 0], qty_sums, orders.columns["o_orderkey"])
    cust_name = lookup_values(customer.columns["c_custkey"], customer.columns["c_name"], orders.columns["o_custkey"])
    rows = []
    for index in order_mask.nonzero().flatten():
        rows.append(
            {
                "c_name": decode(customer, "c_name", cust_name[index]),
                "c_custkey": int(orders.columns["o_custkey"][index].item()),
                "o_orderkey": int(orders.columns["o_orderkey"][index].item()),
                "o_orderdate": yyyymmdd_to_iso(int(orders.columns["o_orderdate"][index].item())),
                "o_totalprice": float(orders.columns["o_totalprice"][index].cpu().item()),
                "sum(l_quantity)": float(qty_by_order[index].cpu().item()),
            }
        )
    return sorted(rows, key=lambda row: (-row["o_totalprice"], row["o_orderdate"]))[:100]
