"""TPC-H Q4 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_count_by_keys, decode, fetch_tensor_table


def execute_q4(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_commitdate", "l_receiptdate"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_orderdate", "o_orderpriority"], device)

    late_lineitem = lineitem.columns["l_commitdate"] < lineitem.columns["l_receiptdate"]
    late_orderkeys = torch.unique(lineitem.columns["l_orderkey"][late_lineitem])
    mask = (
        (orders.columns["o_orderdate"] >= 19930701)
        & (orders.columns["o_orderdate"] < 19931001)
        & torch.isin(orders.columns["o_orderkey"], late_orderkeys)
    )
    keys, counts = aggregate_count_by_keys([orders.columns["o_orderpriority"][mask]])
    rows = [
        {
            "o_orderpriority": decode(orders, "o_orderpriority", keys[index, 0]),
            "order_count": int(counts[index].cpu().item()),
        }
        for index in range(int(keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: row["o_orderpriority"])
