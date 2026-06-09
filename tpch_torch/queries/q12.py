"""TPC-H Q12 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values, string_in


def execute_q12(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_orderpriority"], device)

    priority = lookup_values(orders.columns["o_orderkey"], orders.columns["o_orderpriority"], lineitem.columns["l_orderkey"])
    mask = (
        string_in(lineitem, "l_shipmode", ("MAIL", "SHIP"))
        & (lineitem.columns["l_commitdate"] < lineitem.columns["l_receiptdate"])
        & (lineitem.columns["l_shipdate"] < lineitem.columns["l_commitdate"])
        & (lineitem.columns["l_receiptdate"] >= 19940101)
        & (lineitem.columns["l_receiptdate"] < 19950101)
        & (priority >= 0)
    )
    high_ids = [index for index, value in enumerate(orders.dictionaries["o_orderpriority"]) if value in {"1-URGENT", "2-HIGH"}]
    high = torch.isin(priority[mask], torch.tensor(high_ids, dtype=priority.dtype, device=device)).to(dtype=torch.int64)
    keys, high_counts = aggregate_sum_by_keys([lineitem.columns["l_shipmode"][mask]], high.to(dtype=torch.float64))
    _, total_counts = aggregate_sum_by_keys([lineitem.columns["l_shipmode"][mask]], torch.ones_like(high, dtype=torch.float64))
    rows = [
        {
            "l_shipmode": decode(lineitem, "l_shipmode", keys[i, 0]),
            "high_line_count": int(high_counts[i].cpu().item()),
            "low_line_count": int((total_counts[i] - high_counts[i]).cpu().item()),
        }
        for i in range(int(keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: row["l_shipmode"])
