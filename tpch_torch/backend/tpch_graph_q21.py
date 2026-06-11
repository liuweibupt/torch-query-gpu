"""TPC-H Q21 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_count_by_keys, decode, fetch_tensor_table, lookup_values, string_eq


def execute_q21_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_orderstatus"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_name", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    order_finished = lookup_values(
        orders.columns["o_orderkey"],
        string_eq(orders, "o_orderstatus", "F").to(dtype=torch.int64),
        lineitem.columns["l_orderkey"],
        missing_value=0,
    ) == 1
    supplier_row = lookup_values(supplier.columns["s_suppkey"], _row_ids(supplier), lineitem.columns["l_suppkey"])
    saudi_key = nation.columns["n_nationkey"][string_eq(nation, "n_name", "SAUDI ARABIA")][0]
    saudi_supplier = supplier.columns["s_nationkey"][supplier_row.clamp_min(0)] == saudi_key
    late = lineitem.columns["l_receiptdate"] > lineitem.columns["l_commitdate"]
    order_distinct_supp = _distinct_supplier_count_by_order(lineitem.columns["l_orderkey"], lineitem.columns["l_suppkey"])
    late_distinct_supp = _distinct_supplier_count_by_order(lineitem.columns["l_orderkey"][late], lineitem.columns["l_suppkey"][late])
    all_supplier_count = lookup_values(order_distinct_supp[0], order_distinct_supp[1], lineitem.columns["l_orderkey"], missing_value=0)
    late_supplier_count = lookup_values(late_distinct_supp[0], late_distinct_supp[1], lineitem.columns["l_orderkey"], missing_value=0)
    mask = late & order_finished & (supplier_row >= 0) & saudi_supplier & (all_supplier_count > 1) & (late_supplier_count == 1)
    keys, counts = aggregate_count_by_keys([supplier_row[mask]])
    rows = [
        {
            "s_name": decode(supplier, "s_name", supplier.columns["s_name"][int(keys[index, 0].item())]),
            "numwait": int(counts[index].cpu().item()),
        }
        for index in range(int(keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: (-row["numwait"], row["s_name"]))[:100]


def _row_ids(table):
    return torch.arange(len(table), dtype=torch.int64, device=next(iter(table.columns.values())).device)


def _distinct_supplier_count_by_order(orderkeys: torch.Tensor, suppkeys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if orderkeys.numel() == 0:
        return orderkeys, orderkeys.to(dtype=torch.int64)
    multiplier = int(suppkeys.max().item()) + 1
    pair = orderkeys.to(dtype=torch.int64) * multiplier + suppkeys.to(dtype=torch.int64)
    unique_pair = torch.unique(pair)
    unique_orders = torch.div(unique_pair, multiplier, rounding_mode="floor")
    keys, counts = aggregate_count_by_keys([unique_orders])
    return keys[:, 0], counts
