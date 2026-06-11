"""TPC-H Q10 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.backend.graph_nodes import aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values, string_eq


def execute_q10_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey", "o_orderdate"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_name", "c_acctbal", "c_nationkey", "c_address", "c_phone", "c_comment"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    order_idx = lookup_values(orders.columns["o_orderkey"], torch.arange(len(orders), device=device), lineitem.columns["l_orderkey"])
    orderdate = orders.columns["o_orderdate"][order_idx.clamp_min(0)]
    custkey = orders.columns["o_custkey"][order_idx.clamp_min(0)]
    cust_idx = lookup_values(customer.columns["c_custkey"], torch.arange(len(customer), device=device), custkey)
    mask = (
        (order_idx >= 0)
        & (cust_idx >= 0)
        & (orderdate >= 19931001)
        & (orderdate < 19940101)
        & string_eq(lineitem, "l_returnflag", "R")
    )
    revenue = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    keys, sums = aggregate_sum_by_keys([cust_idx[mask]], revenue)
    rows: list[dict[str, Any]] = []
    for i in range(int(keys.shape[0])):
        customer_row = int(keys[i, 0].item())
        nation_name = lookup_values(
            nation.columns["n_nationkey"],
            nation.columns["n_name"],
            customer.columns["c_nationkey"][customer_row].reshape(1),
        )[0]
        rows.append(
            {
                "c_custkey": int(customer.columns["c_custkey"][customer_row].item()),
                "c_name": decode(customer, "c_name", customer.columns["c_name"][customer_row]),
                "revenue": float(sums[i].cpu().item()),
                "c_acctbal": float(customer.columns["c_acctbal"][customer_row].cpu().item()),
                "n_name": decode(nation, "n_name", nation_name),
                "c_address": decode(customer, "c_address", customer.columns["c_address"][customer_row]),
                "c_phone": decode(customer, "c_phone", customer.columns["c_phone"][customer_row]),
                "c_comment": decode(customer, "c_comment", customer.columns["c_comment"][customer_row]),
            }
        )
    return sorted(rows, key=lambda row: -row["revenue"])[:20]
