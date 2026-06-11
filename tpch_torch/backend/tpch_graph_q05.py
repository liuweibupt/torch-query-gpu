"""TPC-H Q5 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values, string_eq


def execute_q5_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey", "o_orderdate"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_nationkey"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name", "n_regionkey"], device)
    region = fetch_tensor_table(con, "region", ["r_regionkey", "r_name"], device)

    asia_region = region.columns["r_regionkey"][string_eq(region, "r_name", "ASIA")]
    asia_nations = nation.columns["n_nationkey"][torch.isin(nation.columns["n_regionkey"], asia_region)]
    order_idx = lookup_values(orders.columns["o_orderkey"], torch.arange(len(orders), device=device), lineitem.columns["l_orderkey"])
    supp_nation = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_nationkey"], lineitem.columns["l_suppkey"])
    cust_key = orders.columns["o_custkey"][order_idx.clamp_min(0)]
    cust_nation = lookup_values(customer.columns["c_custkey"], customer.columns["c_nationkey"], cust_key)
    mask = (
        (order_idx >= 0)
        & (orders.columns["o_orderdate"][order_idx.clamp_min(0)] >= 19940101)
        & (orders.columns["o_orderdate"][order_idx.clamp_min(0)] < 19950101)
        & (cust_nation == supp_nation)
        & torch.isin(supp_nation, asia_nations)
    )
    revenue = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    nation_name = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], supp_nation[mask])
    keys, sums = aggregate_sum_by_keys([nation_name], revenue)
    rows = [{"n_name": decode(nation, "n_name", keys[i, 0]), "revenue": float(sums[i].cpu().item())} for i in range(int(keys.shape[0]))]
    return sorted(rows, key=lambda row: -row["revenue"])
