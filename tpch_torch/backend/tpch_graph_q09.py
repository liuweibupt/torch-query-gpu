"""TPC-H Q9 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import aggregate_sum_by_keys, composite_key, decode, fetch_tensor_table, lookup_values, string_contains, yyyymmdd_to_year

KEY_MULTIPLIER = 1_000_000


def execute_q9_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_suppkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"], device)
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_name"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_nationkey"], device)
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_orderdate"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    part_mask = lookup_values(part.columns["p_partkey"], string_contains(part, "p_name", "green").to(dtype=lineitem.columns["l_partkey"].dtype), lineitem.columns["l_partkey"], 0) == 1
    supp_nation = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_nationkey"], lineitem.columns["l_suppkey"])
    nation_name = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], supp_nation)
    orderdate = lookup_values(orders.columns["o_orderkey"], orders.columns["o_orderdate"], lineitem.columns["l_orderkey"])
    ps_key = composite_key(partsupp.columns["ps_partkey"], partsupp.columns["ps_suppkey"], KEY_MULTIPLIER)
    li_key = composite_key(lineitem.columns["l_partkey"], lineitem.columns["l_suppkey"], KEY_MULTIPLIER)
    supplycost = lookup_values(ps_key, partsupp.columns["ps_supplycost"], li_key)
    mask = part_mask & (supplycost >= 0) & (orderdate >= 0) & (nation_name >= 0)
    amount = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask]) - supplycost[mask] * lineitem.columns["l_quantity"][mask]
    keys, sums = aggregate_sum_by_keys([nation_name[mask], yyyymmdd_to_year(orderdate[mask])], amount)
    rows = [{"nation": decode(nation, "n_name", keys[i, 0]), "o_year": int(keys[i, 1]), "sum_profit": float(sums[i].cpu().item())} for i in range(int(keys.shape[0]))]
    return sorted(rows, key=lambda row: (row["nation"], -row["o_year"]))
