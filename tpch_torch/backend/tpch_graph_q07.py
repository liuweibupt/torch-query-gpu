"""TPC-H Q7 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph_nodes import aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values, string_eq, yyyymmdd_to_year


def execute_q7_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_nationkey"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    order_idx = lookup_values(orders.columns["o_orderkey"], orders.columns["o_custkey"], lineitem.columns["l_orderkey"])
    cust_nation = lookup_values(customer.columns["c_custkey"], customer.columns["c_nationkey"], order_idx)
    supp_nation = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_nationkey"], lineitem.columns["l_suppkey"])
    supp_name = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], supp_nation)
    cust_name = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], cust_nation)
    france = int(nation.columns["n_name"][string_eq(nation, "n_name", "FRANCE")][0])
    germany = int(nation.columns["n_name"][string_eq(nation, "n_name", "GERMANY")][0])
    pair = ((supp_name == france) & (cust_name == germany)) | ((supp_name == germany) & (cust_name == france))
    date_mask = (lineitem.columns["l_shipdate"] >= 19950101) & (lineitem.columns["l_shipdate"] <= 19961231)
    mask = (order_idx >= 0) & pair & date_mask
    volume = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    keys, sums = aggregate_sum_by_keys([supp_name[mask], cust_name[mask], yyyymmdd_to_year(lineitem.columns["l_shipdate"][mask])], volume)
    rows = [{"supp_nation": decode(nation, "n_name", keys[i, 0]), "cust_nation": decode(nation, "n_name", keys[i, 1]), "l_year": int(keys[i, 2]), "revenue": float(sums[i].cpu().item())} for i in range(int(keys.shape[0]))]
    return sorted(rows, key=lambda row: (row["supp_nation"], row["cust_nation"], row["l_year"]))
