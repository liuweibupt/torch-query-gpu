"""TPC-H Q8 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.backend.graph_nodes import aggregate_sum_by_keys, fetch_tensor_table, lookup_values, string_eq, yyyymmdd_to_year


def execute_q8_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_suppkey", "l_orderkey", "l_extendedprice", "l_discount"], device)
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_type"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_nationkey"], device)
    orders = fetch_tensor_table(con, "orders", ["o_orderkey", "o_custkey", "o_orderdate"], device)
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name", "n_regionkey"], device)
    region = fetch_tensor_table(con, "region", ["r_regionkey", "r_name"], device)

    part_type = lookup_values(part.columns["p_partkey"], part.columns["p_type"], lineitem.columns["l_partkey"])
    target_type = int(part.columns["p_type"][string_eq(part, "p_type", "ECONOMY ANODIZED STEEL")][0])
    order_idx = lookup_values(orders.columns["o_orderkey"], torch.arange(len(orders), device=device), lineitem.columns["l_orderkey"])
    cust_nation = lookup_values(customer.columns["c_custkey"], customer.columns["c_nationkey"], orders.columns["o_custkey"][order_idx.clamp_min(0)])
    cust_region = lookup_values(nation.columns["n_nationkey"], nation.columns["n_regionkey"], cust_nation)
    america = region.columns["r_regionkey"][string_eq(region, "r_name", "AMERICA")][0]
    supp_nation = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_nationkey"], lineitem.columns["l_suppkey"])
    supp_name = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], supp_nation)
    brazil = int(nation.columns["n_name"][string_eq(nation, "n_name", "BRAZIL")][0])
    orderdate = orders.columns["o_orderdate"][order_idx.clamp_min(0)]
    mask = (order_idx >= 0) & (part_type == target_type) & (cust_region == america) & (orderdate >= 19950101) & (orderdate <= 19961231)
    volume = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    years = yyyymmdd_to_year(orderdate[mask])
    keys, total = aggregate_sum_by_keys([years], volume)
    _, brazil_total = aggregate_sum_by_keys([years], torch.where(supp_name[mask] == brazil, volume, torch.zeros_like(volume)))
    rows = [{"o_year": int(keys[i, 0]), "mkt_share": float((brazil_total[i] / total[i]).cpu().item())} for i in range(int(keys.shape[0]))]
    return sorted(rows, key=lambda row: row["o_year"])
