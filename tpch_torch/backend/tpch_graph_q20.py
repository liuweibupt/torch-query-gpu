"""TPC-H Q20 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.backend.graph_nodes import GroupedScalarSubqueryNode, SemiJoinNode, decode, fetch_tensor_table, string_eq, string_startswith


def execute_q20_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_name"], device)
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"], device)
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_name", "s_address", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    ship_mask = (lineitem.columns["l_shipdate"] >= 19940101) & (lineitem.columns["l_shipdate"] < 19950101)
    shipped_quantity_by_pair = GroupedScalarSubqueryNode.sum(
        (lineitem.columns["l_partkey"][ship_mask], lineitem.columns["l_suppkey"][ship_mask]),
        lineitem.columns["l_quantity"][ship_mask],
    )
    shipped_quantity = shipped_quantity_by_pair.lookup(
        (partsupp.columns["ps_partkey"], partsupp.columns["ps_suppkey"]),
        missing_value=0.0,
    )
    has_shipment = shipped_quantity > 0.0
    forest_partkeys = part.columns["p_partkey"][string_startswith(part, "p_name", "forest")]
    qualifying_suppkeys = torch.unique(partsupp.columns["ps_suppkey"][
        SemiJoinNode(partsupp.columns["ps_partkey"], forest_partkeys).execute()
        & has_shipment
        & (partsupp.columns["ps_availqty"] > 0.5 * shipped_quantity)
    ])
    canada_key = nation.columns["n_nationkey"][string_eq(nation, "n_name", "CANADA")][0]
    supplier_mask = torch.isin(supplier.columns["s_suppkey"], qualifying_suppkeys) & (supplier.columns["s_nationkey"] == canada_key)
    rows = [
        {
            "s_name": decode(supplier, "s_name", supplier.columns["s_name"][index]),
            "s_address": decode(supplier, "s_address", supplier.columns["s_address"][index]),
        }
        for index in supplier_mask.nonzero().flatten()
    ]
    return sorted(rows, key=lambda row: row["s_name"])
