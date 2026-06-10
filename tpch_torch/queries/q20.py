"""TPC-H Q20 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_sum_by_keys, composite_key, decode, fetch_tensor_table, lookup_values, string_eq, string_startswith


def execute_q20(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_name"], device)
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"], device)
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_name", "s_address", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    multiplier = int(max(partsupp.columns["ps_suppkey"].max().item(), lineitem.columns["l_suppkey"].max().item())) + 1
    ship_mask = (lineitem.columns["l_shipdate"] >= 19940101) & (lineitem.columns["l_shipdate"] < 19950101)
    line_keys, quantity_sum = aggregate_sum_by_keys(
        [lineitem.columns["l_partkey"][ship_mask], lineitem.columns["l_suppkey"][ship_mask]],
        lineitem.columns["l_quantity"][ship_mask],
    )
    line_composite = composite_key(line_keys[:, 0], line_keys[:, 1], multiplier)
    ps_composite = composite_key(partsupp.columns["ps_partkey"], partsupp.columns["ps_suppkey"], multiplier)
    shipped_quantity = lookup_values(line_composite, quantity_sum, ps_composite, missing_value=0.0)
    forest_partkeys = part.columns["p_partkey"][string_startswith(part, "p_name", "forest")]
    qualifying_suppkeys = torch.unique(partsupp.columns["ps_suppkey"][
        torch.isin(partsupp.columns["ps_partkey"], forest_partkeys)
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
