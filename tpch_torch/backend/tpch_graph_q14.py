"""TPC-H Q14 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import fetch_tensor_table, lookup_values, string_startswith


def execute_q14_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"], device)
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_type"], device)

    is_promo_part = string_startswith(part, "p_type", "PROMO").to(dtype=lineitem.columns["l_partkey"].dtype)
    promo = lookup_values(part.columns["p_partkey"], is_promo_part, lineitem.columns["l_partkey"], missing_value=0)
    mask = (lineitem.columns["l_shipdate"] >= 19950901) & (lineitem.columns["l_shipdate"] < 19951001)
    volume = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    promo_volume = volume[promo[mask] == 1].sum()
    promo_revenue = 100.0 * promo_volume / volume.sum()
    return [{"promo_revenue": float(promo_revenue.cpu().item())}]
