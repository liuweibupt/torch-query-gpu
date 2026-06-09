"""TPC-H Q15 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import aggregate_sum_by_keys, decode, fetch_tensor_table, lookup_values


def execute_q15(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_name", "s_address", "s_phone"], device)

    mask = (lineitem.columns["l_shipdate"] >= 19960101) & (lineitem.columns["l_shipdate"] < 19960401)
    revenue = lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])
    keys, sums = aggregate_sum_by_keys([lineitem.columns["l_suppkey"][mask]], revenue)
    max_revenue = sums.max()
    selected_supplier = keys[:, 0][sums == max_revenue]
    supplier_rows = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_suppkey"], selected_supplier)
    rows = []
    for suppkey in supplier_rows:
        supplier_mask = supplier.columns["s_suppkey"] == suppkey
        supplier_index = int(supplier_mask.nonzero()[0].item())
        rows.append(
            {
                "s_suppkey": int(suppkey.item()),
                "s_name": decode(supplier, "s_name", supplier.columns["s_name"][supplier_index]),
                "s_address": decode(supplier, "s_address", supplier.columns["s_address"][supplier_index]),
                "s_phone": decode(supplier, "s_phone", supplier.columns["s_phone"][supplier_index]),
                "total_revenue": float(max_revenue.cpu().item()),
            }
        )
    return sorted(rows, key=lambda row: row["s_suppkey"])
