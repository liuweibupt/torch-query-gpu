"""TPC-H Q6 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import fetch_tensor_table

LINEITEM_COLUMNS = (
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_shipdate",
)


def execute_q6(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    table = fetch_tensor_table(con, "lineitem", LINEITEM_COLUMNS, device=device)
    columns = table.columns
    mask = (
        (columns["l_shipdate"] >= 19940101)
        & (columns["l_shipdate"] < 19950101)
        & (columns["l_discount"] >= 0.05)
        & (columns["l_discount"] <= 0.07)
        & (columns["l_quantity"] < 24.0)
    )
    revenue = (columns["l_extendedprice"][mask] * columns["l_discount"][mask]).sum()
    return [{"revenue": float(revenue.cpu().item())}]
