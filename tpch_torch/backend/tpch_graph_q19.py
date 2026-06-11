"""TPC-H Q19 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph_nodes import fetch_tensor_table, lookup_values, string_eq, string_in


def execute_q19_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(
        con,
        "lineitem",
        ["l_partkey", "l_extendedprice", "l_discount", "l_quantity", "l_shipmode", "l_shipinstruct"],
        device,
    )
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_brand", "p_container", "p_size"], device)

    brand = lookup_values(part.columns["p_partkey"], part.columns["p_brand"], lineitem.columns["l_partkey"])
    container = lookup_values(part.columns["p_partkey"], part.columns["p_container"], lineitem.columns["l_partkey"])
    size = lookup_values(part.columns["p_partkey"], part.columns["p_size"], lineitem.columns["l_partkey"])
    shipmode_ok = string_in(lineitem, "l_shipmode", ("AIR", "AIR REG"))
    instruct_ok = string_eq(lineitem, "l_shipinstruct", "DELIVER IN PERSON")
    base = shipmode_ok & instruct_ok
    mask = base & (
        _brand_case(lineitem, part, brand, container, size, "Brand#12", ("SM CASE", "SM BOX", "SM PACK", "SM PKG"), 1, 11, 1, 5)
        | _brand_case(lineitem, part, brand, container, size, "Brand#23", ("MED BAG", "MED BOX", "MED PKG", "MED PACK"), 10, 20, 1, 10)
        | _brand_case(lineitem, part, brand, container, size, "Brand#34", ("LG CASE", "LG BOX", "LG PACK", "LG PKG"), 20, 30, 1, 15)
    )
    revenue = (lineitem.columns["l_extendedprice"][mask] * (1.0 - lineitem.columns["l_discount"][mask])).sum()
    return [{"revenue": float(revenue.cpu().item())}]


def _brand_case(lineitem, part, brand, container, size, brand_name, containers, min_qty, max_qty, min_size, max_size):
    brand_id = int(part.columns["p_brand"][string_eq(part, "p_brand", brand_name)][0])
    container_ids = [index for index, value in enumerate(part.dictionaries["p_container"]) if value in set(containers)]
    import torch

    return (
        (brand == brand_id)
        & torch.isin(container, torch.tensor(container_ids, dtype=container.dtype, device=container.device))
        & (lineitem.columns["l_quantity"] >= float(min_qty))
        & (lineitem.columns["l_quantity"] <= float(max_qty))
        & (size >= min_size)
        & (size <= max_size)
    )
