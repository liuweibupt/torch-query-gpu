"""TPC-H Q17 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_count_by_keys, aggregate_sum_by_keys, fetch_tensor_table, lookup_values, string_eq


def execute_q17_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    lineitem = fetch_tensor_table(con, "lineitem", ["l_partkey", "l_quantity", "l_extendedprice"], device)
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_brand", "p_container"], device)

    part_mask = string_eq(part, "p_brand", "Brand#23") & string_eq(part, "p_container", "MED BOX")
    candidate_partkeys = part.columns["p_partkey"][part_mask]
    grouped_keys, quantity_sums = aggregate_sum_by_keys([lineitem.columns["l_partkey"]], lineitem.columns["l_quantity"])
    _, quantity_counts = aggregate_count_by_keys([lineitem.columns["l_partkey"]])
    avg_quantity = quantity_sums / quantity_counts.to(dtype=quantity_sums.dtype)
    lineitem_avg_quantity = lookup_values(grouped_keys[:, 0], avg_quantity, lineitem.columns["l_partkey"])
    mask = (
        torch.isin(lineitem.columns["l_partkey"], candidate_partkeys)
        & (lineitem.columns["l_quantity"] < 0.2 * lineitem_avg_quantity)
    )
    if not bool(mask.any().item()):
        return [{"avg_yearly": None}]
    avg_yearly = lineitem.columns["l_extendedprice"][mask].sum() / 7.0
    return [{"avg_yearly": float(avg_yearly.cpu().item())}]
