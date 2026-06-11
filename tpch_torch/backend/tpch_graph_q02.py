"""TPC-H Q2 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph_nodes import GroupedScalarSubqueryNode, decode, fetch_tensor_table, lookup_values, string_eq


def execute_q2_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_mfgr", "p_size", "p_type"], device)
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"], device)
    supplier = fetch_tensor_table(
        con,
        "supplier",
        ["s_suppkey", "s_acctbal", "s_name", "s_address", "s_phone", "s_comment", "s_nationkey"],
        device,
    )
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name", "n_regionkey"], device)
    region = fetch_tensor_table(con, "region", ["r_regionkey", "r_name"], device)

    europe = region.columns["r_regionkey"][string_eq(region, "r_name", "EUROPE")][0]
    part_row = lookup_values(part.columns["p_partkey"], _row_ids(part), partsupp.columns["ps_partkey"])
    supplier_row = lookup_values(supplier.columns["s_suppkey"], _row_ids(supplier), partsupp.columns["ps_suppkey"])
    nation_row = lookup_values(nation.columns["n_nationkey"], _row_ids(nation), supplier.columns["s_nationkey"])
    supplier_region = nation.columns["n_regionkey"][nation_row.clamp_min(0)]
    mask = (
        (part_row >= 0)
        & (supplier_row >= 0)
        & (part.columns["p_size"][part_row.clamp_min(0)] == 15)
        & _part_type_endswith(part, part.columns["p_type"][part_row.clamp_min(0)], "BRASS")
        & (supplier_region[supplier_row.clamp_min(0)] == europe)
    )
    min_cost_by_part = _min_cost_by_part(partsupp, mask)
    rows = _materialize_rows(part, partsupp, supplier, nation, part_row, supplier_row, min_cost_by_part, mask)
    return sorted(rows, key=lambda row: (-row["s_acctbal"], row["n_name"], row["s_name"], row["p_partkey"]))[:100]


def _row_ids(table):
    import torch

    return torch.arange(len(table), dtype=torch.int64, device=next(iter(table.columns.values())).device)


def _part_type_endswith(part, encoded_type, suffix: str):
    import torch

    matching_ids = [index for index, value in enumerate(part.dictionaries["p_type"]) if value.endswith(suffix)]
    return torch.isin(encoded_type, torch.tensor(matching_ids, dtype=encoded_type.dtype, device=encoded_type.device))


def _min_cost_by_part(partsupp, mask):
    return GroupedScalarSubqueryNode.min(
        (partsupp.columns["ps_partkey"][mask],),
        partsupp.columns["ps_supplycost"][mask],
    )


def _materialize_rows(part, partsupp, supplier, nation, part_row, supplier_row, min_cost_by_part, mask):
    rows = []
    indices = mask.nonzero().flatten().cpu().tolist()
    supplier_nation = lookup_values(nation.columns["n_nationkey"], nation.columns["n_name"], supplier.columns["s_nationkey"])
    for raw_index in indices:
        index = int(raw_index)
        partkey = int(partsupp.columns["ps_partkey"][index].item())
        cost = float(partsupp.columns["ps_supplycost"][index].cpu().item())
        min_cost = min_cost_by_part.lookup((partsupp.columns["ps_partkey"][index].reshape(1),))[0]
        if cost != float(min_cost.cpu().item()):
            continue
        p_index = int(part_row[index].item())
        s_index = int(supplier_row[index].item())
        rows.append(
            {
                "s_acctbal": float(supplier.columns["s_acctbal"][s_index].cpu().item()),
                "s_name": decode(supplier, "s_name", supplier.columns["s_name"][s_index]),
                "n_name": decode(nation, "n_name", supplier_nation[s_index]),
                "p_partkey": partkey,
                "p_mfgr": decode(part, "p_mfgr", part.columns["p_mfgr"][p_index]),
                "s_address": decode(supplier, "s_address", supplier.columns["s_address"][s_index]),
                "s_phone": decode(supplier, "s_phone", supplier.columns["s_phone"][s_index]),
                "s_comment": decode(supplier, "s_comment", supplier.columns["s_comment"][s_index]),
            }
        )
    return rows
