"""TPC-H Q11 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.relational import aggregate_sum_by_keys, fetch_tensor_table, lookup_values, string_eq


def execute_q11(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_nationkey"], device)
    nation = fetch_tensor_table(con, "nation", ["n_nationkey", "n_name"], device)

    germany = nation.columns["n_nationkey"][string_eq(nation, "n_name", "GERMANY")][0]
    supplier_nation = lookup_values(supplier.columns["s_suppkey"], supplier.columns["s_nationkey"], partsupp.columns["ps_suppkey"])
    mask = supplier_nation == germany
    value = partsupp.columns["ps_supplycost"][mask] * partsupp.columns["ps_availqty"][mask]
    keys, sums = aggregate_sum_by_keys([partsupp.columns["ps_partkey"][mask]], value)
    threshold = float(value.sum().cpu().item()) * 0.0001
    rows = [
        {"ps_partkey": int(keys[i, 0].item()), "value": float(sums[i].cpu().item())}
        for i in range(int(keys.shape[0]))
        if float(sums[i].cpu().item()) > threshold
    ]
    return sorted(rows, key=lambda row: -row["value"])
