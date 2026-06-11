"""TPC-H Q22 graph-query execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import aggregate_count_by_keys, aggregate_sum_by_keys, fetch_tensor_table

Q22_COUNTRY_CODES = ("13", "31", "23", "29", "30", "18", "17")


def execute_q22_graph(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    customer = fetch_tensor_table(con, "customer", ["c_custkey", "c_phone", "c_acctbal"], device)
    orders = fetch_tensor_table(con, "orders", ["o_custkey"], device)

    country_ids = _country_ids(customer, device)
    positive_in_country = (country_ids >= 0) & (customer.columns["c_acctbal"] > 0.0)
    avg_acctbal = customer.columns["c_acctbal"][positive_in_country].mean()
    has_order = torch.isin(customer.columns["c_custkey"], torch.unique(orders.columns["o_custkey"]))
    mask = (country_ids >= 0) & (customer.columns["c_acctbal"] > avg_acctbal) & ~has_order
    if not bool(mask.any().item()):
        return []
    keys, counts = aggregate_count_by_keys([country_ids[mask]])
    _, sums = aggregate_sum_by_keys([country_ids[mask]], customer.columns["c_acctbal"][mask])
    rows = [
        {
            "cntrycode": Q22_COUNTRY_CODES[int(keys[index, 0].item())],
            "numcust": int(counts[index].cpu().item()),
            "totacctbal": float(sums[index].cpu().item()),
        }
        for index in range(int(keys.shape[0]))
    ]
    return sorted(rows, key=lambda row: row["cntrycode"])


def _country_ids(customer, device: str) -> torch.Tensor:
    mapping = torch.full(
        (len(customer.dictionaries["c_phone"]),),
        -1,
        dtype=torch.int64,
        device=device,
    )
    code_to_id = {code: index for index, code in enumerate(Q22_COUNTRY_CODES)}
    for phone_id, phone in enumerate(customer.dictionaries["c_phone"]):
        code_id = code_to_id.get(phone[:2])
        if code_id is not None:
            mapping[phone_id] = code_id
    return mapping[customer.columns["c_phone"].to(dtype=torch.int64)]
