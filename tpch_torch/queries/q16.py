"""TPC-H Q16 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.relational import decode, fetch_tensor_table, lookup_values, string_eq, string_startswith

Q16_SIZES = (49, 14, 23, 45, 19, 3, 36, 9)


def execute_q16(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    partsupp = fetch_tensor_table(con, "partsupp", ["ps_partkey", "ps_suppkey"], device)
    part = fetch_tensor_table(con, "part", ["p_partkey", "p_brand", "p_type", "p_size"], device)
    supplier = fetch_tensor_table(con, "supplier", ["s_suppkey", "s_comment"], device)

    bad_suppkeys = _complaint_supplier_keys(supplier)
    brand = lookup_values(part.columns["p_partkey"], part.columns["p_brand"], partsupp.columns["ps_partkey"])
    part_type = lookup_values(part.columns["p_partkey"], part.columns["p_type"], partsupp.columns["ps_partkey"])
    size = lookup_values(part.columns["p_partkey"], part.columns["p_size"], partsupp.columns["ps_partkey"])
    valid = (
        (brand >= 0)
        & (part_type >= 0)
        & ~torch.isin(partsupp.columns["ps_suppkey"], bad_suppkeys)
        & (brand != int(part.columns["p_brand"][string_eq(part, "p_brand", "Brand#45")][0].item()))
        & ~_type_startswith(part, part_type, "MEDIUM POLISHED")
        & torch.isin(size, torch.tensor(Q16_SIZES, dtype=size.dtype, device=device))
    )
    groups: dict[tuple[int, int, int], set[int]] = {}
    host_brand = brand[valid].cpu().tolist()
    host_type = part_type[valid].cpu().tolist()
    host_size = size[valid].cpu().tolist()
    host_suppkey = partsupp.columns["ps_suppkey"][valid].cpu().tolist()
    for brand_id, type_id, part_size, suppkey in zip(host_brand, host_type, host_size, host_suppkey):
        groups.setdefault((int(brand_id), int(type_id), int(part_size)), set()).add(int(suppkey))
    rows = [
        {
            "p_brand": decode(part, "p_brand", brand_id),
            "p_type": decode(part, "p_type", type_id),
            "p_size": part_size,
            "supplier_cnt": len(suppkeys),
        }
        for (brand_id, type_id, part_size), suppkeys in groups.items()
    ]
    return sorted(rows, key=lambda row: (-row["supplier_cnt"], row["p_brand"], row["p_type"], row["p_size"]))


def _complaint_supplier_keys(supplier):
    matching_ids = [
        index
        for index, comment in enumerate(supplier.dictionaries["s_comment"])
        if "Customer" in comment and "Complaints" in comment and comment.index("Customer") < comment.rindex("Complaints")
    ]
    if not matching_ids:
        return torch.empty(0, dtype=supplier.columns["s_suppkey"].dtype, device=supplier.columns["s_suppkey"].device)
    mask = torch.isin(
        supplier.columns["s_comment"],
        torch.tensor(matching_ids, dtype=supplier.columns["s_comment"].dtype, device=supplier.columns["s_comment"].device),
    )
    return supplier.columns["s_suppkey"][mask]


def _type_startswith(part, encoded_type: torch.Tensor, prefix: str) -> torch.Tensor:
    matching_ids = [index for index, value in enumerate(part.dictionaries["p_type"]) if value.startswith(prefix)]
    return torch.isin(encoded_type, torch.tensor(matching_ids, dtype=encoded_type.dtype, device=encoded_type.device))
