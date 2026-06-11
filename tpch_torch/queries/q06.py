"""TPC-H Q6 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.compressed import PlainMask, RLEMask, mask_and, mask_to_index, plain_to_rle
from tpch_torch.relational import fetch_tensor_table
from tpch_torch.storage import TensorTable

LINEITEM_COLUMNS = (
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_shipdate",
)


def execute_q6(
    con: duckdb.DuckDBPyConnection,
    device: str = "cpu",
    use_compressed_masks: bool = False,
) -> list[dict[str, Any]]:
    table = fetch_tensor_table(con, "lineitem", LINEITEM_COLUMNS, device=device)
    if use_compressed_masks:
        return [_execute_q6_compressed_mask_row(table)]
    return [_execute_q6_plain_mask_row(table)]


def _execute_q6_plain_mask_row(table: TensorTable) -> dict[str, Any]:
    columns = table.columns
    mask = _q6_plain_mask(table)
    revenue = (columns["l_extendedprice"][mask] * columns["l_discount"][mask]).sum()
    return {"revenue": _to_float(revenue)}


def _execute_q6_compressed_mask_row(table: TensorTable) -> dict[str, Any]:
    columns = table.columns
    mask = _q6_compressed_mask(table)
    positions = mask_to_index(mask)
    extended_price = columns["l_extendedprice"].index_select(0, positions)
    discount = columns["l_discount"].index_select(0, positions)
    return {"revenue": _to_float((extended_price * discount).sum())}


def _q6_plain_mask(table: TensorTable) -> torch.Tensor:
    columns = table.columns
    return (
        (columns["l_shipdate"] >= 19940101)
        & (columns["l_shipdate"] < 19950101)
        & (columns["l_discount"] >= 0.05)
        & (columns["l_discount"] <= 0.07)
        & (columns["l_quantity"] < 24.0)
    )


def _q6_compressed_mask(table: TensorTable):
    columns = table.columns
    date_mask = RLEMask(
        plain_to_rle((columns["l_shipdate"] >= 19940101) & (columns["l_shipdate"] < 19950101)),
        row_count=len(table),
    )
    discount_mask = PlainMask((columns["l_discount"] >= 0.05) & (columns["l_discount"] <= 0.07))
    quantity_mask = PlainMask(columns["l_quantity"] < 24.0)
    return mask_and(mask_and(date_mask, discount_mask), quantity_mask)


def _to_float(value: torch.Tensor) -> float:
    return float(value.cpu().item())
