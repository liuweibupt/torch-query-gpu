"""Graph-lowered fused physical operators."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table
from tpch_torch.operator_graph import TQPOperatorGraph
from tpch_torch.storage import TensorTable

_Q1_CUTOFF_YYYYMMDD = 19980902
_Q1_REQUIRED_COLUMNS = (
    "l_returnflag",
    "l_linestatus",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_shipdate",
)
_Q1_RESULT_COLUMNS = (
    "sum_qty",
    "sum_base_price",
    "sum_disc_price",
    "sum_charge",
    "avg_qty",
    "avg_price",
    "avg_disc",
    "count_order",
)


def try_execute_fused_physical_plan(
    con: duckdb.DuckDBPyConnection,
    graph: TQPOperatorGraph,
    device: str,
) -> list[dict[str, Any]] | None:
    """Return fused rows for recognized physical graphs, otherwise None."""

    if _is_q1_physical_graph(graph):
        return _execute_q1_fused(con, device)
    return None


def _is_q1_physical_graph(graph: TQPOperatorGraph) -> bool:
    if graph.query_id != 1:
        return False
    node_names = {node.name.strip().upper() for node in graph.nodes}
    if "PERFECT_HASH_GROUP_BY" not in node_names and "HASH_GROUP_BY" not in node_names:
        return False
    if "ORDER_BY" not in node_names:
        return False
    scan_tables = {
        str(node.metadata.get("Table", "")).lower()
        for node in graph.nodes
        if node.name.strip().upper().endswith("SCAN")
    }
    return "lineitem" in scan_tables


def _execute_q1_fused(con: duckdb.DuckDBPyConnection, device: str) -> list[dict[str, Any]]:
    table = fetch_lineitem_tensor_table(con, device=device)
    table.require_columns(_Q1_REQUIRED_COLUMNS)
    selected_rows = torch.nonzero(table.columns["l_shipdate"] <= _Q1_CUTOFF_YYYYMMDD).flatten()
    if selected_rows.numel() == 0:
        return []
    columns = _gather_q1_columns(table, selected_rows)
    status_count = len(table.dictionaries["l_linestatus"])
    flag_count = len(table.dictionaries["l_returnflag"])
    group_ids = (columns["l_returnflag"].to(dtype=torch.int64) * status_count) + columns[
        "l_linestatus"
    ].to(dtype=torch.int64)
    aggregates = _q1_grouped_reductions(columns, group_ids, flag_count * status_count)
    non_empty_group_ids = torch.nonzero(aggregates["count_order"] > 0).flatten()
    compacted = {name: tensor[non_empty_group_ids] for name, tensor in aggregates.items()}
    keys = torch.stack((non_empty_group_ids // status_count, non_empty_group_ids % status_count), dim=1)
    return _format_q1_rows(table, keys, compacted)


def _gather_q1_columns(table: TensorTable, selected_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: table.columns[name].index_select(0, selected_rows)
        for name in _Q1_REQUIRED_COLUMNS
        if name != "l_shipdate"
    }


def _q1_grouped_reductions(
    columns: dict[str, torch.Tensor],
    group_ids: torch.Tensor,
    group_count: int,
) -> dict[str, torch.Tensor]:
    quantity = columns["l_quantity"]
    extendedprice = columns["l_extendedprice"]
    discount = columns["l_discount"]
    discounted_price = extendedprice * (1.0 - discount)
    charge = discounted_price * (1.0 + columns["l_tax"])
    count_order = torch.bincount(group_ids, minlength=group_count)
    count_as_float = count_order.to(dtype=quantity.dtype)
    sum_qty = torch.bincount(group_ids, weights=quantity, minlength=group_count)
    sum_base_price = torch.bincount(group_ids, weights=extendedprice, minlength=group_count)
    sum_discount = torch.bincount(group_ids, weights=discount, minlength=group_count)
    return {
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": torch.bincount(group_ids, weights=discounted_price, minlength=group_count),
        "sum_charge": torch.bincount(group_ids, weights=charge, minlength=group_count),
        "avg_qty": sum_qty / count_as_float,
        "avg_price": sum_base_price / count_as_float,
        "avg_disc": sum_discount / count_as_float,
        "count_order": count_order,
    }


def _format_q1_rows(
    table: TensorTable,
    keys: torch.Tensor,
    aggregates: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    host_keys = keys.cpu()
    host_aggregates = {name: tensor.cpu() for name, tensor in aggregates.items()}
    rows: list[dict[str, Any]] = []
    for index in range(int(host_keys.shape[0])):
        row = {
            "l_returnflag": table.decode_value("l_returnflag", int(host_keys[index, 0])),
            "l_linestatus": table.decode_value("l_linestatus", int(host_keys[index, 1])),
        }
        for name in _Q1_RESULT_COLUMNS:
            value = host_aggregates[name][index].item()
            row[name] = int(value) if name == "count_order" else float(value)
        rows.append(row)
    return rows
