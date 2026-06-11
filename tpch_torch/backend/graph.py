"""PyTorch execution for explicit TQP operator graphs."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.compressed import PlainMask, RLEMask, mask_and, mask_to_index, plain_to_rle
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.storage import TensorTable
from tpch_torch.substrait import (
    Q1_GROUP_KEYS,
    Q1_ORDER_KEYS,
    Q1_REQUIRED_COLUMNS,
    Q1_SHIPDATE_CUTOFF_YYYYMMDD,
    Q1Plan,
    compile_q1_substrait_plan,
)


class PyTorchGraphExecutor:
    """Execute a frontend-lowered TQP operator graph with PyTorch tensors."""

    def execute(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        *,
        device: str = "cpu",
        use_compressed_masks: bool = False,
    ) -> list[dict[str, Any]]:
        graph = plan.operator_graph
        if graph is None:
            raise UnsupportedPlanError("TQP operator graph is required for PyTorch graph execution")
        if plan.query_id is None:
            return self._execute_generic_plan(con, plan, device)
        return self._execute_tpch_graph(con, plan, graph, device, use_compressed_masks)

    def _execute_generic_plan(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        device: str,
    ) -> list[dict[str, Any]]:
        if plan.generic_plan is None:
            detail = plan.generic_error or "generic SQL plan is missing executable operator plan"
            raise UnsupportedPlanError(f"generic SQL is not executable by PyTorch backend: {detail}")
        return execute_generic_sql_plan(con, plan.generic_plan, device=device)

    def _execute_tpch_graph(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        graph: TQPOperatorGraph,
        device: str,
        use_compressed_masks: bool,
    ) -> list[dict[str, Any]]:
        root = graph.root
        if root.kind == OperatorKind.COMPILED_TPCH:
            raise UnsupportedPlanError("compiled TPC-H compatibility roots are no longer executable")
        if plan.query_id == 1:
            return _execute_q1_graph(con, plan, device)
        if plan.query_id == 6:
            return _execute_q6_graph(con, device, use_compressed_masks)
        return _execute_tpch_graph_query(con, plan, device)


def _execute_tpch_graph_query(
    con: duckdb.DuckDBPyConnection,
    plan: TQPPlan,
    device: str,
) -> list[dict[str, Any]]:
    if plan.query_id is None:
        raise UnsupportedPlanError("TPC-H graph query execution requires query_id")
    module = __import__(
        f"tpch_torch.backend.tpch_graph_q{plan.query_id:02d}",
        fromlist=[f"execute_q{plan.query_id}_graph"],
    )
    executor = getattr(module, f"execute_q{plan.query_id}_graph")
    return executor(con, device=device)


def _execute_q1_graph(
    con: duckdb.DuckDBPyConnection,
    plan: TQPPlan,
    device: str,
) -> list[dict[str, Any]]:
    from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

    return _execute_q1_tensor_graph(fetch_lineitem_tensor_table(con, device=device), _compile_q1_plan(plan.plan_json))


def _execute_q1_tensor_graph(table: TensorTable, plan: Q1Plan) -> list[dict[str, Any]]:
    table.require_columns(plan.required_columns)
    mask = table.columns["l_shipdate"] <= plan.shipdate_cutoff_yyyymmdd
    selected_rows = torch.nonzero(mask).flatten()
    if selected_rows.numel() == 0:
        return []
    columns = {
        name: table.columns[name].index_select(0, selected_rows)
        for name in plan.required_columns
        if name != "l_shipdate"
    }
    status_count = len(table.dictionaries["l_linestatus"])
    flag_count = len(table.dictionaries["l_returnflag"])
    group_ids = (columns["l_returnflag"].to(dtype=torch.int64) * status_count) + columns[
        "l_linestatus"
    ].to(dtype=torch.int64)
    group_count = flag_count * status_count
    aggregates = _q1_aggregates(columns, group_ids, group_count)
    non_empty_group_ids = torch.nonzero(aggregates["count_order"] > 0).flatten()
    keys = torch.stack((non_empty_group_ids // status_count, non_empty_group_ids % status_count), dim=1)
    compacted = {name: tensor[non_empty_group_ids] for name, tensor in aggregates.items()}
    return _format_q1_rows(table, keys, compacted)


def _q1_aggregates(
    columns: dict[str, torch.Tensor],
    group_ids: torch.Tensor,
    group_count: int,
) -> dict[str, torch.Tensor]:
    quantity = columns["l_quantity"]
    extendedprice = columns["l_extendedprice"]
    discount = columns["l_discount"]
    tax = columns["l_tax"]
    discounted_price = extendedprice * (1.0 - discount)
    charge = discounted_price * (1.0 + tax)
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
    rows: list[dict[str, Any]] = []
    host_keys = keys.cpu()
    host_aggregates = {name: tensor.cpu() for name, tensor in aggregates.items()}
    result_columns = (
        "sum_qty",
        "sum_base_price",
        "sum_disc_price",
        "sum_charge",
        "avg_qty",
        "avg_price",
        "avg_disc",
        "count_order",
    )
    for index in range(int(host_keys.shape[0])):
        row = {
            "l_returnflag": table.decode_value("l_returnflag", int(host_keys[index, 0])),
            "l_linestatus": table.decode_value("l_linestatus", int(host_keys[index, 1])),
        }
        for name in result_columns:
            value = host_aggregates[name][index].item()
            row[name] = int(value) if name == "count_order" else float(value)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["l_returnflag"], row["l_linestatus"]))


def _execute_q6_graph(
    con: duckdb.DuckDBPyConnection,
    device: str,
    use_compressed_masks: bool,
) -> list[dict[str, Any]]:
    from tpch_torch.relational import fetch_tensor_table

    table = fetch_tensor_table(
        con,
        "lineitem",
        ("l_quantity", "l_extendedprice", "l_discount", "l_shipdate"),
        device=device,
    )
    if use_compressed_masks:
        mask = _q6_compressed_mask(table)
        positions = mask_to_index(mask)
        extended_price = table.columns["l_extendedprice"].index_select(0, positions)
        discount = table.columns["l_discount"].index_select(0, positions)
        return [{"revenue": _to_float((extended_price * discount).sum())}]
    mask = _q6_plain_mask(table)
    revenue = (table.columns["l_extendedprice"][mask] * table.columns["l_discount"][mask]).sum()
    return [{"revenue": _to_float(revenue)}]


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


def _compile_q1_plan(plan_json: dict[str, Any] | None) -> Q1Plan:
    if plan_json:
        return compile_q1_substrait_plan(plan_json)
    return Q1Plan(
        table_name="lineitem",
        shipdate_cutoff_yyyymmdd=Q1_SHIPDATE_CUTOFF_YYYYMMDD,
        required_columns=Q1_REQUIRED_COLUMNS,
        group_keys=Q1_GROUP_KEYS,
        order_keys=Q1_ORDER_KEYS,
    )


def _to_float(value: torch.Tensor) -> float:
    return float(value.cpu().item())
