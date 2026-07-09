"""PyTorch execution for explicit TQP operator graphs."""

from __future__ import annotations

from typing import Any

import duckdb
import torch

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.backend.physical_chunked import ScanChunkConfig, execute_chunked_physical_plan
from tpch_torch.backend.physical import execute_physical_plan
from tpch_torch.backend.physical_partitionable import (
    PartitionConfig,
    execute_partitionable_physical_plan,
)
from tpch_torch.compressed import PlainMask, RLEMask, mask_and, mask_to_index, plain_to_rle
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph
from tpch_torch.storage import TensorTable

_PHYSICAL_TPCH_QUERIES = frozenset(range(1, 23))


class PyTorchGraphExecutor:
    """Execute a frontend-lowered TQP operator graph with PyTorch tensors."""

    def execute(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        *,
        device: str = "cpu",
        use_compressed_masks: bool = False,
        partition_config: PartitionConfig | None = None,
        scan_chunk_config: ScanChunkConfig | None = None,
    ) -> list[dict[str, Any]]:
        graph = plan.operator_graph
        if graph is None:
            if partition_config is not None:
                raise UnsupportedPlanError("partitionable execution requires a physical operator graph")
            if scan_chunk_config is not None:
                raise UnsupportedPlanError("scan chunk execution requires a physical operator graph")
            if plan.generic_plan is not None:
                return execute_generic_sql_plan(con, plan.generic_plan, device=device)
            raise UnsupportedPlanError("TQP operator graph is required for PyTorch graph execution")
        if partition_config is not None and scan_chunk_config is not None:
            raise UnsupportedPlanError("partitionable execution cannot be combined with scan chunk execution")
        if partition_config is not None and use_compressed_masks:
            raise UnsupportedPlanError("partitionable execution cannot be combined with compressed mask mode")
        if scan_chunk_config is not None and use_compressed_masks:
            raise UnsupportedPlanError("scan chunk execution cannot be combined with compressed mask mode")
        if plan.query_id is None:
            if partition_config is not None:
                return execute_partitionable_physical_plan(con, graph, partition_config, device=device)
            if scan_chunk_config is not None:
                return execute_chunked_physical_plan(con, graph, scan_chunk_config, device=device)
            return execute_physical_plan(con, graph, device=device)
        return self._execute_tpch_graph(
            con,
            plan,
            graph,
            device,
            use_compressed_masks,
            partition_config,
            scan_chunk_config,
        )

    def _execute_generic_plan(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        device: str,
    ) -> list[dict[str, Any]]:
        if plan.generic_plan is not None:
            return execute_generic_sql_plan(con, plan.generic_plan, device=device)
        if plan.operator_graph is not None:
            return execute_physical_plan(con, plan.operator_graph, device=device)
        detail = plan.generic_error or "generic SQL plan is missing executable operator plan"
        raise UnsupportedPlanError(f"generic SQL is not executable by PyTorch backend: {detail}")

    def _execute_tpch_graph(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        graph: TQPOperatorGraph,
        device: str,
        use_compressed_masks: bool,
        partition_config: PartitionConfig | None = None,
        scan_chunk_config: ScanChunkConfig | None = None,
    ) -> list[dict[str, Any]]:
        root = graph.root
        if root.kind == OperatorKind.COMPILED_TPCH:
            raise UnsupportedPlanError("compiled TPC-H compatibility roots are no longer executable")
        if partition_config is not None:
            return execute_partitionable_physical_plan(con, graph, partition_config, device=device)
        if scan_chunk_config is not None:
            return execute_chunked_physical_plan(con, graph, scan_chunk_config, device=device)
        if plan.query_id == 6 and use_compressed_masks:
            return _execute_q6_graph(con, device, use_compressed_masks)
        if plan.query_id in _PHYSICAL_TPCH_QUERIES:
            return execute_physical_plan(con, graph, device=device)
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


def _to_float(value: torch.Tensor) -> float:
    return float(value.cpu().item())
