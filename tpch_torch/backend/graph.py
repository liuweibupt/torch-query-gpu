"""PyTorch execution for explicit TQP operator graphs."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.queries.q01 import execute_q1
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
            return _execute_compiled_tpch_node(con, plan, root, device, use_compressed_masks)
        if plan.query_id == 1:
            from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

            q1_plan = _compile_q1_plan(plan.plan_json)
            return execute_q1(fetch_lineitem_tensor_table(con, device=device), q1_plan)
        if plan.query_id == 6:
            from tpch_torch.queries.q06 import execute_q6

            return execute_q6(con, device=device, use_compressed_masks=use_compressed_masks)
        raise UnsupportedPlanError(
            f"TQP graph for TPC-H Q{plan.query_id} has no executable root: {root.kind}"
        )


def _execute_compiled_tpch_node(
    con: duckdb.DuckDBPyConnection,
    plan: TQPPlan,
    node: TQPOperatorNode,
    device: str,
    use_compressed_masks: bool,
) -> list[dict[str, Any]]:
    query_id = int(node.metadata.get("query_id", plan.query_id or -1))
    if query_id == 1:
        from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

        q1_plan = _compile_q1_plan(plan.plan_json)
        return execute_q1(fetch_lineitem_tensor_table(con, device=device), q1_plan)
    module_name = _EXECUTOR_BY_QUERY.get(query_id)
    if module_name is None:
        raise UnsupportedPlanError(f"TQP graph references TPC-H Q{query_id}, but no PyTorch executor is registered")
    module = __import__(f"tpch_torch.queries.{module_name}", fromlist=[f"execute_q{query_id}"])
    executor = getattr(module, f"execute_q{query_id}")
    if query_id == 6:
        return executor(con, device=device, use_compressed_masks=use_compressed_masks)
    return executor(con, device=device)


_EXECUTOR_BY_QUERY = {
    2: "q02",
    3: "q03",
    4: "q04",
    5: "q05",
    6: "q06",
    7: "q07",
    8: "q08",
    9: "q09",
    10: "q10",
    11: "q11",
    12: "q12",
    13: "q13",
    14: "q14",
    15: "q15",
    16: "q16",
    17: "q17",
    18: "q18",
    19: "q19",
    20: "q20",
    21: "q21",
    22: "q22",
}


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
