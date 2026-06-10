"""PyTorch backend for executing TQP plans."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.ir import TQPPlan
from tpch_torch.queries.q01 import execute_q1
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.substrait import compile_q1_substrait_plan
from tpch_torch.substrait import (
    Q1_GROUP_KEYS,
    Q1_ORDER_KEYS,
    Q1_REQUIRED_COLUMNS,
    Q1_SHIPDATE_CUTOFF_YYYYMMDD,
    Q1Plan,
)


class PyTorchBackend:
    """Execute internal TQP plans with existing PyTorch tensor query kernels."""

    def execute(self, con: duckdb.DuckDBPyConnection, plan: TQPPlan, device: str = "cpu") -> list[dict[str, Any]]:
        if plan.query_id is None:
            if plan.generic_plan is None:
                detail = plan.generic_error or "generic SQL plan is missing executable operator plan"
                raise UnsupportedPlanError(f"generic SQL is not executable by PyTorch backend: {detail}")
            return execute_generic_sql_plan(con, plan.generic_plan, device=device)
        if plan.query_id == 1:
            q1_plan = _compile_q1_plan(plan.plan_json)
            from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

            return execute_q1(fetch_lineitem_tensor_table(con, device=device), q1_plan)
        if plan.query_id == 6:
            from tpch_torch.queries.q06 import execute_q6

            return execute_q6(con, device=device)
        module_name = _EXECUTOR_BY_QUERY.get(plan.query_id)
        if module_name is None:
            raise UnsupportedPlanError(f"TPC-H Q{plan.query_id} exported to frontend but has no PyTorch executor yet")
        module = __import__(f"tpch_torch.queries.{module_name}", fromlist=[f"execute_q{plan.query_id}"])
        return getattr(module, f"execute_q{plan.query_id}")(con, device=device)


_EXECUTOR_BY_QUERY = {
    2: "q02",
    3: "q03",
    4: "q04",
    5: "q05",
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
