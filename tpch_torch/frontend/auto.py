"""Auto frontend selection for compatibility."""

from __future__ import annotations

import duckdb

from tpch_torch.duckdb_bridge import DuckDBSubstraitError
from tpch_torch.frontend.sirius import compile_sirius_plan
from tpch_torch.frontend.substrait import compile_substrait_plan
from tpch_torch.ir import TQPPlan


def compile_auto_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    """Try strict Substrait first, then Sirius-like frontend on export failure."""

    try:
        return compile_substrait_plan(con, sql)
    except DuckDBSubstraitError:
        return compile_sirius_plan(con, sql)
