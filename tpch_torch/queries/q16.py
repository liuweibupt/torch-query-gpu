"""TPC-H Q16 execution on PyTorch tensors."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.substrait import UnsupportedPlanError


def execute_q16(con: duckdb.DuckDBPyConnection, device: str = "cpu") -> list[dict[str, Any]]:
    raise UnsupportedPlanError("TPC-H Q16 logical-plan PyTorch executor is not implemented yet")
