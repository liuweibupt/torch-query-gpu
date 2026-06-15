"""Physical-interpreter coverage probing for TPC-H queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb

from tpch_torch.backend.physical import execute_physical_plan
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.sql import get_tpch_query


@dataclass(frozen=True)
class PhysicalCoverageRecord:
    """One TPC-H physical-interpreter probe result."""

    query_id: int
    supported: bool
    reason: str = ""


def probe_tpch_physical_coverage(
    con: duckdb.DuckDBPyConnection,
    query_ids: Iterable[int],
    *,
    device: str = "cpu",
) -> tuple[PhysicalCoverageRecord, ...]:
    """Run TPC-H queries directly through the physical interpreter."""

    records: list[PhysicalCoverageRecord] = []
    for query_id in query_ids:
        records.append(_probe_one_query(con, query_id, device))
    return tuple(records)


def _probe_one_query(
    con: duckdb.DuckDBPyConnection,
    query_id: int,
    device: str,
) -> PhysicalCoverageRecord:
    try:
        plan = compile_sirius_plan(con, get_tpch_query(con, query_id))
        if plan.operator_graph is None:
            return PhysicalCoverageRecord(query_id, False, "missing operator graph")
        execute_physical_plan(con, plan.operator_graph, device=device)
    except UnsupportedPlanError as exc:
        return PhysicalCoverageRecord(query_id, False, str(exc))
    except (duckdb.Error, KeyError, ValueError, TypeError, RuntimeError) as exc:
        return PhysicalCoverageRecord(query_id, False, f"{type(exc).__name__}: {exc}")
    return PhysicalCoverageRecord(query_id, True)
