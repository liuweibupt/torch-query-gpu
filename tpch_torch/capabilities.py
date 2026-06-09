"""Native DuckDB Substrait capability reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import duckdb

from tpch_torch.duckdb_bridge import DuckDBSubstraitError, export_substrait_json
from tpch_torch.runner import is_query_executor_supported
from tpch_torch.sql import get_tpch_query


@dataclass(frozen=True)
class QueryExportStatus:
    """Native Substrait export status for one TPC-H query."""

    query_id: int
    export_ok: bool
    error_type: str | None
    error_message: str | None
    executor_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "export_ok": self.export_ok,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "executor_supported": self.executor_supported,
        }


def probe_tpch_substrait_exports(
    con: duckdb.DuckDBPyConnection, query_ids: Sequence[int]
) -> list[QueryExportStatus]:
    """Probe native DuckDB Substrait export for original TPC-H SQL."""

    statuses: list[QueryExportStatus] = []
    for query_id in query_ids:
        statuses.append(_probe_one_query(con, query_id))
    return statuses


def _probe_one_query(con: duckdb.DuckDBPyConnection, query_id: int) -> QueryExportStatus:
    sql = get_tpch_query(con, query_id)
    try:
        export_substrait_json(con, sql)
    except DuckDBSubstraitError as exc:
        return QueryExportStatus(
            query_id=query_id,
            export_ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc).splitlines()[0],
            executor_supported=is_query_executor_supported(query_id),
        )
    return QueryExportStatus(
        query_id=query_id,
        export_ok=True,
        error_type=None,
        error_message=None,
        executor_supported=is_query_executor_supported(query_id),
    )
