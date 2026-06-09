"""Native DuckDB Substrait capability reporting."""

from __future__ import annotations

from dataclasses import dataclass


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
