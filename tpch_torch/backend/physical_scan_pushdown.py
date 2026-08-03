"""Scan predicate pushdown planning for batch scan sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import duckdb


@dataclass(frozen=True)
class ScanFilterPushdown:
    """Partition scan filters into data-source predicates and tensor residuals."""

    pushed_filters: tuple[str, ...]
    residual_filters: tuple[str, ...]


def plan_scan_filter_pushdown(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    filters: Sequence[str],
) -> ScanFilterPushdown:
    """Return filters safe to push into DuckDB scan plus tensor residuals.

    Pushdown is an optimization of the scan source.  Filters that are optional,
    empty, positional, or rejected by DuckDB remain outside the pushed set; the
    caller can keep evaluating residual predicates in PyTorch.
    """

    pushed: list[str] = []
    residual: list[str] = []
    for raw_filter in filters:
        filter_text = raw_filter.strip()
        if not filter_text or filter_text.lower().startswith("optional:"):
            continue
        if _can_push_filter(con, table_name, filter_text):
            pushed.append(filter_text)
            continue
        residual.append(filter_text)
    return ScanFilterPushdown(tuple(pushed), tuple(residual))


def where_clause(filters: Sequence[str]) -> str:
    """Return a DuckDB WHERE clause for already validated scan filters."""

    if not filters:
        return ""
    predicates = " and ".join(f"({filter_text})" for filter_text in filters)
    return f" where {predicates}"


def _can_push_filter(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    filter_text: str,
) -> bool:
    if "#" in filter_text:
        return False
    try:
        con.execute(f"select 1 from {table_name} where {filter_text} limit 0")
    except duckdb.Error:
        return False
    return True
