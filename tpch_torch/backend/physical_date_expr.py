"""Date expression helpers for physical-plan evaluation."""

from __future__ import annotations

import re


def parse_extract_year(expr: str) -> str | None:
    """Return the source expression for EXTRACT(year FROM source)."""

    match = re.fullmatch(r"extract\s*\(\s*year\s+FROM\s+(.+)\)", expr, re.I | re.S)
    return match.group(1).strip() if match is not None else None


def parse_scalar_subquery_guard(expr: str) -> str | None:
    """Return the value arm for DuckDB scalar-subquery cardinality guards."""

    match = re.match(r"CASE\s+WHEN\s+\(\(#\d+\s*>\s*1\)\)\s+THEN\s+\(error\(", expr, re.I | re.S)
    if match is None:
        return None
    else_match = re.search(r"\bELSE\s+(#\d+)\s+END\s*$", expr, re.I | re.S)
    return else_match.group(1) if else_match is not None else "#0"
