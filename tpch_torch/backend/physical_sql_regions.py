"""SQL output-region helpers for physical-plan column pruning."""

from __future__ import annotations

import re

from tpch_torch.backend.physical_types import PhysicalTable


def output_requires_column(source_sql: str, table: PhysicalTable, column: str) -> bool:
    """Return whether user-visible SQL clauses reference a join-key column."""

    regions = _output_regions(source_sql)
    if not regions:
        return True
    if _selects_wildcard(regions[0]):
        return True
    column_name = _unqualified(column)
    qualified_refs = _qualified_column_refs(regions, column_name)
    if qualified_refs:
        return _table_matches_qualified_refs(_table_names_for_column(table, column), qualified_refs)
    return any(_contains_unqualified_column(region, column_name) for region in regions)


def _output_regions(sql: str) -> tuple[str, ...]:
    normalized = sql.strip().rstrip(";")
    regions = [_clause_between(normalized, "SELECT", ("FROM",))]
    for clause, stops in (
        ("GROUP BY", ("HAVING", "ORDER BY", "LIMIT")),
        ("HAVING", ("ORDER BY", "LIMIT")),
        ("ORDER BY", ("LIMIT",)),
    ):
        region = _clause_between(normalized, clause, stops)
        if region:
            regions.append(region)
    return tuple(region for region in regions if region)


def _clause_between(sql: str, clause: str, stop_clauses: tuple[str, ...]) -> str:
    start = _find_top_level_keyword(sql, clause, 0)
    if start < 0:
        return ""
    body_start = start + len(clause)
    end_positions = [
        position
        for stop in stop_clauses
        if (position := _find_top_level_keyword(sql, stop, body_start)) >= 0
    ]
    body_end = min(end_positions) if end_positions else len(sql)
    return sql[body_start:body_end].strip()


def _find_top_level_keyword(sql: str, keyword: str, start: int) -> int:
    depth = 0
    in_quote = False
    for index in range(start, len(sql)):
        char = sql[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0 and _keyword_at(sql, keyword, index):
            return index
    return -1


def _keyword_at(sql: str, keyword: str, index: int) -> bool:
    end = index + len(keyword)
    if sql[index:end].upper() != keyword.upper():
        return False
    before = sql[index - 1] if index > 0 else " "
    after = sql[end] if end < len(sql) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _qualified_column_refs(regions: tuple[str, ...], column: str) -> tuple[str, ...]:
    pattern = re.compile(rf"\b([A-Za-z_][\w]*)\s*\.\s*\"?{re.escape(column)}\"?\b", re.I)
    refs = []
    for region in regions:
        refs.extend(match.group(1) for match in pattern.finditer(region))
    return tuple(refs)


def _table_matches_qualified_refs(table_names: tuple[str, ...], refs: tuple[str, ...]) -> bool:
    candidates = {name.lower().rsplit(".", 1)[-1] for name in table_names}
    return any(ref.lower() in candidates for ref in refs)


def _table_names_for_column(table: PhysicalTable, column: str) -> tuple[str, ...]:
    suffix = f".{_unqualified(column)}"
    names = [
        key.rsplit(".", 1)[0]
        for key, value in table.columns.items()
        if value is table.columns[column] and key.endswith(suffix)
    ]
    names.append(table.name)
    return tuple(dict.fromkeys(names))


def _contains_unqualified_column(region: str, column: str) -> bool:
    return re.search(rf"(?<!\.)\b\"?{re.escape(column)}\"?\b", region, re.I) is not None


def _selects_wildcard(select_region: str) -> bool:
    return re.search(r"(^|,)\s*(?:[A-Za-z_][\w]*\s*\.\s*)?\*", select_region) is not None


def _unqualified(expr: str) -> str:
    return expr.replace('"', "").strip().rsplit(".", 1)[-1]
