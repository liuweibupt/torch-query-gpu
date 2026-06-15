"""Small SQL text helpers for final physical projection aliases."""

from __future__ import annotations

import re

_AGGREGATE_NAMES = frozenset({"sum", "avg", "min", "max", "count"})


def select_expressions_by_alias(sql: str) -> dict[str, str]:
    """Map SELECT aliases to their source expressions without executing SQL."""

    mapping: dict[str, str] = {}
    for select_body in _select_bodies(sql):
        for item in _split_csv(select_body):
            expression, alias = _split_alias(item)
            if alias is not None:
                mapping[alias] = expression
    return mapping


def replace_aggregate_calls_with_refs(expression: str) -> str:
    """Replace aggregate calls in source SELECT expression by #0/#1 physical refs."""

    pieces: list[str] = []
    index = 0
    aggregate_index = 0
    while index < len(expression):
        match = _aggregate_call_at(expression, index)
        if match is None:
            pieces.append(expression[index])
            index += 1
            continue
        start, end = match
        pieces.append(f"#{aggregate_index}")
        aggregate_index += 1
        index = end
    return "".join(pieces)


def _select_body(sql: str) -> str:
    normalized = sql.strip().rstrip(";")
    match = re.match(r"\s*SELECT\s+", normalized, re.I)
    if match is None:
        return ""
    start = match.end()
    end = _find_top_level_keyword(normalized, "FROM", start)
    return normalized[start:end].strip() if end >= 0 else normalized[start:].strip()


def _select_bodies(sql: str) -> tuple[str, ...]:
    bodies = []
    normalized = sql.strip().rstrip(";")
    for start in _keyword_positions(normalized, "SELECT"):
        end = _find_top_level_keyword(normalized, "FROM", start + len("SELECT"))
        if end >= 0:
            bodies.append(normalized[start + len("SELECT") : end].strip())
    return tuple(bodies) or (_select_body(sql),)


def _keyword_positions(sql: str, keyword: str):
    for index in range(len(sql)):
        if _keyword_at(sql, keyword, index):
            yield index


def _split_alias(item: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(.+?)\s+AS\s+([A-Za-z_][\w]*)", item.strip(), re.I | re.S)
    if match is None:
        return item.strip(), None
    return match.group(1).strip(), match.group(2)


def _aggregate_call_at(expression: str, index: int) -> tuple[int, int] | None:
    name_match = re.match(r"[A-Za-z_][\w]*", expression[index:])
    if name_match is None:
        return None
    name = name_match.group(0)
    if name.lower() not in _AGGREGATE_NAMES:
        return None
    open_index = index + len(name)
    while open_index < len(expression) and expression[open_index].isspace():
        open_index += 1
    if open_index >= len(expression) or expression[open_index] != "(":
        return None
    close_index = _matching_paren(expression, open_index)
    return index, close_index + 1


def _split_csv(raw: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    for index, char in enumerate(raw):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        elif not in_quote and depth == 0 and char == ",":
            parts.append(raw[start:index].strip())
            start = index + 1
    parts.append(raw[start:].strip())
    return tuple(part for part in parts if part)


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


def _matching_paren(expression: str, open_index: int) -> int:
    depth = 0
    in_quote = False
    for index in range(open_index, len(expression)):
        char = expression[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unbalanced aggregate expression: {expression}")
