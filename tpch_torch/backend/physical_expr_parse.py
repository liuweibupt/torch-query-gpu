"""Parsing helpers for DuckDB physical scalar expressions."""

from __future__ import annotations

import re
from typing import Any, Sequence

_NO_LITERAL = object()
_COMPARISON_OPERATORS = ("!~~", "~~", ">=", "<=", "!=", "<>", "=", ">", "<")


def _parse_literal(expr: str) -> Any:
    date_match = re.fullmatch(r"(?:DATE\s*)?'(?P<value>\d{4}-\d{2}-\d{2})'(?:::DATE)?", expr, re.I)
    if date_match:
        return int(date_match.group("value").replace("-", ""))
    if re.fullmatch(r"'[^']*'", expr):
        return expr[1:-1]
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    if re.fullmatch(r"-?\d+\.\d+", expr):
        return float(expr)
    if expr.upper() == "TRUE":
        return True
    if expr.upper() == "FALSE":
        return False
    return _NO_LITERAL


def _parse_case(expr: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"CASE\s+WHEN\s+(.+)\s+THEN\s+(.+)\s+ELSE\s+(.+)\s+END", expr, re.I)
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()


def _parse_cast(expr: str) -> str | None:
    match = re.fullmatch(r"CAST\s*\((.*)\)", expr, re.I | re.S)
    if match is None or not _balanced(match.group(1)):
        return None
    parts = _split_top_level_keyword(match.group(1), "AS")
    if len(parts) != 2:
        return None
    return parts[0]


def _parse_call(expr: str) -> tuple[str, str] | None:
    match = re.fullmatch(r'"?([A-Za-z_][\w]*)"?\s*\((.*)\)', expr, re.S)
    if match is None or not _balanced(match.group(2)):
        return None
    return match.group(1), match.group(2)


def _parse_in(expr: str) -> tuple[str, str] | None:
    index = _find_top_level_keyword(expr, "IN")
    if index < 0:
        return None
    left = expr[:index].strip()
    right = expr[index + len(" IN "):].strip()
    if not right.startswith("(") or not right.endswith(")"):
        return None
    return left, right[1:-1]


def _split_top_level_comparison(expr: str) -> tuple[str, str, str] | None:
    for index, operator in _top_level_operator_positions(expr, _COMPARISON_OPERATORS):
        return expr[:index].strip(), operator, expr[index + len(operator):].strip()
    return None


def _split_top_level_arithmetic(expr: str) -> tuple[str, str, str] | None:
    for ops in (("+", "-"), ("*", "/")):
        positions = list(_top_level_operator_positions(expr, ops))
        for index, operator in reversed(positions):
            if operator in {"+", "-"} and _is_unary(expr, index):
                continue
            return expr[:index].strip(), operator, expr[index + 1:].strip()
    return None


def _top_level_operator_positions(expr: str, operators: Sequence[str]):
    depth = 0
    in_quote = False
    index = 0
    while index < len(expr):
        char = expr[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0:
            for operator in operators:
                if expr.startswith(operator, index):
                    yield index, operator
                    index += len(operator) - 1
                    break
        index += 1


def _split_top_level_keyword(expr: str, keyword: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    search_from = 0
    while True:
        index = _find_top_level_keyword(expr[search_from:], keyword)
        if index < 0:
            break
        absolute = search_from + index
        parts.append(expr[start:absolute].strip())
        search_from = absolute + len(keyword) + 2
        start = search_from
    parts.append(expr[start:].strip())
    return tuple(part for part in parts if part)


def _find_top_level_keyword(expr: str, keyword: str) -> int:
    depth = 0
    in_quote = False
    needle = f" {keyword.upper()} "
    upper = expr.upper()
    for index, char in enumerate(expr):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0 and upper.startswith(needle, index):
            return index
    return -1


def _split_args(raw: str) -> tuple[str, ...]:
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


def _strip_wrapping_parentheses(expr: str) -> str:
    stripped = expr.strip()
    while stripped.startswith("(") and stripped.endswith(")") and _balanced(stripped[1:-1]):
        stripped = stripped[1:-1].strip()
    return stripped


def _balanced(expr: str) -> bool:
    depth = 0
    in_quote = False
    for char in expr:
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quote


def _is_projection_ref(expr: str) -> bool:
    return re.fullmatch(r"#\d+", expr) is not None


def _is_unary(expr: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and expr[previous].isspace():
        previous -= 1
    return previous < 0 or expr[previous] in "(,+-*/<>="

parse_literal = _parse_literal
parse_case = _parse_case
parse_cast = _parse_cast
parse_call = _parse_call
parse_in = _parse_in
split_top_level_comparison = _split_top_level_comparison
split_top_level_arithmetic = _split_top_level_arithmetic
top_level_operator_positions = _top_level_operator_positions
split_top_level_keyword = _split_top_level_keyword
find_top_level_keyword = _find_top_level_keyword
split_args = _split_args
balanced = _balanced
strip_wrapping_parentheses = _strip_wrapping_parentheses
is_projection_ref = _is_projection_ref
is_unary = _is_unary
