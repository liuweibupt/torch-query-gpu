"""Duplicate-column expression recovery for DuckDB physical-plan filters."""

from __future__ import annotations

import re
from typing import Sequence

import torch

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def evaluate_duplicate_symmetric_filter(table: PhysicalTable, expression: str) -> PhysicalValue | None:
    """Evaluate alias-stripped two-column symmetric equality filters when possible."""

    disjunctions = _split_top_level_keyword(_strip_wrapping_parentheses(expression), "OR")
    if len(disjunctions) != 2:
        return None
    first = _equality_pair(disjunctions[0])
    second = _equality_pair(disjunctions[1])
    if first is None or second is None:
        return None
    column, left_literal, right_literal = first
    second_column, second_left, second_right = second
    if _base_name(column) != _base_name(second_column):
        return None
    if (left_literal, right_literal) != (second_right, second_left):
        return None
    values = _duplicate_ordered_values(table, column)
    if len(values) < 2:
        return None
    left_value, right_value = values[:2]
    first_mask = _compare_literal(left_value, left_literal) & _compare_literal(right_value, right_literal)
    second_mask = _compare_literal(left_value, right_literal) & _compare_literal(right_value, left_literal)
    return PhysicalValue(tensor=first_mask | second_mask)


def _equality_pair(expression: str) -> tuple[str, str, str] | None:
    conjuncts = _split_top_level_keyword(_strip_wrapping_parentheses(expression), "AND")
    if len(conjuncts) != 2:
        return None
    first = _literal_equality(conjuncts[0])
    second = _literal_equality(conjuncts[1])
    if first is None or second is None:
        return None
    if _base_name(first[0]) != _base_name(second[0]):
        return None
    if first[1] == second[1]:
        return None
    return first[0], first[1], second[1]


def _literal_equality(expression: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"\(?\s*(?P<column>[A-Za-z_][\w]*(?:__\d+)?)\s*=\s*'(?P<literal>[^']*)'\s*\)?",
        _strip_wrapping_parentheses(expression),
        re.S,
    )
    if match is None:
        return None
    return match.group("column"), match.group("literal")


def _duplicate_ordered_values(table: PhysicalTable, column: str) -> tuple[PhysicalValue, ...]:
    base = _base_name(column)
    values = []
    for name in table.order:
        if _base_name(name) != base:
            continue
        value = table.columns[name]
        if all(value is not existing for existing in values):
            values.append(value)
    return tuple(values)


def _compare_literal(value: PhysicalValue, literal: str) -> torch.Tensor:
    tensor = value.require_tensor()
    if value.dictionary is None:
        parsed = int(literal) if re.fullmatch(r"-?\d+", literal) else float(literal)
        return tensor == torch.tensor(parsed, dtype=tensor.dtype, device=tensor.device)
    if literal not in value.dictionary:
        return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    return tensor == value.dictionary.index(literal)


def _split_top_level_keyword(expression: str, keyword: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    search_from = 0
    while True:
        index = _find_top_level_keyword(expression[search_from:], keyword)
        if index < 0:
            break
        absolute = search_from + index
        parts.append(expression[start:absolute].strip())
        search_from = absolute + len(keyword) + 2
        start = search_from
    parts.append(expression[start:].strip())
    return tuple(part for part in parts if part)


def _find_top_level_keyword(expression: str, keyword: str) -> int:
    depth = 0
    in_quote = False
    needle = f" {keyword.upper()} "
    upper = expression.upper()
    for index, char in enumerate(expression):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0 and upper.startswith(needle, index):
            return index
    return -1


def _strip_wrapping_parentheses(expression: str) -> str:
    stripped = expression.strip()
    while stripped.startswith("(") and stripped.endswith(")") and _balanced(stripped[1:-1]):
        stripped = stripped[1:-1].strip()
    return stripped


def _balanced(expression: str) -> bool:
    depth = 0
    in_quote = False
    for char in expression:
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quote


def _base_name(name: str) -> str:
    raw = name.replace('"', "").strip().rsplit(".", 1)[-1]
    base, separator, suffix = raw.rpartition("__")
    return base if separator and suffix.isdigit() else raw
