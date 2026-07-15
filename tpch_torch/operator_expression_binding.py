"""Slot-aware parsing for TQP graph expression metadata."""

from __future__ import annotations

import re
from typing import Sequence

from tpch_torch.operator_refs import TQPExprNode, TQPSlot

_SLOT_REF_PATTERN = re.compile(r"#(?P<ordinal>\d+)")
_IDENTIFIER_PATTERN = re.compile(r'(?<![#."])([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)')
_COMPARISON_OPERATORS = ("!~~", "~~", ">=", "<=", "!=", "<>", "=", ">", "<")


def parse_expression_tree(expression: str, slots: Sequence[TQPSlot]) -> TQPExprNode:
    """Parse a backend expression string into a small slot-aware AST."""

    expr = _strip_wrapping_parentheses(expression.strip())
    if expr == "":
        return TQPExprNode("empty")
    ordinal = _single_ordinal_ref(expr, slots)
    if ordinal is not None:
        return TQPExprNode("slot_ref", ref=ordinal.ref)
    literal = _literal_node(expr)
    if literal is not None:
        return literal
    slot = _single_identifier_slot(expr, slots)
    if slot is not None:
        return TQPExprNode("slot_ref", ref=slot.ref)
    if expr.upper().startswith("NOT "):
        return TQPExprNode("unary", "not", (parse_expression_tree(expr[4:], slots),))
    logical = _split_keyword_expression(expr, "OR")
    if len(logical) > 1:
        return TQPExprNode("logical", "or", tuple(parse_expression_tree(part, slots) for part in logical))
    logical = _split_keyword_expression(expr, "AND")
    if len(logical) > 1:
        return TQPExprNode("logical", "and", tuple(parse_expression_tree(part, slots) for part in logical))
    comparison = _split_top_level_operator(expr, _COMPARISON_OPERATORS)
    if comparison is not None:
        return _binary_node(slots, comparison)
    arithmetic = _split_arithmetic_expression(expr)
    if arithmetic is not None:
        return _binary_node(slots, arithmetic)
    cast = _parse_cast_expression(expr)
    if cast is not None:
        child, type_name = cast
        return TQPExprNode("cast", type_name, (parse_expression_tree(child, slots),))
    extract = _parse_extract_year_expression(expr)
    if extract is not None:
        return TQPExprNode("extract", "year", (parse_expression_tree(extract, slots),))
    call = _parse_call_expression(expr)
    if call is not None:
        name, raw_args = call
        return TQPExprNode("call", name, tuple(parse_expression_tree(arg, slots) for arg in _split_args(raw_args)))
    return TQPExprNode("unknown", expr, tuple(_slot_ref_nodes(expr, slots)))


def _literal_node(expr: str) -> TQPExprNode | None:
    if re.fullmatch(r"-?\d+", expr):
        return TQPExprNode("literal", int(expr))
    if re.fullmatch(r"-?\d+\.\d+", expr):
        return TQPExprNode("literal", float(expr))
    if re.fullmatch(r"'[^']*'", expr):
        return TQPExprNode("literal", expr[1:-1].replace("''", "'"))
    if expr.upper() == "NULL":
        return TQPExprNode("literal", None)
    if expr.upper() == "TRUE":
        return TQPExprNode("literal", True)
    if expr.upper() == "FALSE":
        return TQPExprNode("literal", False)
    return None


def _single_ordinal_ref(expr: str, slots: Sequence[TQPSlot]) -> TQPSlot | None:
    match = re.fullmatch(r"#(\d+)", expr.strip())
    if match is None:
        return None
    ordinal = int(match.group(1))
    return slots[ordinal] if ordinal < len(slots) else None


def _single_identifier_slot(expr: str, slots: Sequence[TQPSlot]) -> TQPSlot | None:
    if not re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?", expr):
        return None
    matches = _matching_slots(expr, slots)
    return matches[0] if len(matches) == 1 else None


def _slot_ref_nodes(expr: str, slots: Sequence[TQPSlot]):
    seen = set()
    for slot in _referenced_slots(expr, slots):
        if slot.slot_id in seen:
            continue
        seen.add(slot.slot_id)
        yield TQPExprNode("slot_ref", ref=slot.ref)


def _referenced_slots(expr: str, slots: Sequence[TQPSlot]) -> tuple[TQPSlot, ...]:
    refs = []
    for match in _SLOT_REF_PATTERN.finditer(expr):
        ordinal = int(match.group("ordinal"))
        if ordinal < len(slots):
            refs.append(slots[ordinal])
    for match in _IDENTIFIER_PATTERN.finditer(expr):
        identifier = match.group(1)
        if _inside_string(expr, identifier):
            continue
        matches = _matching_slots(identifier, slots)
        if len(matches) == 1:
            refs.append(matches[0])
    return tuple(refs)


def _matching_slots(identifier: str, slots: Sequence[TQPSlot]) -> tuple[TQPSlot, ...]:
    lowered = identifier.lower()
    return tuple(slot for slot in slots if lowered in {alias.lower() for alias in slot.aliases})


def _inside_string(expression: str, identifier: str) -> bool:
    start = expression.find(identifier)
    return start >= 0 and expression[:start].count("'") % 2 == 1


def _binary_node(slots: Sequence[TQPSlot], parts: tuple[str, str, str]) -> TQPExprNode:
    left, operator, right = parts
    return TQPExprNode("binary", operator, (parse_expression_tree(left, slots), parse_expression_tree(right, slots)))


def _parse_cast_expression(expr: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"CAST\s*\((.*)\)", expr, re.I | re.S)
    if match is None or not _balanced(match.group(1)):
        return None
    parts = _split_keyword_expression(match.group(1), "AS")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _parse_extract_year_expression(expr: str) -> str | None:
    match = re.fullmatch(r"extract\s*\(\s*year\s+FROM\s+(.+)\)", expr, re.I | re.S)
    return match.group(1).strip() if match is not None else None


def _parse_call_expression(expr: str) -> tuple[str, str] | None:
    match = re.fullmatch(r'"?([A-Za-z_][\w]*)"?\s*\((.*)\)', expr, re.S)
    if match is None or not _balanced(match.group(2)):
        return None
    return match.group(1), match.group(2)


def _split_arithmetic_expression(expr: str) -> tuple[str, str, str] | None:
    for operators in (("+", "-"), ("*", "/")):
        positions = list(_top_level_operator_positions(expr, operators))
        for index, operator in reversed(positions):
            if operator in {"+", "-"} and _is_unary(expr, index):
                continue
            return expr[:index].strip(), operator, expr[index + 1 :].strip()
    return None


def _split_top_level_operator(expr: str, operators: Sequence[str]) -> tuple[str, str, str] | None:
    for index, operator in _top_level_operator_positions(expr, operators):
        return expr[:index].strip(), operator, expr[index + len(operator) :].strip()
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


def _split_keyword_expression(expr: str, keyword: str) -> tuple[str, ...]:
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


def _is_unary(expr: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and expr[previous].isspace():
        previous -= 1
    if previous < 0:
        return True
    return expr[previous] in "(+-*/=<>"
