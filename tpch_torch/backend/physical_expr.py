"""Expression evaluation for DuckDB JSON physical plans."""

from __future__ import annotations

import bisect
import re
from typing import Any, Sequence

import torch

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError

_COMPARISON_OPERATORS = (">=", "<=", "!=", "<>", "=", ">", "<")
_INTERNAL_PREFIXES = ("__internal_compress", "__internal_decompress")


def evaluate_expression(table: PhysicalTable, expression: str) -> PhysicalValue:
    """Evaluate one DuckDB scalar expression against a physical table."""

    expr = _strip_wrapping_parentheses(expression.strip())
    if not expr:
        raise UnsupportedPlanError("empty physical expression")
    literal = _parse_literal(expr)
    if literal is not _NO_LITERAL:
        return PhysicalValue(literal=literal)
    if _is_projection_ref(expr):
        return table.value_at(int(expr[1:]))
    case_parts = _parse_case(expr)
    if case_parts is not None:
        return _evaluate_case(table, case_parts)
    call = _parse_call(expr)
    if call is not None:
        return _evaluate_call(table, call[0], call[1])
    in_parts = _parse_in(expr)
    if in_parts is not None:
        return _evaluate_in(table, in_parts[0], in_parts[1])
    keyword_parts = _split_top_level_keyword(expr, "OR")
    if len(keyword_parts) > 1:
        return _logical_reduce(table, keyword_parts, torch.logical_or)
    keyword_parts = _split_top_level_keyword(expr, "AND")
    if len(keyword_parts) > 1:
        return _logical_reduce(table, keyword_parts, torch.logical_and)
    if expr.upper().startswith("NOT "):
        return PhysicalValue(tensor=torch.logical_not(_bool_tensor(evaluate_expression(table, expr[4:]))))
    comparison = _split_top_level_comparison(expr)
    if comparison is not None:
        left, operator, right = comparison
        return _compare(evaluate_expression(table, left), operator, evaluate_expression(table, right))
    arithmetic = _split_top_level_arithmetic(expr)
    if arithmetic is not None:
        left, operator, right = arithmetic
        try:
            return _arithmetic(evaluate_expression(table, left), operator, evaluate_expression(table, right))
        except UnsupportedPlanError:
            pass
    try:
        return table.value_named(expr)
    except KeyError as exc:
        raise UnsupportedPlanError(f"unsupported physical expression: {expression}") from exc


def projection_name(table: PhysicalTable, expression: str, index: int) -> tuple[str, tuple[str, ...]]:
    """Return the preferred output name and aliases for a projection expression."""

    expr = _strip_wrapping_parentheses(expression.strip())
    call = _parse_call(expr)
    if call is not None and call[0].startswith(_INTERNAL_PREFIXES):
        args = _split_args(call[1])
        if args:
            name, aliases = projection_name(table, args[0], index)
            return name, aliases + (expr,)
    if _is_projection_ref(expr):
        try:
            position = int(expr[1:])
            name = table.order[position]
            value = table.value_at(position)
            aliases = tuple(key for key, candidate in table.columns.items() if candidate is value)
            return name, (expr, *aliases)
        except IndexError as exc:
            raise UnsupportedPlanError(f"projection reference out of range: {expr}") from exc
    if _is_column_reference(expr):
        return _unqualified(expr), (expr,)
    return expr or f"col{index}", (expr,)


def aggregate_output_aliases(function: str, argument: str, child_name: str | None) -> tuple[str, ...]:
    aliases = [f"{function}({argument})"]
    if child_name is not None:
        aliases.append(f"{function}({child_name})")
        aliases.append(f"{function}({_unqualified(child_name)})")
        aliases.append(f"{function}({_qualified_wildcard(child_name)})")
    return tuple(dict.fromkeys(aliases))


def expression_sort_key_name(expression: str) -> str:
    expr = _strip_wrapping_parentheses(_strip_order_direction(expression)[0])
    call = _parse_call(expr)
    if call is None:
        return _unqualified(expr)
    name, args = call
    if name.lower() in {"sum", "avg", "min", "max", "count"}:
        inner = _strip_wrapping_parentheses(args)
        return f"{name.lower()}({_unqualified(inner)})"
    return expr


def strip_order_direction(expression: str) -> tuple[str, bool]:
    return _strip_order_direction(expression)


def _evaluate_case(table: PhysicalTable, parts: tuple[str, str, str]) -> PhysicalValue:
    condition = _bool_tensor(evaluate_expression(table, parts[0]))
    then_value = evaluate_expression(table, parts[1])
    else_value = evaluate_expression(table, parts[2])
    then_tensor, else_tensor = _coerce_binary_tensors(then_value, else_value)
    return PhysicalValue(tensor=torch.where(condition, then_tensor, else_tensor))


def _evaluate_call(table: PhysicalTable, name: str, raw_args: str) -> PhysicalValue:
    args = _split_args(raw_args)
    lowered = name.lower()
    if lowered.startswith(_INTERNAL_PREFIXES):
        if not args:
            raise UnsupportedPlanError(f"internal wrapper has no argument: {name}")
        return evaluate_expression(table, args[0])
    if lowered in {"prefix", "contains", "suffix"} and len(args) == 2:
        value = evaluate_expression(table, args[0])
        literal = _literal_string(evaluate_expression(table, args[1]))
        return PhysicalValue(tensor=_string_function(value, literal, lowered))
    if lowered == "constant_or_null" and args:
        return evaluate_expression(table, args[0])
    raise UnsupportedPlanError(f"unsupported physical function: {name}")


def _evaluate_in(table: PhysicalTable, left_expr: str, raw_values: str) -> PhysicalValue:
    left = evaluate_expression(table, left_expr)
    values = [_parse_literal(item.strip()) for item in _split_args(raw_values)]
    if any(value is _NO_LITERAL for value in values):
        raise UnsupportedPlanError(f"unsupported IN literal list: {raw_values}")
    tensor = left.require_tensor()
    if left.dictionary is not None:
        accepted = [left.dictionary.index(str(value)) for value in values if str(value) in left.dictionary]
        return PhysicalValue(tensor=_isin_ids(tensor, accepted))
    literal_tensor = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    return PhysicalValue(tensor=torch.isin(tensor, literal_tensor))


def _logical_reduce(table: PhysicalTable, parts: Sequence[str], reducer) -> PhysicalValue:
    masks = [_bool_tensor(evaluate_expression(table, part)) for part in parts]
    result = masks[0]
    for mask in masks[1:]:
        result = reducer(result, mask)
    return PhysicalValue(tensor=result)


def _compare(left: PhysicalValue, operator: str, right: PhysicalValue) -> PhysicalValue:
    if left.dictionary is not None and isinstance(right.literal, str):
        return PhysicalValue(tensor=_compare_string_literal(left, operator, right.literal))
    if right.dictionary is not None and isinstance(left.literal, str):
        return PhysicalValue(tensor=_reverse_compare(_compare_string_literal(right, operator, left.literal), operator))
    left_tensor, right_tensor = _coerce_binary_tensors(left, right)
    if operator == "=":
        return PhysicalValue(tensor=left_tensor == right_tensor)
    if operator in {"!=", "<>"}:
        return PhysicalValue(tensor=left_tensor != right_tensor)
    if operator == ">":
        return PhysicalValue(tensor=left_tensor > right_tensor)
    if operator == ">=":
        return PhysicalValue(tensor=left_tensor >= right_tensor)
    if operator == "<":
        return PhysicalValue(tensor=left_tensor < right_tensor)
    if operator == "<=":
        return PhysicalValue(tensor=left_tensor <= right_tensor)
    raise UnsupportedPlanError(f"unsupported comparison operator: {operator}")


def _arithmetic(left: PhysicalValue, operator: str, right: PhysicalValue) -> PhysicalValue:
    left_tensor, right_tensor = _coerce_binary_tensors(left, right)
    if operator == "+":
        return PhysicalValue(tensor=left_tensor + right_tensor)
    if operator == "-":
        return PhysicalValue(tensor=left_tensor - right_tensor)
    if operator == "*":
        return PhysicalValue(tensor=left_tensor * right_tensor)
    if operator == "/":
        return PhysicalValue(tensor=left_tensor / right_tensor)
    raise UnsupportedPlanError(f"unsupported arithmetic operator: {operator}")


_NO_LITERAL = object()


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


def _parse_call(expr: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"([A-Za-z_][\w]*)\s*\((.*)\)", expr, re.S)
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


def _coerce_binary_tensors(left: PhysicalValue, right: PhysicalValue) -> tuple[torch.Tensor, torch.Tensor]:
    if left.tensor is not None and right.tensor is not None:
        return left.tensor, right.tensor.to(dtype=left.tensor.dtype) if _numeric_literal_like(right) else right.tensor
    if left.tensor is not None:
        return left.tensor, _literal_tensor(right.literal, left.tensor)
    if right.tensor is not None:
        return _literal_tensor(left.literal, right.tensor), right.tensor
    tensor = torch.tensor(left.literal, dtype=torch.float64)
    return tensor, torch.tensor(right.literal, dtype=tensor.dtype)


def _literal_tensor(value: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, str):
        parsed = _parse_literal(value)
        value = parsed if parsed is not _NO_LITERAL else value
    return torch.tensor(value, dtype=like.dtype, device=like.device)


def _numeric_literal_like(value: PhysicalValue) -> bool:
    return value.literal is not None and not isinstance(value.literal, str)


def _bool_tensor(value: PhysicalValue) -> torch.Tensor:
    tensor = value.require_tensor()
    if tensor.dtype is not torch.bool:
        raise UnsupportedPlanError("physical boolean expression did not produce a boolean tensor")
    return tensor


def _literal_string(value: PhysicalValue) -> str:
    if not isinstance(value.literal, str):
        raise UnsupportedPlanError("physical string function requires a string literal")
    return value.literal


def _string_function(value: PhysicalValue, literal: str, function: str) -> torch.Tensor:
    tensor = value.require_tensor()
    if value.dictionary is None:
        raise UnsupportedPlanError(f"{function} requires an encoded string column")
    if function == "prefix":
        ids = [index for index, item in enumerate(value.dictionary) if item.startswith(literal)]
    elif function == "suffix":
        ids = [index for index, item in enumerate(value.dictionary) if item.endswith(literal)]
    else:
        ids = [index for index, item in enumerate(value.dictionary) if literal in item]
    return _isin_ids(tensor, ids)


def _compare_string_literal(value: PhysicalValue, operator: str, literal: str) -> torch.Tensor:
    tensor = value.require_tensor()
    vocabulary = value.dictionary or ()
    left = bisect.bisect_left(vocabulary, literal)
    right = bisect.bisect_right(vocabulary, literal)
    if operator == "=":
        return (tensor == left) if left < len(vocabulary) and vocabulary[left] == literal else tensor < 0
    if operator in {"!=", "<>"}:
        return torch.logical_not(_compare_string_literal(value, "=", literal))
    if operator == "<":
        return tensor < left
    if operator == "<=":
        return tensor < right
    if operator == ">":
        return tensor >= right
    if operator == ">=":
        return tensor >= left
    raise UnsupportedPlanError(f"unsupported string comparison operator: {operator}")


def _reverse_compare(mask: torch.Tensor, operator: str) -> torch.Tensor:
    if operator == "=":
        return mask
    if operator in {"!=", "<>"}:
        return mask
    raise UnsupportedPlanError(f"literal-left string comparison is not supported: {operator}")


def _isin_ids(values: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    accepted = torch.tensor(tuple(ids), dtype=values.dtype, device=values.device)
    return torch.isin(values, accepted)


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


def _strip_order_direction(expression: str) -> tuple[str, bool]:
    expr = expression.strip()
    match = re.fullmatch(r"(.+?)\s+(ASC|DESC)(?:\s+NULLS\s+(?:FIRST|LAST))?", expr, re.I)
    if match is None:
        return expr, False
    return match.group(1).strip(), match.group(2).upper() == "DESC"


def _is_projection_ref(expr: str) -> bool:
    return re.fullmatch(r"#\d+", expr) is not None


def _is_column_reference(expr: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*", expr.replace('"', "")) is not None


def _unqualified(expr: str) -> str:
    raw = expr.replace('"', "").strip()
    return raw.rsplit(".", 1)[-1] if "." in raw else raw


def _qualified_wildcard(child_name: str) -> str:
    return f"*.{child_name}"


def _is_unary(expr: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and expr[previous].isspace():
        previous -= 1
    return previous < 0 or expr[previous] in "(,+-*/<>="
