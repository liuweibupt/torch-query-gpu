"""Expression evaluation for DuckDB JSON physical plans."""

from __future__ import annotations

import bisect
import re
from typing import Any, Sequence

import torch

from tpch_torch.backend.physical_expr_folding import fold_same_column_literal_or
from tpch_torch.backend.physical_expr_parse import (
    CaseExpression,
    _NO_LITERAL,
    balanced as _balanced,
    is_projection_ref as _is_projection_ref,
    parse_call as _parse_call,
    parse_case as _parse_case,
    parse_cast as _parse_cast,
    parse_in as _parse_in,
    parse_literal as _parse_literal,
    split_args as _split_args,
    split_top_level_arithmetic as _split_top_level_arithmetic,
    split_top_level_comparison as _split_top_level_comparison,
    split_top_level_keyword as _split_top_level_keyword,
    strip_wrapping_parentheses as _strip_wrapping_parentheses,
)
from tpch_torch.backend.physical_duplicate_expr import evaluate_duplicate_symmetric_filter
from tpch_torch.backend.physical_date_expr import parse_extract_year, parse_scalar_subquery_guard
from tpch_torch.backend.physical_like import like_matches
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.operators import membership_mask

_COMPARISON_OPERATORS = ("!~~", "~~", ">=", "<=", "!=", "<>", "=", ">", "<")
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
    if expr.upper() == "IN (...)":
        return table.value_named("SUBQUERY")
    if (guard_expr := parse_scalar_subquery_guard(expr)) is not None:
        return evaluate_expression(table, guard_expr)
    cast_expr = _parse_cast(expr)
    if cast_expr is not None:
        return evaluate_expression(table, cast_expr)
    try:
        return table.value_named(expr)
    except KeyError:
        pass
    duplicate_filter = evaluate_duplicate_symmetric_filter(table, expr)
    if duplicate_filter is not None:
        return duplicate_filter
    case_parts = _parse_case(expr)
    if case_parts is not None:
        return _evaluate_case(table, case_parts)
    extract_expr = parse_extract_year(expr)
    if extract_expr is not None:
        return PhysicalValue(tensor=evaluate_expression(table, extract_expr).require_tensor() // 10000)
    call = _parse_call(expr)
    if call is not None:
        return _evaluate_call(table, call[0], call[1])
    in_parts = _parse_in(expr)
    if in_parts is not None:
        return _evaluate_in(table, in_parts[0], in_parts[1])
    keyword_parts = _split_top_level_keyword(expr, "OR")
    if len(keyword_parts) > 1:
        folded = fold_same_column_literal_or(
            table,
            keyword_parts,
            parse_literal=_parse_literal,
            split_comparison=_split_top_level_comparison,
            strip_parentheses=_strip_wrapping_parentheses,
            no_literal=_NO_LITERAL,
        )
        if folded is not None:
            return folded
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
    raise UnsupportedPlanError(f"unsupported physical expression: {expression}")


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
        inner = _strip_wrapping_parentheses(_unqualify_column_references(args.strip()))
        return f"{name.lower()}({inner})"
    return expr


def strip_order_direction(expression: str) -> tuple[str, bool]:
    return _strip_order_direction(expression)


def _evaluate_case(table: PhysicalTable, case_expression: CaseExpression) -> PhysicalValue:
    result = evaluate_expression(table, case_expression.else_expression)
    for condition_expression, result_expression in reversed(case_expression.branches):
        condition = _bool_tensor(evaluate_expression(table, condition_expression))
        then_value = evaluate_expression(table, result_expression)
        then_tensor, else_tensor = _coerce_binary_tensors(then_value, result)
        valid = _combine_validity(then_value, result, then_tensor)
        result = PhysicalValue(tensor=torch.where(condition, then_tensor, else_tensor), valid=valid)
    return result


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
    if lowered == "substring" and len(args) == 3:
        return _evaluate_substring(table, args)
    if lowered == "substring" and len(args) == 1:
        return _evaluate_substring(table, _parse_substring_from_args(args[0]))
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
    return PhysicalValue(tensor=membership_mask(tensor, values))


def _logical_reduce(table: PhysicalTable, parts: Sequence[str], reducer) -> PhysicalValue:
    masks = [_bool_tensor(evaluate_expression(table, part)) for part in parts]
    result = masks[0]
    for mask in masks[1:]:
        result = reducer(result, mask)
    return PhysicalValue(tensor=result)


def _compare(left: PhysicalValue, operator: str, right: PhysicalValue) -> PhysicalValue:
    if left.dictionary is not None and isinstance(right.literal, str):
        if operator in {"~~", "!~~"}:
            return PhysicalValue(tensor=_compare_like_literal(left, operator, right.literal))
        return PhysicalValue(tensor=_compare_string_literal(left, operator, right.literal))
    if right.dictionary is not None and isinstance(left.literal, str):
        return PhysicalValue(tensor=_reverse_compare(_compare_string_literal(right, operator, left.literal), operator))
    left_tensor, right_tensor = _coerce_binary_tensors(left, right)
    valid = _combine_validity(left, right, left_tensor)
    if operator == "=":
        return PhysicalValue(tensor=left_tensor == right_tensor, valid=valid)
    if operator in {"!=", "<>"}:
        return PhysicalValue(tensor=left_tensor != right_tensor, valid=valid)
    if operator == ">":
        return PhysicalValue(tensor=left_tensor > right_tensor, valid=valid)
    if operator == ">=":
        return PhysicalValue(tensor=left_tensor >= right_tensor, valid=valid)
    if operator == "<":
        return PhysicalValue(tensor=left_tensor < right_tensor, valid=valid)
    if operator == "<=":
        return PhysicalValue(tensor=left_tensor <= right_tensor, valid=valid)
    raise UnsupportedPlanError(f"unsupported comparison operator: {operator}")


def _arithmetic(left: PhysicalValue, operator: str, right: PhysicalValue) -> PhysicalValue:
    left_tensor, right_tensor = _coerce_binary_tensors(left, right)
    valid = _combine_validity(left, right, left_tensor)
    if operator == "+":
        return PhysicalValue(tensor=left_tensor + right_tensor, valid=valid)
    if operator == "-":
        return PhysicalValue(tensor=left_tensor - right_tensor, valid=valid)
    if operator == "*":
        return PhysicalValue(tensor=left_tensor * right_tensor, valid=valid)
    if operator == "/":
        return PhysicalValue(tensor=left_tensor / right_tensor, valid=valid)
    raise UnsupportedPlanError(f"unsupported arithmetic operator: {operator}")


def _coerce_binary_tensors(left: PhysicalValue, right: PhysicalValue) -> tuple[torch.Tensor, torch.Tensor]:
    if left.tensor is not None and right.tensor is not None:
        return left.tensor, right.tensor.to(dtype=left.tensor.dtype) if _numeric_literal_like(right) else right.tensor
    if left.tensor is not None:
        return left.tensor, _literal_tensor(right.literal, left.tensor)
    if right.tensor is not None:
        return _literal_tensor(left.literal, right.tensor), right.tensor
    tensor = torch.tensor(left.literal, dtype=torch.float64)
    return tensor, torch.tensor(right.literal, dtype=tensor.dtype)


def _combine_validity(left: PhysicalValue, right: PhysicalValue, like: torch.Tensor) -> torch.Tensor | None:
    valid = None
    if left.valid is not None:
        valid = left.valid
    if right.valid is not None:
        valid = right.valid if valid is None else valid & right.valid
    if valid is None:
        return None
    return valid.to(device=like.device)


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


def _evaluate_substring(table: PhysicalTable, args: Sequence[str]) -> PhysicalValue:
    value = evaluate_expression(table, args[0])
    if value.dictionary is None:
        raise UnsupportedPlanError("substring requires an encoded string column")
    start = _literal_int(evaluate_expression(table, args[1]))
    length = _literal_int(evaluate_expression(table, args[2]))
    offset = start - 1
    substrings = tuple(item[offset : offset + length] for item in value.dictionary)
    vocabulary = tuple(sorted(set(substrings)))
    ids = {item: index for index, item in enumerate(vocabulary)}
    remapped = [ids[substrings[int(index)]] for index in value.require_tensor().cpu().tolist()]
    tensor = torch.tensor(remapped, dtype=torch.int64, device=value.require_tensor().device)
    return PhysicalValue(tensor=tensor, dictionary=vocabulary, valid=value.valid)


def _parse_substring_from_args(raw_args: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(.+)\s+FROM\s+(\d+)\s+FOR\s+(\d+)", raw_args.strip(), re.I | re.S)
    if match is None:
        raise UnsupportedPlanError(f"unsupported substring arguments: {raw_args}")
    return match.group(1).strip(), match.group(2), match.group(3)


def _literal_int(value: PhysicalValue) -> int:
    if not isinstance(value.literal, int):
        raise UnsupportedPlanError("substring bounds must be integer literals")
    return value.literal


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


def _compare_like_literal(value: PhysicalValue, operator: str, pattern: str) -> torch.Tensor:
    ids = [index for index, item in enumerate(value.dictionary or ()) if like_matches(item, pattern)]
    matched = _isin_ids(value.require_tensor(), ids)
    return torch.logical_not(matched) if operator == "!~~" else matched


def _reverse_compare(mask: torch.Tensor, operator: str) -> torch.Tensor:
    if operator == "=":
        return mask
    if operator in {"!=", "<>"}:
        return mask
    raise UnsupportedPlanError(f"literal-left string comparison is not supported: {operator}")


def _isin_ids(values: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    return membership_mask(values, ids)


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


def _unqualify_column_references(expr: str) -> str:
    pattern = re.compile(r'(?:(?:[A-Za-z_][\w]*|"[^"]+")\.)+(?:([A-Za-z_][\w]*)|"([^"]+)")')
    return pattern.sub(lambda match: match.group(1) or match.group(2), expr)


def _is_unary(expr: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and expr[previous].isspace():
        previous -= 1
    return previous < 0 or expr[previous] in "(,+-*/<>="
