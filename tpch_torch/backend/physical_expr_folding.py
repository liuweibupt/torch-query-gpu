"""Expression rewrite helpers for DuckDB physical-plan scalar expressions."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.operators import membership_mask

ParseLiteral = Callable[[str], Any]
SplitComparison = Callable[[str], tuple[str, str, str] | None]
StripParentheses = Callable[[str], str]


def fold_same_column_literal_or(
    table: PhysicalTable,
    parts: Sequence[str],
    *,
    parse_literal: ParseLiteral,
    split_comparison: SplitComparison,
    strip_parentheses: StripParentheses,
    no_literal: object,
) -> PhysicalValue | None:
    """Fold `col = literal OR col = literal` into one membership mask."""

    column_value: PhysicalValue | None = None
    literals: list[int | float] = []
    for part in parts:
        parsed = _parse_literal_equality(
            table,
            part,
            parse_literal=parse_literal,
            split_comparison=split_comparison,
            strip_parentheses=strip_parentheses,
            no_literal=no_literal,
        )
        if parsed is None:
            return None
        value, literal = parsed
        if column_value is not None and value is not column_value:
            return None
        column_value = value
        literals.append(literal)
    if column_value is None:
        return None
    return PhysicalValue(tensor=membership_mask(column_value.require_tensor(), literals))


def _parse_literal_equality(
    table: PhysicalTable,
    expression: str,
    *,
    parse_literal: ParseLiteral,
    split_comparison: SplitComparison,
    strip_parentheses: StripParentheses,
    no_literal: object,
) -> tuple[PhysicalValue, int | float] | None:
    comparison = split_comparison(strip_parentheses(expression.strip()))
    if comparison is None or comparison[1] != "=":
        return None
    left, _, right = comparison
    parsed = _literal_equality_side(table, left, right, parse_literal, strip_parentheses, no_literal)
    if parsed is not None:
        return parsed
    return _literal_equality_side(table, right, left, parse_literal, strip_parentheses, no_literal)


def _literal_equality_side(
    table: PhysicalTable,
    column_expr: str,
    literal_expr: str,
    parse_literal: ParseLiteral,
    strip_parentheses: StripParentheses,
    no_literal: object,
) -> tuple[PhysicalValue, int | float] | None:
    literal = parse_literal(strip_parentheses(literal_expr.strip()))
    if literal is no_literal:
        return None
    try:
        value = table.value_named(strip_parentheses(column_expr.strip()))
    except KeyError:
        return None
    encoded = _encode_membership_literal(value, literal)
    return (value, encoded) if encoded is not None else None


def _encode_membership_literal(value: PhysicalValue, literal: Any) -> int | float | None:
    if value.dictionary is None:
        return literal if isinstance(literal, (int, float, bool)) else None
    if not isinstance(literal, str):
        return None
    try:
        return value.dictionary.index(literal)
    except ValueError:
        return -1
