"""Projection alias helpers for DuckDB physical-plan execution."""

from __future__ import annotations

import re
from typing import Sequence

from tpch_torch.backend.physical_expr import projection_name
from tpch_torch.backend.physical_sql import replace_aggregate_calls_with_refs
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def projection_output_name(
    child: PhysicalTable,
    expression: str,
    index: int,
    value: PhysicalValue | None = None,
    select_aliases: dict[str, str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Return the output name for a physical projection expression."""

    expression_for_name = _projection_reference_name_expression(child, expression, value)
    name, aliases = projection_name(child, expression, index)
    if expression_for_name != expression:
        name, aliases = projection_name(child, expression_for_name, index)
    aliases = _without_positional(aliases)
    if value is not None:
        aliases = tuple(dict.fromkeys((
            *aliases,
            *_without_positional(_existing_aliases(child, value)),
            *_equivalent_key_aliases(child, expression, value),
        )))
    if select_aliases is not None:
        aliases = tuple(dict.fromkeys((*aliases, *_matching_select_aliases(select_aliases, child, value))))
    aliases = tuple(dict.fromkeys((*aliases, f"#{index}")))
    if expression in child.columns and expression != name and not _is_projection_position(expression):
        return expression, tuple(dict.fromkeys((expression, *aliases)))
    return name, aliases


def _projection_reference_name_expression(
    child: PhysicalTable,
    expression: str,
    value: PhysicalValue | None,
) -> str:
    match = re.fullmatch(r"#(\d+)", expression.strip())
    if match is None:
        return expression
    index = int(match.group(1))
    try:
        position_value = child.value_at(index) if value is None else value
        return _preferred_projection_ref_name(child, index, position_value)
    except IndexError:
        return expression


def projection_value_expression(
    select_aliases: dict[str, str],
    child: PhysicalTable,
    expression: str,
) -> str:
    """Map SELECT aliases back to the child expression that carries their value."""

    if _is_projection_position(expression.strip()):
        return expression.strip()
    if _has_named_value(child, expression):
        return _value_ref(child, expression)
    source = select_aliases.get(expression)
    if source is None:
        if _single_child_alias_candidate(child, expression):
            return "#0"
        return _aggregate_ref_expression(expression, child) or _aggregate_ref_candidate(expression, child) or expression
    if _has_named_value(child, source):
        return _value_ref(child, source)
    aggregate_alias = _matching_aggregate_alias(child, source)
    if aggregate_alias is not None:
        return _value_ref(child, aggregate_alias)
    return _replace_aggregate_calls_with_child_refs(_expand_alias_refs(source, select_aliases, child), child)


def resolve_alias_projections(
    select_aliases: dict[str, str],
    child: PhysicalTable,
    expressions: Sequence[str],
) -> tuple[str, ...]:
    """Keep SELECT aliases when the child already exposes the aliased value."""

    resolved = []
    for expression in expressions:
        source = select_aliases.get(expression)
        if source is None:
            resolved.append(expression)
        elif projection_value_expression(select_aliases, child, expression) != expression:
            resolved.append(expression)
        else:
            resolved.append(replace_aggregate_calls_with_refs(_expand_alias_refs(source, select_aliases, child)))
    return tuple(resolved)


def normalize_projection_expressions(expressions: Sequence[str]) -> tuple[str, ...]:
    """Repair DuckDB JSON projection lists split inside scalar-subquery error text."""

    normalized = []
    skip_next = False
    for index, expression in enumerate(expressions):
        if skip_next:
            skip_next = False
            continue
        if _is_split_scalar_subquery_guard(expression) and index + 1 < len(expressions):
            normalized.append(f"{expression}, {expressions[index + 1]}")
            skip_next = True
        else:
            normalized.append(expression)
    return tuple(normalized)


def aggregate_order_alias(select_aliases: dict[str, str], table: PhysicalTable, expression: str) -> str | None:
    """Return SELECT alias carrying an ORDER BY aggregate expression, if materialized."""

    target = _aggregate_signature(expression)
    if target is None:
        return None
    for alias, source in select_aliases.items():
        if _aggregate_signature(source) == target and _has_named_value(table, alias):
            return alias
    return _matching_aggregate_alias(table, expression)


def matching_aggregate_alias(table: PhysicalTable, expression: str) -> str | None:
    """Return a materialized aggregate column matching an expression signature."""

    return _matching_aggregate_alias(table, expression)


def matching_expression_alias(table: PhysicalTable, expression: str) -> str | None:
    """Return a materialized column matching a normalized scalar expression."""

    return _matching_expression_alias(table, expression)


def order_alias_value(select_aliases: dict[str, str], table: PhysicalTable, key_name: str) -> str | None:
    """Return a child value reference for an ORDER BY alias."""

    source = select_aliases.get(key_name)
    if source is None:
        source = _select_source_matching_expression(select_aliases, key_name)
    if source is None and _single_child_alias_candidate(table, key_name):
        return "#0"
    if source is None:
        return None
    if _has_named_value(table, source):
        return _value_ref(table, source)
    alias = _select_alias_matching_expression(select_aliases, key_name)
    if alias is not None and _has_named_value(table, alias):
        return _value_ref(table, alias)
    source_alias = _matching_expression_alias(table, source)
    if source_alias is not None:
        return _value_ref(table, source_alias)
    return "#0" if len(table.order) == 1 else None


def _has_named_value(table: PhysicalTable, name: str) -> bool:
    try:
        table.value_named(name)
    except KeyError:
        return False
    return True


def _single_child_alias_candidate(table: PhysicalTable, expression: str) -> bool:
    if len(table.order) != 1:
        return False
    return re.fullmatch(r'(?:[A-Za-z_][\w]*\.)?"?[A-Za-z_][\w]*"?', expression.strip()) is not None


def _is_split_scalar_subquery_guard(expression: str) -> bool:
    return "scalar subqueries can only return a single row" in expression and " ELSE " not in expression


def _all_column_names(table: PhysicalTable) -> tuple[str, ...]:
    aliases = getattr(table, "aliases", {}) or {}
    return tuple(dict.fromkeys((*tuple(table.columns), *tuple(aliases))))


def _existing_aliases(table: PhysicalTable, value: PhysicalValue) -> tuple[str, ...]:
    names = []
    for name in _all_column_names(table):
        try:
            if table.columns[name] is value:
                names.append(name)
        except KeyError:
            continue
    return tuple(dict.fromkeys(names))


def _preferred_projection_ref_name(table: PhysicalTable, index: int, value: PhysicalValue) -> str:
    aliases = _semantic_aliases(table, value)
    if aliases:
        return aliases[0]
    return table.order[index]


def _semantic_aliases(table: PhysicalTable, value: PhysicalValue) -> tuple[str, ...]:
    aliases = _without_positional(_existing_aliases(table, value))
    aliases = tuple(alias for alias in aliases if not alias.startswith("__internal_"))
    return tuple(sorted(aliases, key=_semantic_alias_rank))


def _semantic_alias_rank(alias: str) -> tuple[int, int, str]:
    clean = alias.replace('"', "")
    has_qualifier = "." in clean
    has_unique_suffix = re.search(r"__\d+$", clean) is not None
    return int(has_qualifier), int(has_unique_suffix), clean


def _equivalent_key_aliases(
    table: PhysicalTable,
    expression: str,
    value: PhysicalValue,
) -> tuple[str, ...]:
    match = re.fullmatch(r"#(\d+)", expression.strip())
    if match is None:
        return ()
    try:
        output_base = _alias_base(_preferred_projection_ref_name(table, int(match.group(1)), value))
    except IndexError:
        return ()
    aliases = []
    output_tail = _tail_base(output_base)
    if not output_tail.endswith("key"):
        return ()
    for alias in _without_positional(_all_column_names(table)):
        alias_base = _alias_base(alias)
        if alias_base == output_base or _tail_base(alias_base) == output_tail:
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


def _alias_base(alias: str) -> str:
    return _strip_unique_suffix(alias.replace('"', "").strip().rsplit(".", 1)[-1])


def _strip_unique_suffix(name: str) -> str:
    base, separator, suffix = name.rpartition("__")
    return base if separator and suffix.isdigit() else name


def _tail_base(alias: str) -> str:
    base = _alias_base(alias)
    parts = base.split("_")
    return "_".join(parts[1:]) if len(parts) > 1 else base


def _without_positional(aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(alias for alias in aliases if not _is_projection_position(alias))


def _is_projection_position(alias: str) -> bool:
    return alias.startswith("#") and alias[1:].isdigit()


def _matching_select_aliases(
    select_aliases: dict[str, str],
    child: PhysicalTable,
    value: PhysicalValue | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    matched = []
    for alias, source in select_aliases.items():
        if _has_named_value(child, source):
            if child.value_named(source) is value:
                matched.append(alias)
            continue
        aggregate_alias = _matching_aggregate_alias(child, source)
        if aggregate_alias is not None and _has_named_value(child, aggregate_alias):
            if child.value_named(aggregate_alias) is value:
                matched.append(alias)
            continue
        expression_alias = _matching_expression_alias(child, source)
        if expression_alias is not None and child.value_named(expression_alias) is value:
            matched.append(alias)
    return tuple(matched)


def _select_source_matching_expression(select_aliases: dict[str, str], expression: str) -> str | None:
    alias = _select_alias_matching_expression(select_aliases, expression)
    return select_aliases[alias] if alias is not None else None


def _select_alias_matching_expression(select_aliases: dict[str, str], expression: str) -> str | None:
    target = _normalize_expression(expression)
    for alias, source in select_aliases.items():
        if _normalize_expression(source) == target:
            return alias
    return None


def _matching_expression_alias(table: PhysicalTable, expression: str) -> str | None:
    target = _normalize_expression(expression)
    for candidate in _all_column_names(table):
        if _is_projection_position(candidate):
            continue
        if _normalize_expression(candidate) == target:
            return candidate
    return None


def _aggregate_ref_expression(expression: str, table: PhysicalTable) -> str | None:
    replaced = _replace_aggregate_calls_with_child_refs(expression, table)
    if replaced == expression:
        return None
    refs = re.findall(r"#\d+", replaced)
    if all(_has_named_value(table, ref) for ref in refs):
        return replaced
    return None


def _aggregate_ref_candidate(expression: str, table: PhysicalTable) -> str | None:
    replaced = _replace_aggregate_calls_with_child_refs(expression, table)
    return None if replaced == expression else replaced


def _expand_alias_refs(
    expression: str,
    select_aliases: dict[str, str],
    child: PhysicalTable | None = None,
) -> str:
    expanded = expression
    for alias, source in select_aliases.items():
        if alias == expression:
            continue
        if child is not None and _has_named_value(child, alias):
            continue
        expanded = re.sub(rf"\b{re.escape(alias)}\b", f"({source})", expanded)
    return expanded


def _matching_aggregate_alias(table: PhysicalTable, expression: str) -> str | None:
    target = _aggregate_signature(expression)
    if target is None:
        return None
    for candidate in _all_column_names(table):
        if _aggregate_signature(candidate) == target:
            return candidate
    return None


def _value_ref(table: PhysicalTable, name: str) -> str:
    value = table.value_named(name)
    for index, column_name in enumerate(table.order):
        if table.columns[column_name] is value:
            return f"#{index}"
    return name


def _aggregate_signature(expression: str) -> tuple[str, str] | None:
    expression = _strip_cast_wrapper(expression)
    match = re.fullmatch(r"\s*(sum|avg|min|max|count)\s*\((.*)\)\s*", expression, re.I | re.S)
    if match is None:
        return None
    return match.group(1).lower(), _normalize_expression(_strip_distinct_prefix(match.group(2)))


def _strip_distinct_prefix(expression: str) -> str:
    stripped = expression.strip()
    return stripped[8:].strip() if stripped.upper().startswith("DISTINCT ") else expression


def _strip_cast_wrapper(expression: str) -> str:
    match = re.fullmatch(r"\s*CAST\s*\((.*)\)\s*", expression, re.I | re.S)
    if match is None:
        return expression
    body = match.group(1)
    parts = _split_top_level_keyword(body, "AS")
    return parts[0] if len(parts) == 2 else expression


def _replace_aggregate_calls_with_child_refs(expression: str, table: PhysicalTable) -> str:
    positions = _aggregate_column_positions(table)
    if not positions:
        return replace_aggregate_calls_with_refs(expression)
    pieces: list[str] = []
    index = 0
    aggregate_index = 0
    while index < len(expression):
        match = _aggregate_call_at(expression, index)
        if match is None:
            pieces.append(expression[index])
            index += 1
            continue
        ref_index = positions[min(aggregate_index, len(positions) - 1)]
        pieces.append(f"#{ref_index}")
        aggregate_index += 1
        index = match[1]
    return "".join(pieces)


def _aggregate_column_positions(table: PhysicalTable) -> tuple[int, ...]:
    positions = [
        index
        for index, name in enumerate(table.order)
        if re.match(r"\s*(sum|avg|min|max|count)\s*\(", name, re.I)
    ]
    return tuple(positions)


def _aggregate_call_at(expression: str, index: int) -> tuple[int, int] | None:
    name_match = re.match(r"[A-Za-z_][\w]*", expression[index:])
    if name_match is None:
        return None
    name = name_match.group(0).lower()
    if name not in {"sum", "avg", "min", "max", "count"}:
        return None
    open_index = index + len(name)
    while open_index < len(expression) and expression[open_index].isspace():
        open_index += 1
    if open_index >= len(expression) or expression[open_index] != "(":
        return None
    return index, _matching_paren(expression, open_index) + 1


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


def _split_top_level_keyword(expression: str, keyword: str) -> tuple[str, ...]:
    needle = f" {keyword.upper()} "
    upper = expression.upper()
    for index in range(len(expression)):
        if upper.startswith(needle, index) and _top_level(expression, index):
            return expression[:index].strip(), expression[index + len(needle) :].strip()
    return (expression.strip(),)


def _top_level(expression: str, target: int) -> bool:
    depth = 0
    in_quote = False
    for index, char in enumerate(expression[:target]):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
    return depth == 0 and not in_quote


def _normalize_expression(expression: str) -> str:
    unqualified = _unqualify_column_references(expression)
    compact = re.sub(r"\s+", "", _strip_wrapping_parentheses(unqualified))
    return _normalize_numeric_literals(compact).lower()


def _normalize_numeric_literals(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        text = match.group(0)
        return str(int(float(text))) if float(text).is_integer() else text.rstrip("0").rstrip(".")

    return re.sub(r"\b\d+\.\d+\b", replace, expression)


def _unqualify_column_references(expr: str) -> str:
    pattern = re.compile(r'(?:(?:[A-Za-z_][\w]*|"[^"]+")\.)+(?:([A-Za-z_][\w]*)|"([^"]+)")')
    return pattern.sub(lambda match: match.group(1) or match.group(2), expr)


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
