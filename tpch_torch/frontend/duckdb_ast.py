"""DuckDB parser JSON helpers for Sirius-grade frontend metadata.

This module uses DuckDB's own SQL parser (`json_serialize_sql`) to extract
SELECT aliases and expression structure. It replaces backend-side regex parsing
for AS/projection metadata while the repository is still Python-only.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import duckdb

from tpch_torch.planner import DuckDBPlannerError

_BINARY_COMPARISONS = {
    "COMPARE_EQUAL": "=",
    "COMPARE_NOTEQUAL": "!=",
    "COMPARE_LESSTHAN": "<",
    "COMPARE_GREATERTHAN": ">",
    "COMPARE_LESSTHANOREQUALTO": "<=",
    "COMPARE_GREATERTHANOREQUALTO": ">=",
}
_BINARY_FUNCTION_OPERATORS = frozenset({"+", "-", "*", "/", "%", "~~", "!~~"})
_LOGICAL_CONJUNCTIONS = {"CONJUNCTION_AND": "AND", "CONJUNCTION_OR": "OR"}
_UNARY_OPERATORS = {"OPERATOR_NOT": "NOT", "OPERATOR_IS_NULL": "IS NULL", "OPERATOR_IS_NOT_NULL": "IS NOT NULL"}


def serialize_sql_ast(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    """Return DuckDB parser JSON for SQL without binding or executing the query."""

    try:
        raw = con.execute("select json_serialize_sql(?)", [sql]).fetchone()[0]
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB SQL JSON serialization failed: {exc}") from exc
    parsed = json.loads(str(raw))
    if parsed.get("error"):
        raise DuckDBPlannerError(f"DuckDB SQL JSON serialization returned error: {parsed}")
    return parsed


def select_expressions_by_alias(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, str]:
    """Return SELECT alias -> rendered expression using DuckDB's parser AST."""

    aliases: dict[str, str] = {}
    for select_node in _select_nodes(serialize_sql_ast(con, sql)):
        for expression in select_node.get("select_list") or ():
            alias = str(expression.get("alias") or "")
            if alias:
                aliases[alias] = render_expression(expression)
    return aliases


def render_expression(expression: Mapping[str, Any]) -> str:
    """Render a DuckDB parser expression JSON node into backend expression text."""

    expression_class = str(expression.get("class") or "")
    expression_type = str(expression.get("type") or "")
    if expression_class == "COLUMN_REF":
        return _render_column_ref(expression)
    if expression_class == "CONSTANT":
        return _render_constant(expression)
    if expression_class == "FUNCTION":
        return _render_function(expression)
    if expression_class == "COMPARISON":
        return _render_binary(expression, _comparison_operator(expression_type))
    if expression_class == "CONJUNCTION":
        return _render_joined(expression.get("children") or (), _conjunction_operator(expression_type))
    if expression_class == "CAST":
        return _render_cast(expression)
    if expression_class == "CASE":
        return _render_case(expression)
    if expression_class == "BETWEEN":
        return _render_between(expression)
    if expression_class == "OPERATOR":
        return _render_operator(expression)
    if expression_class == "STAR":
        return "*"
    raise DuckDBPlannerError(f"unsupported DuckDB parser expression class: {expression_class}")


def _select_nodes(serialized: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    nodes: list[Mapping[str, Any]] = []
    for statement in serialized.get("statements") or ():
        node = statement.get("node") or {}
        nodes.extend(_walk_select_nodes(node))
    return tuple(nodes)


def _walk_select_nodes(node: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    found = [node] if node.get("type") == "SELECT_NODE" else []
    for value in node.values():
        if isinstance(value, dict):
            found.extend(_walk_select_nodes(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    found.extend(_walk_select_nodes(item))
    return tuple(found)


def _render_column_ref(expression: Mapping[str, Any]) -> str:
    names = tuple(str(part) for part in expression.get("column_names") or ())
    return ".".join(_quote_identifier(part) for part in names)


def _render_constant(expression: Mapping[str, Any]) -> str:
    value = expression.get("value") or {}
    if value.get("is_null"):
        return "NULL"
    type_info = value.get("type") or {}
    type_id = str(type_info.get("id") or "").upper()
    literal = value.get("value")
    if type_id == "VARCHAR":
        return _quote_string(str(literal))
    if type_id == "DECIMAL":
        return _render_decimal(literal, type_info.get("type_info") or {})
    if type_id == "BOOLEAN":
        return "TRUE" if bool(literal) else "FALSE"
    return str(literal)


def _render_decimal(value: Any, type_info: Mapping[str, Any]) -> str:
    scale = int(type_info.get("scale") or 0)
    if scale <= 0:
        return str(value)
    sign = "-" if int(value) < 0 else ""
    digits = str(abs(int(value))).rjust(scale + 1, "0")
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


def _render_function(expression: Mapping[str, Any]) -> str:
    name = str(expression.get("function_name") or "")
    children = tuple(render_expression(child) for child in expression.get("children") or ())
    if expression.get("is_operator") and name in _BINARY_FUNCTION_OPERATORS and len(children) == 2:
        return f"({children[0]} {name} {children[1]})"
    if name == "date_part" and len(children) == 2 and children[0].lower() == "'year'":
        return f"extract(year FROM {children[1]})"
    if name == "count_star" and not children:
        return "count(*)"
    distinct = "DISTINCT " if expression.get("distinct") else ""
    return f"{name}({distinct}{', '.join(children)})"


def _render_binary(expression: Mapping[str, Any], operator: str) -> str:
    left = render_expression(expression.get("left") or {})
    right = render_expression(expression.get("right") or {})
    return f"({left} {operator} {right})"


def _render_joined(children: Sequence[Mapping[str, Any]], operator: str) -> str:
    rendered = tuple(render_expression(child) for child in children)
    if not rendered:
        raise DuckDBPlannerError(f"{operator} expression has no children")
    return f" {operator} ".join(f"({child})" for child in rendered)


def _render_cast(expression: Mapping[str, Any]) -> str:
    child = render_expression(expression.get("child") or {})
    cast_type = _render_type(expression.get("cast_type") or {})
    return f"CAST({child} AS {cast_type})"


def _render_case(expression: Mapping[str, Any]) -> str:
    parts = ["CASE"]
    for check in expression.get("case_checks") or ():
        when_expr = render_expression(check.get("when_expr") or {})
        then_expr = render_expression(check.get("then_expr") or {})
        parts.append(f"WHEN {when_expr} THEN {then_expr}")
    parts.append(f"ELSE {render_expression(expression.get('else_expr') or {})} END")
    return " ".join(parts)


def _render_between(expression: Mapping[str, Any]) -> str:
    input_expr = render_expression(expression.get("input") or {})
    lower = render_expression(expression.get("lower") or {})
    upper = render_expression(expression.get("upper") or {})
    return f"({input_expr} BETWEEN {lower} AND {upper})"


def _render_operator(expression: Mapping[str, Any]) -> str:
    operator = _unary_operator(str(expression.get("type") or ""))
    children = tuple(render_expression(child) for child in expression.get("children") or ())
    if len(children) != 1:
        raise DuckDBPlannerError(f"{operator} expression expects one child")
    if operator == "NOT":
        return f"NOT ({children[0]})"
    return f"({children[0]} {operator})"


def _render_type(type_node: Mapping[str, Any]) -> str:
    type_id = str(type_node.get("id") or "").upper()
    info = type_node.get("type_info") or {}
    if type_id == "DECIMAL":
        return f"DECIMAL({int(info.get('width'))},{int(info.get('scale'))})"
    return type_id


def _comparison_operator(expression_type: str) -> str:
    if expression_type not in _BINARY_COMPARISONS:
        raise DuckDBPlannerError(f"unsupported comparison expression type: {expression_type}")
    return _BINARY_COMPARISONS[expression_type]


def _conjunction_operator(expression_type: str) -> str:
    if expression_type not in _LOGICAL_CONJUNCTIONS:
        raise DuckDBPlannerError(f"unsupported conjunction expression type: {expression_type}")
    return _LOGICAL_CONJUNCTIONS[expression_type]


def _unary_operator(expression_type: str) -> str:
    if expression_type not in _UNARY_OPERATORS:
        raise DuckDBPlannerError(f"unsupported operator expression type: {expression_type}")
    return _UNARY_OPERATORS[expression_type]


def _quote_identifier(identifier: str) -> str:
    if identifier == "*" or identifier.replace("_", "").isalnum() and not identifier[0].isdigit():
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
