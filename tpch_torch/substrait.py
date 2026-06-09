"""Compile the supported TPC-H Q1 Substrait plan shape into an executable plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

Q1_REQUIRED_COLUMNS = (
    "l_returnflag",
    "l_linestatus",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_shipdate",
)
Q1_GROUP_KEYS = ("l_returnflag", "l_linestatus")
Q1_ORDER_KEYS = ("l_returnflag", "l_linestatus")
Q1_SHIPDATE_CUTOFF_YYYYMMDD = 19980902
SUBSTRAIT_DATE_EPOCH = date(1970, 1, 1)
PROJECT_EXPRESSION_PREFIX = "__project_expr_"


class UnsupportedPlanError(ValueError):
    """Raised when a Substrait plan is outside this prototype's Q1 subset."""


@dataclass(frozen=True)
class Q1Plan:
    """Validated execution parameters for the supported TPC-H Q1 pipeline."""

    table_name: str
    shipdate_cutoff_yyyymmdd: int
    required_columns: tuple[str, ...]
    group_keys: tuple[str, str]
    order_keys: tuple[str, str]


def compile_q1_substrait_plan(plan_json: dict[str, Any]) -> Q1Plan:
    """Validate a DuckDB Substrait JSON plan and return Q1 execution metadata."""

    read_node = _find_single_node(plan_json, "read")
    aggregate_node = _find_single_node(plan_json, "aggregate")
    sort_node = _find_single_node(plan_json, "sort")

    read_columns = tuple(read_node.get("baseSchema", {}).get("names", ()))
    _require_columns(read_columns)

    table_name = _read_table_name(read_node)
    if table_name != "lineitem":
        raise UnsupportedPlanError(f"expected lineitem read, found {table_name!r}")

    cutoff = _read_shipdate_cutoff(plan_json)
    if cutoff != Q1_SHIPDATE_CUTOFF_YYYYMMDD:
        raise UnsupportedPlanError(f"expected Q1 shipdate cutoff 19980902, found {cutoff}")

    aggregate_input_columns = _read_aggregate_input_columns(aggregate_node, read_node, read_columns)
    group_keys = _read_key_names(aggregate_node, aggregate_input_columns, "groupingExpressions")
    if group_keys != Q1_GROUP_KEYS:
        raise UnsupportedPlanError(f"expected Q1 group keys {Q1_GROUP_KEYS}, found {group_keys}")

    order_keys = _read_sort_key_names(sort_node, _read_root_output_columns(plan_json))
    if order_keys != Q1_ORDER_KEYS:
        raise UnsupportedPlanError(f"expected Q1 order keys {Q1_ORDER_KEYS}, found {order_keys}")

    return Q1Plan(
        table_name=table_name,
        shipdate_cutoff_yyyymmdd=cutoff,
        required_columns=Q1_REQUIRED_COLUMNS,
        group_keys=Q1_GROUP_KEYS,
        order_keys=Q1_ORDER_KEYS,
    )


def _find_single_node(value: Any, key: str) -> dict[str, Any]:
    matches = list(_walk_key(value, key))
    if not matches:
        raise UnsupportedPlanError(f"Substrait plan does not contain required {key} node")
    if len(matches) > 1:
        raise UnsupportedPlanError(f"Substrait plan contains multiple {key} nodes")
    node = matches[0]
    if not isinstance(node, dict):
        raise UnsupportedPlanError(f"Substrait {key} node is not an object")
    return node


def _walk_key(value: Any, key: str):
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from _walk_key(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_key(item, key)


def _require_columns(columns: tuple[str, ...]) -> None:
    missing = [name for name in Q1_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise UnsupportedPlanError(f"lineitem read missing required columns: {missing}")


def _read_aggregate_input_columns(
    aggregate_node: dict[str, Any], read_node: dict[str, Any], read_columns: tuple[str, ...]
) -> tuple[str, ...]:
    read_output_columns = _read_read_output_columns(read_node, read_columns)
    aggregate_input = aggregate_node.get("input", {})
    if not isinstance(aggregate_input, dict):
        raise UnsupportedPlanError("Q1 aggregate input is not an object")
    project_node = aggregate_input.get("project")
    if project_node is None:
        return read_output_columns
    if not isinstance(project_node, dict):
        raise UnsupportedPlanError("Q1 aggregate input project is not an object")
    return _read_project_output_columns(project_node, read_output_columns)


def _read_read_output_columns(
    read_node: dict[str, Any], read_columns: tuple[str, ...]
) -> tuple[str, ...]:
    projection = read_node.get("projection")
    if projection is None:
        return read_columns
    if not isinstance(projection, dict):
        raise UnsupportedPlanError("read projection is not an object")
    struct_items = projection.get("select", {}).get("structItems", [])
    if not struct_items:
        return read_columns
    return tuple(read_columns[_struct_item_field(item)] for item in struct_items)


def _read_project_output_columns(
    project_node: dict[str, Any], input_columns: tuple[str, ...]
) -> tuple[str, ...]:
    expressions = project_node.get("expressions", [])
    if not expressions:
        return input_columns
    return tuple(
        _read_project_expression_name(expression, input_columns, index)
        for index, expression in enumerate(expressions)
    )


def _read_project_expression_name(
    expression: dict[str, Any], input_columns: tuple[str, ...], index: int
) -> str:
    if "selection" in expression:
        return input_columns[_selection_field(expression)]
    return f"{PROJECT_EXPRESSION_PREFIX}{index}"


def _read_root_output_columns(plan_json: dict[str, Any]) -> tuple[str, ...]:
    root_node = _find_single_node(plan_json, "root")
    names = tuple(root_node.get("names", ()))
    if not names:
        raise UnsupportedPlanError("Q1 root must contain output names to validate sort keys")
    return names


def _read_table_name(read_node: dict[str, Any]) -> str:
    names = read_node.get("namedTable", {}).get("names", [])
    if not names:
        raise UnsupportedPlanError("read node does not contain namedTable.names")
    return str(names[-1])


def _read_shipdate_cutoff(plan_json: dict[str, Any]) -> int:
    literals = list(_walk_key(plan_json, "literal"))
    date_literals = [_literal_date_to_yyyymmdd(literal) for literal in literals if "date" in literal]
    if len(date_literals) > 1:
        raise UnsupportedPlanError("Q1 plan contains multiple date literals")
    if date_literals:
        return date_literals[0]
    # DuckDB 1.2.x can export Q1 Substrait JSON without preserving the scan
    # filter literal. The compiler only accepts the canonical Q1 plan shape, so
    # use the canonical cutoff instead of silently dropping the filter.
    return Q1_SHIPDATE_CUTOFF_YYYYMMDD


def _literal_date_to_yyyymmdd(literal: dict[str, Any]) -> int:
    raw_date = literal["date"]
    if isinstance(raw_date, str) and raw_date.isdigit():
        raw_date = int(raw_date)
    if isinstance(raw_date, int) and raw_date > 10_000_000:
        return raw_date
    if isinstance(raw_date, int):
        return _date_to_yyyymmdd(SUBSTRAIT_DATE_EPOCH + timedelta(days=raw_date))
    if isinstance(raw_date, str):
        return int(raw_date.replace("-", ""))
    raise UnsupportedPlanError(f"unsupported Substrait date literal: {raw_date!r}")


def _date_to_yyyymmdd(value: date) -> int:
    return (value.year * 10_000) + (value.month * 100) + value.day


def _read_key_names(
    aggregate_node: dict[str, Any], columns: tuple[str, ...], grouping_key: str
) -> tuple[str, ...]:
    groupings = aggregate_node.get("groupings", [])
    if len(groupings) != 1:
        raise UnsupportedPlanError("Q1 aggregate must contain exactly one grouping set")
    selections = groupings[0].get(grouping_key, [])
    return tuple(columns[_selection_field(selection)] for selection in selections)


def _read_sort_key_names(sort_node: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    sorts = sort_node.get("sorts", [])
    if not sorts:
        raise UnsupportedPlanError("Q1 sort node must contain sort keys")
    return tuple(columns[_selection_field(sort.get("expr", sort))] for sort in sorts)


def _selection_field(selection_expr: dict[str, Any]) -> int:
    selection = selection_expr.get("selection")
    if not isinstance(selection, dict):
        raise UnsupportedPlanError(f"expected selection expression, found {selection_expr}")
    direct_reference = selection.get("directReference", {})
    if not isinstance(direct_reference, dict):
        raise UnsupportedPlanError(f"selection directReference is not an object: {selection_expr}")
    struct_field = direct_reference.get("structField")
    if not isinstance(struct_field, dict):
        raise UnsupportedPlanError(f"selection does not contain structField: {selection_expr}")
    return int(struct_field.get("field", 0))


def _struct_item_field(struct_item: dict[str, Any]) -> int:
    if not isinstance(struct_item, dict) or "field" not in struct_item:
        raise UnsupportedPlanError(f"read projection structItem missing field: {struct_item}")
    return int(struct_item["field"])
