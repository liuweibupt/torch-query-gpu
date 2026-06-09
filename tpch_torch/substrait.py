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

    columns = tuple(read_node.get("baseSchema", {}).get("names", ()))
    _require_columns(columns)

    table_name = _read_table_name(read_node)
    if table_name != "lineitem":
        raise UnsupportedPlanError(f"expected lineitem read, found {table_name!r}")

    cutoff = _read_shipdate_cutoff(read_node)
    if cutoff != Q1_SHIPDATE_CUTOFF_YYYYMMDD:
        raise UnsupportedPlanError(f"expected Q1 shipdate cutoff 19980902, found {cutoff}")

    group_keys = _read_key_names(aggregate_node, columns, "groupingExpressions")
    if group_keys != Q1_GROUP_KEYS:
        raise UnsupportedPlanError(f"expected Q1 group keys {Q1_GROUP_KEYS}, found {group_keys}")

    order_keys = _read_sort_key_names(sort_node, columns)
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


def _read_table_name(read_node: dict[str, Any]) -> str:
    names = read_node.get("namedTable", {}).get("names", [])
    if not names:
        raise UnsupportedPlanError("read node does not contain namedTable.names")
    return str(names[-1])


def _read_shipdate_cutoff(read_node: dict[str, Any]) -> int:
    literals = list(_walk_key(read_node.get("filter", {}), "literal"))
    date_literals = [_literal_date_to_yyyymmdd(literal) for literal in literals if "date" in literal]
    if len(date_literals) != 1:
        raise UnsupportedPlanError("Q1 filter must contain one date literal")
    return date_literals[0]


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
    fields = list(_walk_key(selection_expr, "structField"))
    if len(fields) != 1:
        raise UnsupportedPlanError(f"expected one structField in selection, found {fields}")
    return int(fields[0].get("field", 0))
