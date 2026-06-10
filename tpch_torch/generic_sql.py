"""Small generic SQL subset parser for TQP plans.

This parser is intentionally explicit about the supported subset. DuckDB still
performs admission in the Sirius-like frontend; this module describes what the
PyTorch backend can execute without falling back to DuckDB result rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from tpch_torch.errors import UnsupportedPlanError

ProjectionKind = Literal["column", "count_star", "count", "sum", "min", "max", "avg", "mul_const"]
FilterKind = Literal["comparison", "in", "like", "and", "or", "not"]
FilterOperator = Literal["=", "!=", "<>", ">", ">=", "<", "<="]


@dataclass(frozen=True)
class GenericProjection:
    kind: ProjectionKind
    alias: str
    column: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class GenericFilter:
    column: str = ""
    kind: FilterKind = "comparison"
    operator: FilterOperator | None = None
    value: int | float | str | None = None
    values: tuple[int | float | str, ...] = ()
    children: tuple["GenericFilter", ...] = ()

    def __getitem__(self, index: int) -> "GenericFilter":
        if self.kind == "and":
            return self.children[index]
        if index == 0:
            return self
        raise IndexError(index)


@dataclass(frozen=True)
class GenericOrderBy:
    column: str
    descending: bool = False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.column == other and not self.descending
        if isinstance(other, GenericOrderBy):
            return self.column == other.column and self.descending == other.descending
        return False


@dataclass(frozen=True)
class GenericSQLPlan:
    table: str
    projections: tuple[GenericProjection, ...]
    filters: GenericFilter | None = None
    group_by: tuple[str, ...] = ()
    order_by: tuple[GenericOrderBy, ...] = ()
    limit: int | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for projection in self.projections:
            if projection.column is not None:
                columns.append(projection.column)
        if self.filters is not None:
            columns.extend(_filter_columns(self.filters))
        columns.extend(self.group_by)
        columns.extend(order.column for order in self.order_by)
        return tuple(dict.fromkeys(columns))


def parse_generic_sql(sql: str) -> GenericSQLPlan:
    normalized = _normalize_sql(sql)
    lowered = normalized.lower()
    if " join " in lowered:
        raise UnsupportedPlanError("generic SQL joins are not supported yet")
    if any(token in lowered for token in (" over ", " union ", " intersect ", " except ", " with ", " having ")):
        raise UnsupportedPlanError("generic SQL feature is not supported yet")
    return _parse_select(normalized)


def _parse_select(sql: str) -> GenericSQLPlan:
    match = re.fullmatch(
        r"select\s+(?P<select>.+?)\s+from\s+(?P<table>[A-Za-z_][\w]*)"
        r"(?:\s+where\s+(?P<where>.+?))?"
        r"(?:\s+group\s+by\s+(?P<group>.+?))?"
        r"(?:\s+order\s+by\s+(?P<order>.+?))?"
        r"(?:\s+limit\s+(?P<limit>\d+))?",
        sql,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise UnsupportedPlanError("generic SQL subset only supports simple SELECT ... FROM queries")
    return GenericSQLPlan(
        table=match.group("table"),
        projections=_parse_projections(match.group("select")),
        filters=_parse_filter_expression(match.group("where")),
        group_by=_parse_column_list(match.group("group")),
        order_by=_parse_order_by(match.group("order")),
        limit=_parse_limit(match.group("limit")),
    )


def _parse_projections(raw: str) -> tuple[GenericProjection, ...]:
    return tuple(_parse_projection(item.strip()) for item in _split_csv(raw))


def _parse_projection(raw: str) -> GenericProjection:
    expression, alias = _split_alias(raw)
    count_match = re.fullmatch(r"count\s*\(\s*\*\s*\)", expression, flags=re.IGNORECASE)
    if count_match:
        return GenericProjection(kind="count_star", alias=alias or "count_star")
    agg_match = re.fullmatch(
        r"(?P<kind>count|sum|min|max|avg)\s*\(\s*(?P<column>[A-Za-z_][\w]*)\s*\)",
        expression,
        flags=re.IGNORECASE,
    )
    if agg_match:
        kind = agg_match.group("kind").lower()
        column = agg_match.group("column")
        return GenericProjection(kind=kind, column=column, alias=alias or f"{kind}({column})")
    mul_match = re.fullmatch(
        r"(?P<column>[A-Za-z_][\w]*)\s*\*\s*(?P<value>-?\d+(?:\.\d+)?)",
        expression,
        flags=re.IGNORECASE,
    )
    if mul_match:
        column = mul_match.group("column")
        return GenericProjection(kind="mul_const", column=column, value=float(mul_match.group("value")), alias=alias or expression)
    if re.fullmatch(r"[A-Za-z_][\w]*", expression):
        return GenericProjection(kind="column", column=expression, alias=alias or expression)
    raise UnsupportedPlanError(f"generic SQL projection is not supported: {raw}")


def _split_alias(raw: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(?P<expr>.+?)\s+as\s+(?P<alias>[A-Za-z_][\w]*)", raw, flags=re.IGNORECASE)
    if match:
        return match.group("expr").strip(), match.group("alias")
    return raw.strip(), None


def _parse_filter_expression(raw: str | None) -> GenericFilter | None:
    if raw is None:
        return None
    return _parse_filter_node(raw)


def _parse_filter_node(raw: str) -> GenericFilter:
    raw = raw.strip()
    or_parts = _split_keyword(raw, "or")
    if len(or_parts) > 1:
        return GenericFilter(kind="or", column="", children=tuple(_parse_filter_node(part) for part in or_parts))
    and_parts = _split_keyword(raw, "and")
    if len(and_parts) > 1:
        return GenericFilter(kind="and", column="", children=tuple(_parse_filter_node(part) for part in and_parts))
    if raw.lower().startswith("not "):
        return GenericFilter(kind="not", column="", children=(_parse_filter_node(raw[4:].strip()),))
    return _parse_filter(raw)


def _parse_filter(raw: str) -> GenericFilter:
    in_match = re.fullmatch(
        r"(?P<column>[A-Za-z_][\w]*)\s+in\s*\((?P<values>.+)\)",
        raw,
        flags=re.IGNORECASE,
    )
    if in_match is not None:
        return GenericFilter(
            kind="in",
            column=in_match.group("column"),
            values=tuple(_parse_literal(item.strip()) for item in _split_csv(in_match.group("values"))),
        )
    like_match = re.fullmatch(
        r"(?P<column>[A-Za-z_][\w]*)\s+like\s*(?P<value>'[^']*')",
        raw,
        flags=re.IGNORECASE,
    )
    if like_match is not None:
        return GenericFilter(kind="like", column=like_match.group("column"), value=_parse_literal(like_match.group("value")))
    match = re.fullmatch(
        r"(?P<column>[A-Za-z_][\w]*)\s*(?P<operator>=|!=|<>|>=|>|<=|<)\s*(?P<value>'[^']*'|-?\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise UnsupportedPlanError(f"generic SQL filter is not supported: {raw}")
    return GenericFilter(
        kind="comparison",
        column=match.group("column"),
        operator=match.group("operator"),
        value=_parse_literal(match.group("value")),
    )


def _parse_column_list(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    columns = tuple(item.strip() for item in raw.split(","))
    if any(re.fullmatch(r"[A-Za-z_][\w]*", column) is None for column in columns):
        raise UnsupportedPlanError(f"generic SQL column list is not supported: {raw}")
    return columns


def _parse_order_by(raw: str | None) -> tuple[GenericOrderBy, ...]:
    if raw is None:
        return ()
    return tuple(_parse_order_item(item.strip()) for item in _split_csv(raw))


def _parse_order_item(raw: str) -> GenericOrderBy:
    match = re.fullmatch(r"(?P<column>[A-Za-z_][\w]*)(?:\s+(?P<dir>asc|desc))?", raw, flags=re.IGNORECASE)
    if match is None:
        raise UnsupportedPlanError(f"generic SQL order item is not supported: {raw}")
    return GenericOrderBy(column=match.group("column"), descending=(match.group("dir") or "").lower() == "desc")


def _parse_limit(raw: str | None) -> int | None:
    return None if raw is None else int(raw)


def _parse_literal(raw: str) -> int | float | str:
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if "." in raw:
        return float(raw)
    return int(raw)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";"))


def _filter_columns(filter_: GenericFilter) -> tuple[str, ...]:
    if filter_.kind in {"and", "or", "not"}:
        columns: list[str] = []
        for child in filter_.children:
            columns.extend(_filter_columns(child))
        return tuple(columns)
    return (filter_.column,)


def _split_csv(raw: str) -> tuple[str, ...]:
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


def _split_keyword(raw: str, keyword: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    index = 0
    needle = f" {keyword.lower()} "
    lowered = raw.lower()
    while index < len(raw):
        char = raw[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        if not in_quote and depth == 0 and lowered.startswith(needle, index):
            parts.append(raw[start:index].strip())
            index += len(needle)
            start = index
            continue
        index += 1
    parts.append(raw[start:].strip())
    return tuple(part for part in parts if part)
