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

ProjectionKind = Literal["column", "count_star", "sum", "mul_const"]
FilterOperator = Literal["=", "!=", "<>", ">", ">=", "<", "<="]


@dataclass(frozen=True)
class GenericProjection:
    kind: ProjectionKind
    alias: str
    column: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class GenericFilter:
    column: str
    operator: FilterOperator
    value: int | float | str


@dataclass(frozen=True)
class GenericSQLPlan:
    table: str
    projections: tuple[GenericProjection, ...]
    filters: tuple[GenericFilter, ...] = ()
    group_by: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()
    limit: int | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for projection in self.projections:
            if projection.column is not None:
                columns.append(projection.column)
        columns.extend(filter_.column for filter_ in self.filters)
        columns.extend(self.group_by)
        columns.extend(self.order_by)
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
        filters=_parse_filters(match.group("where")),
        group_by=_parse_column_list(match.group("group")),
        order_by=_parse_column_list(match.group("order")),
        limit=_parse_limit(match.group("limit")),
    )


def _parse_projections(raw: str) -> tuple[GenericProjection, ...]:
    return tuple(_parse_projection(item.strip()) for item in raw.split(","))


def _parse_projection(raw: str) -> GenericProjection:
    expression, alias = _split_alias(raw)
    count_match = re.fullmatch(r"count\s*\(\s*\*\s*\)", expression, flags=re.IGNORECASE)
    if count_match:
        return GenericProjection(kind="count_star", alias=alias or "count_star")
    sum_match = re.fullmatch(r"sum\s*\(\s*(?P<column>[A-Za-z_][\w]*)\s*\)", expression, flags=re.IGNORECASE)
    if sum_match:
        column = sum_match.group("column")
        return GenericProjection(kind="sum", column=column, alias=alias or f"sum({column})")
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


def _parse_filters(raw: str | None) -> tuple[GenericFilter, ...]:
    if raw is None:
        return ()
    parts = re.split(r"\s+and\s+", raw, flags=re.IGNORECASE)
    return tuple(_parse_filter(part.strip()) for part in parts)


def _parse_filter(raw: str) -> GenericFilter:
    match = re.fullmatch(
        r"(?P<column>[A-Za-z_][\w]*)\s*(?P<operator>=|!=|<>|>=|>|<=|<)\s*(?P<value>'[^']*'|-?\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise UnsupportedPlanError(f"generic SQL filter is not supported: {raw}")
    return GenericFilter(
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
