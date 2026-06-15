"""SQL alias recovery helpers for DuckDB physical plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class _TableAlias:
    table: str
    alias: str


@dataclass(frozen=True)
class _AliasJoinKey:
    alias: str
    column: str
    partner: str


def qualified_aliases_for_join_side(
    source_sql: str,
    base_table: str,
    column: str,
    own_keys: Sequence[str],
    other_keys: Sequence[str],
) -> tuple[str, ...]:
    """Return SQL alias-qualified names that identify a joined table side."""

    aliases = _aliases_for_join_side(source_sql, base_table, own_keys, other_keys)
    column_name = _unqualified(column)
    return tuple(f"{alias}.{column_name}" for alias in aliases)


def _aliases_for_join_side(
    source_sql: str,
    base_table: str,
    own_keys: Sequence[str],
    other_keys: Sequence[str],
) -> tuple[str, ...]:
    tables = _table_aliases(source_sql)
    if not tables:
        return ()
    join_keys = _alias_join_keys(source_sql)
    base = _unqualified(base_table).lower()
    own_names = {_unqualified(key).lower() for key in own_keys}
    other_names = {_unqualified(key).lower() for key in other_keys}
    matched = [
        table.alias
        for table in tables
        if table.table.lower() == base
        and _alias_matches_join_keys(table.alias, own_names, other_names, join_keys)
    ]
    return tuple(dict.fromkeys(matched))


def _table_aliases(sql: str) -> tuple[_TableAlias, ...]:
    pattern = re.compile(
        r"(?:\bFROM\b|\bJOIN\b|,)\s+"
        r"(?P<table>[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)"
        r"(?:\s+(?:AS\s+)?(?P<alias>(?!" + _RESERVED_PATTERN + r"\b)[A-Za-z_][\w]*))?\b",
        re.I,
    )
    aliases = []
    for match in pattern.finditer(sql):
        if _is_select_alias_match(sql, match.start()):
            continue
        table = _unqualified(match.group("table"))
        alias = match.group("alias")
        if alias is None:
            continue
        if alias.upper() in _RESERVED_WORDS or alias == table:
            continue
        aliases.append(_TableAlias(table, alias))
    return tuple(dict.fromkeys(aliases))


def _alias_join_keys(sql: str) -> tuple[_AliasJoinKey, ...]:
    pattern = re.compile(
        r"(?P<left>(?:(?:[A-Za-z_][\w]*)\.)?[A-Za-z_][\w]*)"
        r"\s*=\s*"
        r"(?P<right>(?:(?:[A-Za-z_][\w]*)\.)?[A-Za-z_][\w]*)",
        re.I,
    )
    keys = []
    for match in pattern.finditer(sql):
        left = _split_qualified(match.group("left"))
        right = _split_qualified(match.group("right"))
        if left[0] is not None:
            keys.append(_AliasJoinKey(left[0], left[1], right[1]))
        if right[0] is not None:
            keys.append(_AliasJoinKey(right[0], right[1], left[1]))
    return tuple(keys)


def _alias_matches_join_keys(
    alias: str,
    own_keys: set[str],
    other_keys: set[str],
    join_keys: Sequence[_AliasJoinKey],
) -> bool:
    for key in join_keys:
        if key.alias != alias:
            continue
        if key.column.lower() in own_keys and key.partner.lower() in other_keys:
            return True
    return False


def _unqualified(name: str) -> str:
    return name.replace('"', "").strip().rsplit(".", 1)[-1].split("__", 1)[0]


def _split_qualified(expression: str) -> tuple[str | None, str]:
    clean = expression.replace('"', "").strip()
    if "." not in clean:
        return None, clean
    alias, column = clean.rsplit(".", 1)
    return alias, column


def _is_select_alias_match(sql: str, start: int) -> bool:
    return sql[start] == "," and _inside_select_list(sql, start)


def _nearest_prior_keyword(sql: str, end: int) -> str | None:
    candidates = ("SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "ORDER", "HAVING")
    last_position = -1
    last_keyword = None
    for keyword in candidates:
        pattern = re.compile(rf"\b{keyword}\b", re.I)
        for match in pattern.finditer(sql[:end]):
            if match.start() > last_position:
                last_position = match.start()
                last_keyword = keyword
    return last_keyword


def _inside_select_list(sql: str, index: int) -> bool:
    select_start = _nearest_prior_keyword_position(sql, index, "SELECT")
    if select_start < 0:
        return False
    from_start = _nearest_prior_keyword_position(sql, index, "FROM")
    return from_start < select_start


def _nearest_prior_keyword_position(sql: str, end: int, keyword: str) -> int:
    position = -1
    pattern = re.compile(rf"\b{keyword}\b", re.I)
    for match in pattern.finditer(sql[:end]):
        position = match.start()
    return position


_RESERVED_WORDS = frozenset(
    {
        "WHERE",
        "JOIN",
        "ON",
        "GROUP",
        "ORDER",
        "HAVING",
        "LIMIT",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "CROSS",
        "AS",
    }
)

_RESERVED_PATTERN = "|".join(sorted(_RESERVED_WORDS))
