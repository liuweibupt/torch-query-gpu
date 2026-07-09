"""Alias/key helper functions for physical join execution."""

from __future__ import annotations

import re
from typing import Sequence

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def existing_aliases(table: PhysicalTable, value: PhysicalValue) -> tuple[str, ...]:
    names = []
    aliases = getattr(table, "aliases", {}) or {}
    for name in (*tuple(table.columns), *tuple(aliases)):
        try:
            if table.columns[name] is value:
                names.append(name)
        except KeyError:
            continue
    return tuple(dict.fromkeys(names))


def matches_any_key(column: str, keys: Sequence[str]) -> bool:
    return any(same_column(column, key) for key in keys)


def same_column(left: str, right: str) -> bool:
    left_name = strip_unique_suffix(unqualified(left))
    right_name = strip_unique_suffix(unqualified(right))
    return left == right or left_name == right_name or left_name in identifier_tokens(right)


def unqualified(expression: str) -> str:
    return expression.replace('"', "").strip().rsplit(".", 1)[-1]


def strip_unique_suffix(name: str) -> str:
    base, separator, suffix = name.rpartition("__")
    return base if separator and suffix.isdigit() else name


def identifier_tokens(expression: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z_][\w]*", expression.replace('"', "")))


def has_positional_reference(expressions: Sequence[str]) -> bool:
    return any(re.search(r"#\d+", expression) is not None for expression in expressions)
