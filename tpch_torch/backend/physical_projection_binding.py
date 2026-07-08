"""Parent-aware projection binding helpers for DuckDB physical refs."""

from __future__ import annotations

import re
from typing import Sequence

from tpch_torch.backend.physical_types import PhysicalTable


def parent_bound_projection_expression(
    child: PhysicalTable,
    expression: str,
    projection_count: int,
    parent_required: Sequence[str],
) -> str | None:
    """Resolve a single positional projection to a parent-required named value."""

    if projection_count != 1 or not _is_projection_position(expression):
        return None
    candidates = _parent_named_column_candidates(parent_required, child)
    return candidates[0] if len(candidates) == 1 else None


def _parent_named_column_candidates(
    parent_required: Sequence[str],
    child: PhysicalTable,
) -> tuple[str, ...]:
    candidates = []
    for expression in parent_required:
        stripped = expression.replace('"', "").strip()
        if not re.fullmatch(r"(?:[A-Za-z_][\w]*\.)?[A-Za-z_][\w]*", stripped):
            continue
        try:
            child.value_named(stripped)
        except KeyError:
            continue
        candidates.append(stripped)
    return tuple(dict.fromkeys(candidates))


def _is_projection_position(expression: str) -> bool:
    stripped = expression.strip()
    return stripped.startswith("#") and stripped[1:].isdigit()
