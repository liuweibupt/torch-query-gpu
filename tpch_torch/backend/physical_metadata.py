"""Stable metadata access for DuckDB-lowered physical graph nodes."""

from __future__ import annotations

from typing import Any

from tpch_torch.operator_graph import TQPOperatorNode

_CANONICAL_KEYS = {
    "Aggregates": "aggregates",
    "Conditions": "conditions",
    "CTE Index": "cte_index",
    "Delim Index": "delim_index",
    "Estimated Cardinality": "estimated_cardinality",
    "Expression": "expression",
    "Expressions": "expressions",
    "Filters": "filters",
    "Groups": "groups",
    "Join Type": "join_type",
    "Limit": "limit",
    "Order By": "order_by",
    "Projections": "projections",
    "Table": "table",
    "Table Index": "table_index",
    "Top": "top",
    "Type": "scan_type",
}
_RAW_KEYS = {canonical: raw for raw, canonical in _CANONICAL_KEYS.items()}


def metadata_value(node: TQPOperatorNode, key: str) -> Any:
    """Return canonical metadata first, then DuckDB raw-key compatibility data."""

    for candidate in _candidate_keys(key):
        if candidate in node.metadata:
            return node.metadata[candidate]
    return None


def metadata_list(node: TQPOperatorNode, key: str) -> tuple[str, ...]:
    """Return a stable tuple for DuckDB metadata that may be scalar or repeated."""

    value = metadata_value(node, key)
    if value is None or value == "":
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def metadata_string(node: TQPOperatorNode, key: str) -> str | None:
    """Return the first normalized metadata value for scalar physical fields."""

    values = metadata_list(node, key)
    return values[0] if values else None


def metadata_int(node: TQPOperatorNode, key: str) -> int | None:
    """Return an integer metadata value when DuckDB exposed one."""

    value = metadata_value(node, key)
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text.lstrip("-").isdigit():
        return None
    return int(text)


def _candidate_keys(key: str) -> tuple[str, ...]:
    canonical = _CANONICAL_KEYS.get(key, key)
    raw = _RAW_KEYS.get(canonical)
    candidates = [canonical]
    if raw is not None:
        candidates.append(raw)
    if key not in candidates:
        candidates.append(key)
    return tuple(candidates)
