"""Bind DuckDB-lowered node metadata to stable TQP slots."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from tpch_torch.operator_graph import OperatorKind, TQPOutputColumn
from tpch_torch.operator_refs import TQPBoundExpression, TQPSlot, TQPSlotRef

_SLOT_REF_PATTERN = re.compile(r"#(?P<ordinal>\d+)")
_IDENTIFIER_PATTERN = re.compile(r'(?<![#."])([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)')
_RESERVED_IDENTIFIERS = frozenset(
    {
        "and",
        "as",
        "between",
        "case",
        "cast",
        "count",
        "date",
        "date_part",
        "distinct",
        "else",
        "end",
        "extract",
        "from",
        "in",
        "is",
        "max",
        "min",
        "not",
        "null",
        "or",
        "sum",
        "sum_no_overflow",
        "then",
        "when",
        "year",
    }
)


def bind_node_slots(
    *,
    node_id: str,
    kind: OperatorKind,
    metadata: Mapping[str, Any],
    child_slots: Sequence[Sequence[TQPSlot]],
    output_schema: Sequence[TQPOutputColumn],
    select_aliases: Mapping[str, str],
    is_root: bool,
) -> tuple[dict[str, Any], tuple[TQPSlot, ...]]:
    """Return metadata enriched with slot-bound expressions and node output slots."""

    child_flat = tuple(slot for slots in child_slots for slot in slots)
    metadata_dict = dict(metadata)
    if kind == OperatorKind.SCAN:
        return _bind_scan_node(node_id, metadata_dict)
    if kind == OperatorKind.PROJECT:
        return _bind_project_node(node_id, metadata_dict, child_flat, output_schema, select_aliases, is_root)
    if kind == OperatorKind.AGGREGATE:
        return _bind_aggregate_node(node_id, metadata_dict, child_flat, output_schema, is_root)
    if kind == OperatorKind.JOIN:
        return _bind_join_node(node_id, metadata_dict, child_slots)
    return metadata_dict, _pass_through_slots(node_id, child_flat, output_schema, is_root)


def _bind_scan_node(node_id: str, metadata: dict[str, Any]) -> tuple[dict[str, Any], tuple[TQPSlot, ...]]:
    table = _metadata_string(metadata, "table") or _metadata_string(metadata, "Table")
    projections = _metadata_tuple(metadata, "projections") or _metadata_tuple(metadata, "Projections")
    slots = tuple(_scan_slot(node_id, index, column, table) for index, column in enumerate(projections))
    metadata["output_slots"] = slots
    return metadata, slots


def _bind_project_node(
    node_id: str,
    metadata: dict[str, Any],
    child_slots: Sequence[TQPSlot],
    output_schema: Sequence[TQPOutputColumn],
    select_aliases: Mapping[str, str],
    is_root: bool,
) -> tuple[dict[str, Any], tuple[TQPSlot, ...]]:
    expressions = _metadata_tuple(metadata, "projections") or _metadata_tuple(metadata, "Projections")
    output_slots = _projection_output_slots(node_id, expressions, child_slots, output_schema, is_root)
    bound = tuple(
        _bound_expression(raw, _canonical_projection(raw, select_aliases), child_slots, output_slots[index])
        for index, raw in enumerate(expressions)
    )
    metadata["slot_projections"] = bound
    metadata["output_slots"] = output_slots
    return metadata, output_slots


def _bind_aggregate_node(
    node_id: str,
    metadata: dict[str, Any],
    child_slots: Sequence[TQPSlot],
    output_schema: Sequence[TQPOutputColumn],
    is_root: bool,
) -> tuple[dict[str, Any], tuple[TQPSlot, ...]]:
    groups = _metadata_tuple(metadata, "groups") or _metadata_tuple(metadata, "Groups")
    aggregates = _metadata_tuple(metadata, "aggregates") or _metadata_tuple(metadata, "Aggregates")
    output_slots = _aggregate_output_slots(node_id, groups, aggregates, output_schema, is_root)
    group_slots = output_slots[: len(groups)]
    aggregate_slots = output_slots[len(groups) :]
    metadata["slot_groups"] = tuple(_bound_expression(raw, raw, child_slots, slot) for raw, slot in zip(groups, group_slots))
    metadata["slot_aggregates"] = tuple(
        _bound_expression(raw, raw, child_slots, slot) for raw, slot in zip(aggregates, aggregate_slots)
    )
    metadata["output_slots"] = output_slots
    return metadata, output_slots


def _bind_join_node(
    node_id: str,
    metadata: dict[str, Any],
    child_slots: Sequence[Sequence[TQPSlot]],
) -> tuple[dict[str, Any], tuple[TQPSlot, ...]]:
    output_slots = _joined_output_slots(node_id, child_slots)
    conditions = _metadata_tuple(metadata, "conditions") or _metadata_tuple(metadata, "Conditions")
    metadata["slot_conditions"] = tuple(_bound_expression(raw, raw, output_slots, None) for raw in conditions)
    metadata["output_slots"] = output_slots
    return metadata, output_slots


def _projection_output_slots(
    node_id: str,
    expressions: Sequence[str],
    child_slots: Sequence[TQPSlot],
    output_schema: Sequence[TQPOutputColumn],
    is_root: bool,
) -> tuple[TQPSlot, ...]:
    schema = tuple(output_schema) if is_root and len(output_schema) == len(expressions) else ()
    slots = []
    for index, expression in enumerate(expressions):
        schema_column = schema[index] if schema else None
        child_ref = _single_ordinal_ref(expression, child_slots)
        name = schema_column.name if schema_column is not None else _projected_name(expression, child_ref)
        type_name = schema_column.type_name if schema_column is not None else _slot_type(child_ref)
        aliases = _projection_aliases(expression, child_ref)
        slots.append(TQPSlot(_slot_id(node_id, index), node_id, index, name, type_name, aliases, _slot_id_of(child_ref)))
    return tuple(slots)


def _aggregate_output_slots(
    node_id: str,
    groups: Sequence[str],
    aggregates: Sequence[str],
    output_schema: Sequence[TQPOutputColumn],
    is_root: bool,
) -> tuple[TQPSlot, ...]:
    expressions = (*groups, *aggregates)
    schema = tuple(output_schema) if is_root and len(output_schema) == len(expressions) else ()
    slots = []
    for index, expression in enumerate(expressions):
        schema_column = schema[index] if schema else None
        name = schema_column.name if schema_column is not None else expression
        type_name = schema_column.type_name if schema_column is not None else None
        slots.append(TQPSlot(_slot_id(node_id, index), node_id, index, name, type_name, (expression,)))
    return tuple(slots)


def _pass_through_slots(
    node_id: str,
    child_slots: Sequence[TQPSlot],
    output_schema: Sequence[TQPOutputColumn],
    is_root: bool,
) -> tuple[TQPSlot, ...]:
    schema = tuple(output_schema) if is_root and len(output_schema) == len(child_slots) else ()
    rebound = []
    for index, slot in enumerate(child_slots):
        schema_column = schema[index] if schema else None
        name = schema_column.name if schema_column is not None else slot.name
        type_name = schema_column.type_name if schema_column is not None else slot.type_name
        rebound.append(_rebound_slot(node_id, index, slot, name, type_name))
    return tuple(rebound)


def _joined_output_slots(node_id: str, child_slots: Sequence[Sequence[TQPSlot]]) -> tuple[TQPSlot, ...]:
    output = []
    for slot in tuple(slot for slots in child_slots for slot in slots):
        output.append(_rebound_slot(node_id, len(output), slot, slot.name, slot.type_name))
    return tuple(output)


def _scan_slot(node_id: str, index: int, column: str, table: str | None) -> TQPSlot:
    aliases = (column,) if table is None else (column, f"{table}.{column}")
    return TQPSlot(_slot_id(node_id, index), node_id, index, column, None, aliases)


def _rebound_slot(node_id: str, index: int, slot: TQPSlot, name: str, type_name: str | None) -> TQPSlot:
    aliases = (*slot.aliases, slot.slot_id)
    return TQPSlot(_slot_id(node_id, index), node_id, index, name, type_name, aliases, slot.slot_id)


def _bound_expression(
    raw: str,
    canonical: str,
    input_slots: Sequence[TQPSlot],
    output_slot: TQPSlot | None,
) -> TQPBoundExpression:
    refs, unresolved = _resolve_expression_refs(canonical, input_slots)
    return TQPBoundExpression(raw, canonical, refs, unresolved, output_slot)


def _resolve_expression_refs(expression: str, slots: Sequence[TQPSlot]) -> tuple[tuple[TQPSlotRef, ...], tuple[str, ...]]:
    refs: list[TQPSlotRef] = []
    unresolved: list[str] = []
    for ordinal in _slot_ordinals(expression):
        if ordinal < len(slots):
            refs.append(slots[ordinal].ref)
        else:
            unresolved.append(f"#{ordinal}")
    for identifier in _identifiers(expression):
        if _identifier_is_inside_string(expression, identifier):
            continue
        matches = _matching_slots(identifier, slots)
        if len(matches) == 1:
            refs.append(matches[0].ref)
        elif len(matches) > 1:
            unresolved.append(identifier)
    return _dedupe_refs(refs), tuple(dict.fromkeys(unresolved))


def _slot_ordinals(expression: str) -> tuple[int, ...]:
    return tuple(int(match.group("ordinal")) for match in _SLOT_REF_PATTERN.finditer(expression))


def _identifiers(expression: str) -> tuple[str, ...]:
    identifiers = []
    for match in _IDENTIFIER_PATTERN.finditer(expression):
        identifier = match.group(1)
        if identifier.lower() not in _RESERVED_IDENTIFIERS:
            identifiers.append(identifier)
    return tuple(dict.fromkeys(identifiers))


def _matching_slots(identifier: str, slots: Sequence[TQPSlot]) -> tuple[TQPSlot, ...]:
    lowered = identifier.lower()
    return tuple(slot for slot in slots if lowered in {alias.lower() for alias in slot.aliases})


def _identifier_is_inside_string(expression: str, identifier: str) -> bool:
    start = expression.find(identifier)
    return start >= 0 and expression[:start].count("'") % 2 == 1


def _dedupe_refs(refs: Sequence[TQPSlotRef]) -> tuple[TQPSlotRef, ...]:
    seen = set()
    deduped = []
    for ref in refs:
        if ref.slot_id in seen:
            continue
        seen.add(ref.slot_id)
        deduped.append(ref)
    return tuple(deduped)


def _canonical_projection(expression: str, select_aliases: Mapping[str, str]) -> str:
    return select_aliases.get(expression, expression)


def _single_ordinal_ref(expression: str, child_slots: Sequence[TQPSlot]) -> TQPSlot | None:
    match = re.fullmatch(r"#(\d+)", expression.strip())
    if match is None:
        return None
    ordinal = int(match.group(1))
    return child_slots[ordinal] if ordinal < len(child_slots) else None


def _projected_name(expression: str, child_ref: TQPSlot | None) -> str:
    return child_ref.name if child_ref is not None else expression


def _slot_type(child_ref: TQPSlot | None) -> str | None:
    return None if child_ref is None else child_ref.type_name


def _slot_id_of(child_ref: TQPSlot | None) -> str | None:
    return None if child_ref is None else child_ref.slot_id


def _projection_aliases(expression: str, child_ref: TQPSlot | None) -> tuple[str, ...]:
    return (expression,) if child_ref is None else (expression, *child_ref.aliases)


def _metadata_tuple(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None or value == "":
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str | None:
    values = _metadata_tuple(metadata, key)
    return values[0] if values else None


def _slot_id(node_id: str, ordinal: int) -> str:
    return f"{node_id}.s{ordinal}"
