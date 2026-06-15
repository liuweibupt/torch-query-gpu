"""Parent-required column analysis for physical-plan execution."""

from __future__ import annotations

from tpch_torch.operator_graph import TQPOperatorGraph, TQPOperatorNode


def parents_by_child(graph: TQPOperatorGraph) -> dict[str, tuple[str, ...]]:
    """Return parent node ids keyed by child node id."""

    parents: dict[str, list[str]] = {}
    for node in graph.nodes:
        for child_id in node.children:
            parents.setdefault(child_id, []).append(node.node_id)
    return {child_id: tuple(parent_ids) for child_id, parent_ids in parents.items()}


def required_columns_from_parents(
    graph: TQPOperatorGraph,
    parents: dict[str, tuple[str, ...]],
    node_id: str,
    metadata_list,
) -> tuple[str, ...]:
    """Return raw parent expressions that may reference columns from a child node."""

    required: list[str] = []
    for parent in _ancestor_nodes(graph, parents, node_id):
        required.extend(metadata_list(parent, "Conditions"))
        required.extend(metadata_list(parent, "Expression"))
        required.extend(metadata_list(parent, "Projections"))
        required.extend(metadata_list(parent, "Order By"))
    return tuple(required)


def _ancestor_nodes(
    graph: TQPOperatorGraph,
    parents: dict[str, tuple[str, ...]],
    node_id: str,
) -> tuple[TQPOperatorNode, ...]:
    seen: set[str] = set()
    ordered: list[TQPOperatorNode] = []
    stack = list(parents.get(node_id, ()))
    while stack:
        parent_id = stack.pop()
        if parent_id in seen:
            continue
        seen.add(parent_id)
        ordered.append(graph.node_by_id(parent_id))
        stack.extend(parents.get(parent_id, ()))
    return tuple(ordered)
