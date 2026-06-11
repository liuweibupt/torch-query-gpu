from __future__ import annotations

import pytest

from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode


def test_operator_graph_returns_root_node() -> None:
    scan = TQPOperatorNode(
        node_id="n1",
        kind=OperatorKind.SCAN,
        name="SEQ_SCAN",
        children=(),
        metadata={"table": "lineitem"},
    )
    graph = TQPOperatorGraph(
        source_sql="select * from lineitem",
        query_id=None,
        root_id="n1",
        nodes=(scan,),
    )

    assert graph.root == scan
    assert graph.node_by_id("n1") == scan


def test_operator_graph_rejects_missing_root() -> None:
    scan = TQPOperatorNode(
        node_id="n1",
        kind=OperatorKind.SCAN,
        name="SEQ_SCAN",
    )

    with pytest.raises(ValueError, match="root node is missing"):
        TQPOperatorGraph(
            source_sql="select * from lineitem",
            query_id=None,
            root_id="missing",
            nodes=(scan,),
        )
