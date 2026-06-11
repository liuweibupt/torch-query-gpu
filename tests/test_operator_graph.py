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

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.sql import get_tpch_query


def test_sirius_frontend_lowers_tpch_query_to_operator_graph(tmp_path) -> None:
    db = tmp_path / "tpch.duckdb"
    con = connect_database(db)
    try:
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")
        con.execute("CALL dbgen(sf=0.01)")
        plan = compile_sirius_plan(con, get_tpch_query(con, 1))
    finally:
        con.close()

    assert plan.operator_graph is not None
    assert plan.operator_graph.query_id == 1
    assert plan.operator_graph.root.name
    assert any(node.name == "SEQ_SCAN" for node in plan.operator_graph.nodes)


def test_sirius_frontend_lowers_all_tpch_queries_to_operator_graph(tmp_path) -> None:
    db = tmp_path / "tpch-all.duckdb"
    con = connect_database(db)
    try:
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")
        con.execute("CALL dbgen(sf=0.01)")
        missing = []
        for query_id in range(1, 23):
            plan = compile_sirius_plan(con, get_tpch_query(con, query_id))
            if plan.operator_graph is None:
                missing.append(query_id)
    finally:
        con.close()

    assert missing == []
