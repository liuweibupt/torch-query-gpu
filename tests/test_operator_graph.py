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

from tpch_torch.duckdb_bridge import connect_database, generate_tpch
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.sql import get_tpch_query


def test_sirius_frontend_lowers_tpch_query_to_operator_graph(tmp_path) -> None:
    db = tmp_path / "tpch.duckdb"
    con = connect_database(db)
    try:
        generate_tpch(con, scale_factor=0.01)
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
        generate_tpch(con, scale_factor=0.01)
        missing = []
        for query_id in range(1, 23):
            plan = compile_sirius_plan(con, get_tpch_query(con, query_id))
            if plan.operator_graph is None:
                missing.append(query_id)
    finally:
        con.close()

    assert missing == []


def test_q1_and_q6_lower_to_real_operator_graph_roots(tmp_path) -> None:
    db = tmp_path / "tpch-q1-q6.duckdb"
    con = connect_database(db)
    try:
        generate_tpch(con, scale_factor=0.01)
        for query_id in (1, 6):
            plan = compile_sirius_plan(con, get_tpch_query(con, query_id))
            assert plan.operator_graph is not None
            assert plan.operator_graph.root.kind != OperatorKind.COMPILED_TPCH
            kinds = {node.kind for node in plan.operator_graph.nodes}
            assert OperatorKind.SCAN in kinds
            assert OperatorKind.AGGREGATE in kinds
    finally:
        con.close()


def test_all_tpch_queries_have_real_lowered_duckdb_graph_roots(tmp_path) -> None:
    db = tmp_path / "tpch-real-roots.duckdb"
    con = connect_database(db)
    try:
        generate_tpch(con, scale_factor=0.01)
        compiled_roots = []
        missing_scan = []
        for query_id in range(1, 23):
            plan = compile_sirius_plan(con, get_tpch_query(con, query_id))
            assert plan.operator_graph is not None
            if plan.operator_graph.root.kind == OperatorKind.COMPILED_TPCH:
                compiled_roots.append(query_id)
            if not any(node.kind == OperatorKind.SCAN for node in plan.operator_graph.nodes):
                missing_scan.append(query_id)
    finally:
        con.close()

    assert compiled_roots == []
    assert missing_scan == []


def test_graph_executor_has_no_complex_tpch_compatibility_entrypoints() -> None:
    import tpch_torch.backend.graph as graph_module

    assert not hasattr(graph_module, "_execute_complex_tpch_graph")
    assert not hasattr(graph_module, "_execute_compiled_tpch_node")
    assert not hasattr(graph_module, "_EXECUTOR_BY_QUERY")


def test_tpch_graph_query_modules_compose_common_graph_nodes() -> None:
    import ast
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[1] / "tpch_torch" / "backend"
    offenders: list[str] = []
    missing_graph_nodes: list[str] = []
    for module_path in sorted(backend_dir.glob("tpch_graph_q*.py")):
        tree = ast.parse(module_path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        if "tpch_torch.relational" in imports:
            offenders.append(module_path.name)
        if "tpch_torch.backend.graph_nodes" not in imports:
            missing_graph_nodes.append(module_path.name)

    assert offenders == []
    assert missing_graph_nodes == []


def test_complex_tpch_graph_modules_use_explicit_subquery_nodes() -> None:
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[1] / "tpch_torch" / "backend"
    expected = {
        "tpch_graph_q02.py": ("GroupedScalarSubqueryNode",),
        "tpch_graph_q04.py": ("SemiJoinNode",),
        "tpch_graph_q11.py": ("ScalarSubqueryNode",),
        "tpch_graph_q15.py": ("MaterializedCTENode", "ScalarSubqueryNode"),
        "tpch_graph_q16.py": ("AntiJoinNode",),
        "tpch_graph_q17.py": ("GroupedScalarSubqueryNode",),
        "tpch_graph_q18.py": ("SemiJoinNode",),
        "tpch_graph_q20.py": ("SemiJoinNode", "GroupedScalarSubqueryNode"),
        "tpch_graph_q21.py": ("AntiJoinNode", "SemiJoinNode"),
        "tpch_graph_q22.py": ("AntiJoinNode", "ScalarSubqueryNode"),
    }
    missing: dict[str, tuple[str, ...]] = {}
    for filename, node_names in expected.items():
        source = (backend_dir / filename).read_text()
        absent = tuple(node_name for node_name in node_names if node_name not in source)
        if absent:
            missing[filename] = absent

    assert missing == {}
