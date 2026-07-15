import duckdb

from tpch_torch.duckdb_plan_json import (
    describe_output_columns,
    export_duckdb_physical_plan_json,
    lower_duckdb_json_to_operator_graph,
)


def test_lower_duckdb_json_adds_stable_metadata_for_scalar_and_list_fields():
    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    sql = "select a as x, b + 1 as y from t where a > 0"

    graph = lower_duckdb_json_to_operator_graph(
        sql,
        None,
        export_duckdb_physical_plan_json(con, sql),
        output_names=describe_output_columns(con, sql),
    )
    project = graph.root
    scan = graph.node_by_id(project.children[0])

    assert graph.output_names == ("x", "y")
    assert project.metadata["projections"] == ("x", "y")
    assert project.metadata["projection_count"] == 2
    assert project.metadata["output_names"] == ("x", "y")
    assert scan.metadata["table"] == "t"
    assert scan.metadata["projections"] == ("a", "b")
    assert isinstance(scan.metadata["estimated_cardinality"], int)


def test_output_aliases_come_from_duckdb_describe_not_sql_alias_regex():
    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    sql = "select sum(b) as total from t"

    graph = lower_duckdb_json_to_operator_graph(
        sql,
        None,
        export_duckdb_physical_plan_json(con, sql),
        output_names=describe_output_columns(con, sql),
    )

    assert graph.output_names == ("total",)
    assert graph.root.metadata["output_names"] == ("total",)
    assert graph.root.metadata["aggregates"] == ("sum_no_overflow(#0)",)


def test_sirius_frontend_carries_alias_metadata_before_physical_execution(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.runner import compile_tqp_plan, run_sql_with_frontend

    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    con.execute("insert into t values (1, 2)")
    sql = "select a as x, b + 1 as y from t where a > 0"
    plan = compile_tqp_plan(con, sql, "sirius")

    assert plan.operator_graph.output_names == ("x", "y")
    assert plan.operator_graph.select_aliases == {"x": "a", "y": "b + 1"}

    def fail_late_sql_alias_parse(sql_text):
        raise AssertionError("backend should use graph.select_aliases")

    def fail_late_describe(con_arg, sql_text):
        raise AssertionError("backend should use graph.output_names")

    monkeypatch.setattr(physical, "select_expressions_by_alias", fail_late_sql_alias_parse)
    monkeypatch.setattr(physical, "_describe_aliases", fail_late_describe)

    result = run_sql_with_frontend(con, sql, device="cpu", frontend="sirius")

    assert result.rows == [{"x": 1, "y": 3}]


def test_physical_executor_runs_with_canonical_only_metadata():
    from tpch_torch.backend.physical import PhysicalPlanExecutor
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    con.execute("insert into t values (1, 2), (3, 4)")
    scan = TQPOperatorNode(
        node_id="n0",
        kind=OperatorKind.SCAN,
        name="SEQ_SCAN",
        metadata={"table": "t", "projections": ("a", "b")},
    )
    filter_node = TQPOperatorNode(
        node_id="n1",
        kind=OperatorKind.FILTER,
        name="FILTER",
        children=("n0",),
        metadata={"expression": "a > 1"},
    )
    project = TQPOperatorNode(
        node_id="n2",
        kind=OperatorKind.PROJECT,
        name="PROJECTION",
        children=("n1",),
        metadata={"projections": ("a", "b + 1")},
    )
    graph = TQPOperatorGraph(
        source_sql="select a as x, b + 1 as y from t where a > 1",
        query_id=None,
        root_id="n2",
        nodes=(scan, filter_node, project),
        output_names=("x", "y"),
        select_aliases={"x": "a", "y": "b + 1"},
    )

    rows = PhysicalPlanExecutor(con, graph, device="cpu").execute()

    assert rows == [{"x": 3, "y": 5}]
