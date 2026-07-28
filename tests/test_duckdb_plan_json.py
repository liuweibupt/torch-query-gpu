import duckdb

from tpch_torch.duckdb_plan_json import (
    describe_output_schema,
    export_duckdb_physical_plan_json,
    lower_duckdb_json_to_operator_graph,
)


def test_lower_duckdb_json_adds_stable_metadata_and_output_schema():
    con = duckdb.connect()
    con.execute("create table t(a int, b decimal(10, 2))")
    sql = "select a as x, b + 1 as y from t where a > 0"

    graph = lower_duckdb_json_to_operator_graph(
        sql,
        None,
        export_duckdb_physical_plan_json(con, sql),
        output_schema=describe_output_schema(con, sql),
    )
    project = graph.root
    scan = graph.node_by_id(project.children[0])

    assert graph.output_names == ("x", "y")
    assert graph.output_types == ("INTEGER", "DECIMAL(13,2)")
    assert project.metadata["projections"] == ("x", "y")
    assert project.metadata["projection_count"] == 2
    assert project.metadata["output_names"] == ("x", "y")
    assert project.metadata["output_types"] == ("INTEGER", "DECIMAL(13,2)")
    assert project.output_slots[0].name == "x"
    assert project.output_slots[0].type_name == "INTEGER"
    assert scan.metadata["table"] == "t"
    assert scan.metadata["projections"] == ("a", "b")
    assert scan.output_slots[0].slot_id == "n0_0.s0"
    assert scan.output_slots[0].aliases == ("a", "t.a")
    assert isinstance(scan.metadata["estimated_cardinality"], int)


def test_output_aliases_come_from_duckdb_describe_not_sql_alias_regex():
    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    sql = "select sum(b) as total from t"

    graph = lower_duckdb_json_to_operator_graph(
        sql,
        None,
        export_duckdb_physical_plan_json(con, sql),
        output_schema=describe_output_schema(con, sql),
    )

    assert graph.output_names == ("total",)
    assert graph.output_types == ("HUGEINT",)
    assert graph.root.metadata["output_names"] == ("total",)
    assert graph.root.metadata["aggregates"] == ("sum_no_overflow(#0)",)
    aggregate = graph.root.metadata["slot_aggregates"][0]
    assert aggregate.raw == "sum_no_overflow(#0)"
    assert aggregate.refs[0].name == "b"
    assert aggregate.refs[0].slot_id == "n0_0.s0"
    assert aggregate.output_slot.name == "total"
    assert aggregate.expression.kind == "call"
    assert aggregate.expression.value == "sum_no_overflow"
    assert aggregate.expression.children[0].ref.name == "b"


def test_sirius_frontend_carries_schema_and_alias_metadata_before_execution(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.runner import compile_tqp_plan, run_sql_with_frontend

    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    con.execute("insert into t values (1, 2)")
    sql = "select a as x, b + 1 as y from t where a > 0"
    plan = compile_tqp_plan(con, sql, "sirius")

    assert plan.operator_graph.output_names == ("x", "y")
    assert plan.operator_graph.output_types == ("INTEGER", "INTEGER")
    assert plan.operator_graph.select_aliases == {"x": "a", "y": "(b + 1)"}
    project = plan.operator_graph.root
    assert project.metadata["slot_projections"][0].canonical == "a"
    assert project.metadata["slot_projections"][0].refs[0].name == "a"
    assert project.metadata["slot_projections"][0].expression.kind == "slot_ref"
    assert project.metadata["slot_projections"][1].canonical == "(b + 1)"
    assert project.metadata["slot_projections"][1].refs[0].name == "b"
    expression = project.metadata["slot_projections"][1].expression
    assert expression.kind == "binary"
    assert expression.value == "+"
    assert expression.children[0].ref.name == "b"
    assert expression.children[1].kind == "literal"
    assert expression.children[1].value == 1

    def fail_late_sql_alias_parse(sql_text):
        raise AssertionError("backend should use graph.select_aliases")

    def fail_late_describe(con_arg, sql_text):
        raise AssertionError("backend should use graph.output_names")

    monkeypatch.setattr(physical, "select_expressions_by_alias", fail_late_sql_alias_parse)
    monkeypatch.setattr(physical, "_describe_aliases", fail_late_describe)

    result = run_sql_with_frontend(con, sql, device="cpu", frontend="sirius")

    assert result.rows == [{"x": 1, "y": 3}]


def test_slot_bound_join_condition_has_expression_ast():
    from tpch_torch.runner import compile_tqp_plan
    from tpch_torch.operator_graph import OperatorKind

    con = duckdb.connect()
    con.execute("create table t(a int, b int)")
    con.execute("create table u(c int, d int)")
    con.execute("insert into t values (1, 2)")
    con.execute("insert into u values (1, 3)")

    graph = compile_tqp_plan(con, "select a, c from t join u on t.a = u.c", "sirius").operator_graph
    join = next(node for node in graph.nodes if node.kind == OperatorKind.JOIN)
    condition = join.metadata["slot_conditions"][0]

    assert condition.raw == "a = c"
    assert [ref.name for ref in condition.refs] == ["a", "c"]
    assert condition.expression.kind == "binary"
    assert condition.expression.value == "="
    assert [child.ref.name for child in condition.expression.children] == ["a", "c"]


def test_slot_bound_expression_preserves_decimal_literal():
    from decimal import Decimal
    from tpch_torch.frontend import compile_sirius_plan

    con = duckdb.connect()
    con.execute("create table t(amount decimal(10, 2))")
    sql = "select amount + 0.05::decimal(3,2) as adjusted from t"

    graph = compile_sirius_plan(con, sql).operator_graph
    expression = graph.root.metadata["slot_projections"][0].expression

    assert expression.kind == "binary"
    assert expression.value == "+"
    assert expression.children[0].ref.name == "amount"
    assert expression.children[1].kind == "literal"
    assert expression.children[1].value == Decimal("0.05")


def test_sirius_frontend_carries_scan_decimal_type_metadata():
    from tpch_torch.frontend import compile_sirius_plan
    from tpch_torch.operator_graph import OperatorKind

    con = duckdb.connect()
    con.execute("create table t(id bigint, amount decimal(10, 2))")
    graph = compile_sirius_plan(con, "select amount from t").operator_graph

    scan = next(node for node in graph.nodes if node.kind == OperatorKind.SCAN)

    assert scan.metadata["scan_output_types"] == {"id": "BIGINT", "amount": "DECIMAL(10,2)"}
    assert scan.output_slots[0].name == "amount"
    assert scan.output_slots[0].type_name == "DECIMAL(10,2)"
