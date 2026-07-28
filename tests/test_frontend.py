from dataclasses import dataclass

import duckdb

from tpch_torch.frontend import compile_sirius_plan, compile_substrait_plan
from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.operator_graph import OperatorKind, TQPOutputColumn, TQPOperatorGraph, TQPOperatorNode


@dataclass(frozen=True)
class DummyLogicalPlan:
    logical_plan: str = "logical"
    logical_opt: str = "optimized"
    physical_plan: str = "physical"


def _dummy_graph(
    sql: str,
    query_id: int | None = 1,
    *,
    output_schema: tuple[TQPOutputColumn, ...] = (),
    select_aliases: dict[str, str] | None = None,
) -> TQPOperatorGraph:
    node = TQPOperatorNode(node_id="n0", kind=OperatorKind.SCAN, name="SEQ_SCAN")
    return TQPOperatorGraph(
        source_sql=sql,
        query_id=query_id,
        root_id="n0",
        nodes=(node,),
        output_schema=output_schema,
        select_aliases=select_aliases or {},
    )



def test_sirius_frontend_returns_tqp_plan_with_duckdb_metadata(monkeypatch):
    calls = []

    def export_logical(con, sql):
        calls.append((con, sql))
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)
    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_physical_plan_json", lambda con, sql: [{"name": "SEQ_SCAN"}])
    monkeypatch.setattr("tpch_torch.frontend.sirius.describe_output_schema", lambda con, sql: (TQPOutputColumn("sum_qty", "HUGEINT"),))
    monkeypatch.setattr("tpch_torch.frontend.sirius.select_expressions_by_alias", lambda con, sql: {"sum_qty": "sum(l_quantity)"})
    monkeypatch.setattr(
        "tpch_torch.frontend.sirius.lower_duckdb_json_to_operator_graph",
        lambda sql, query_id, plan_json, *, output_schema, select_aliases, table_schemas: _dummy_graph(
            sql, query_id, output_schema=output_schema, select_aliases=select_aliases
        ),
    )

    con = duckdb.connect()
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)

    assert plan.query_id == 1
    assert plan.source_sql == TPC_H_Q1_SQL
    assert plan.frontend == "sirius"
    assert plan.duckdb_metadata is not None
    assert plan.duckdb_metadata.logical_opt == "optimized"
    assert plan.plan_json is None
    assert plan.operator_graph.output_names == ("sum_qty",)
    assert plan.operator_graph.output_types == ("HUGEINT",)
    assert plan.operator_graph.select_aliases == {"sum_qty": "sum(l_quantity)"}
    assert calls == [(con, TPC_H_Q1_SQL)]


def test_substrait_frontend_returns_tqp_plan_with_plan_json(monkeypatch):
    calls = []

    def export_substrait(con, sql):
        calls.append((con, sql))
        return {"relations": []}

    monkeypatch.setattr("tpch_torch.frontend.substrait.export_substrait_json", export_substrait)

    con = duckdb.connect()
    plan = compile_substrait_plan(con, TPC_H_Q1_SQL)

    assert plan.query_id == 1
    assert plan.frontend == "substrait"
    assert plan.plan_json == {"relations": []}
    assert plan.duckdb_metadata is None
    assert calls == [(con, TPC_H_Q1_SQL)]


def test_sirius_frontend_accepts_non_tpch_sql_after_duckdb_admission(monkeypatch):
    calls = []

    def export_logical(con, sql):
        calls.append(sql)
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)
    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_physical_plan_json", lambda con, sql: [{"name": "SEQ_SCAN"}])
    monkeypatch.setattr("tpch_torch.frontend.sirius.describe_output_schema", lambda con, sql: (TQPOutputColumn("n", "BIGINT"),))
    monkeypatch.setattr("tpch_torch.frontend.sirius.select_expressions_by_alias", lambda con, sql: {"n": "count(*)"})
    monkeypatch.setattr(
        "tpch_torch.frontend.sirius.lower_duckdb_json_to_operator_graph",
        lambda sql, query_id, plan_json, *, output_schema, select_aliases, table_schemas: _dummy_graph(
            sql, query_id, output_schema=output_schema, select_aliases=select_aliases
        ),
    )

    con = duckdb.connect()
    plan = compile_sirius_plan(con, "select count(*) as n from lineitem")

    assert plan.query_id is None
    assert plan.frontend == "sirius"
    assert plan.source_sql == "select count(*) as n from lineitem"
    assert plan.operator_graph.output_names == ("n",)
    assert plan.operator_graph.select_aliases == {"n": "count(*)"}
    assert calls == ["select count(*) as n from lineitem"]


def test_sirius_frontend_admits_non_executable_generic_sql(monkeypatch):
    def export_logical(con, sql):
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)
    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_physical_plan_json", lambda con, sql: [{"name": "SEQ_SCAN"}])
    monkeypatch.setattr("tpch_torch.frontend.sirius.describe_output_schema", lambda con, sql: (TQPOutputColumn("id", "INTEGER"),))
    monkeypatch.setattr("tpch_torch.frontend.sirius.select_expressions_by_alias", lambda con, sql: {})
    monkeypatch.setattr(
        "tpch_torch.frontend.sirius.lower_duckdb_json_to_operator_graph",
        lambda sql, query_id, plan_json, *, output_schema, select_aliases, table_schemas: _dummy_graph(
            sql, query_id, output_schema=output_schema, select_aliases=select_aliases
        ),
    )

    con = duckdb.connect()
    plan = compile_sirius_plan(con, "select * from t join u on t.id = u.id")

    assert plan.query_id is None
    assert plan.generic_plan is None
    assert "joins are not supported" in plan.generic_error
