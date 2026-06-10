from dataclasses import dataclass

import duckdb

from tpch_torch.duckdb_bridge import DuckDBSubstraitError
from tpch_torch.frontend import compile_auto_plan, compile_sirius_plan, compile_substrait_plan
from tpch_torch.sql import TPC_H_Q1_SQL


@dataclass(frozen=True)
class DummyLogicalPlan:
    logical_plan: str = "logical"
    logical_opt: str = "optimized"
    physical_plan: str = "physical"


def test_sirius_frontend_returns_tqp_plan_with_duckdb_metadata(monkeypatch):
    calls = []

    def export_logical(con, sql):
        calls.append((con, sql))
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)

    con = duckdb.connect()
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)

    assert plan.query_id == 1
    assert plan.source_sql == TPC_H_Q1_SQL
    assert plan.frontend == "sirius"
    assert plan.duckdb_metadata is not None
    assert plan.duckdb_metadata.logical_opt == "optimized"
    assert plan.plan_json is None
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


def test_auto_frontend_falls_back_to_sirius_after_substrait_export_failure(monkeypatch):
    calls = []

    def export_substrait(con, sql):
        calls.append("substrait")
        raise DuckDBSubstraitError("DELIM_JOIN")

    def export_logical(con, sql):
        calls.append("sirius")
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.substrait.export_substrait_json", export_substrait)
    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)

    plan = compile_auto_plan(duckdb.connect(), TPC_H_Q1_SQL)

    assert plan.frontend == "sirius"
    assert calls == ["substrait", "sirius"]
