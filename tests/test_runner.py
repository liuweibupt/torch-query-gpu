from dataclasses import dataclass

import duckdb

from tpch_torch.duckdb_bridge import create_lineitem_fixture, generate_tpch
from tpch_torch.query_catalog import identify_tpch_query
from tpch_torch.runner import (
    load_sql,
    run_sql,
    run_sql_with_frontend,
    validate_sql,
    validate_sql_with_frontend,
)
from tpch_torch.sql import get_tpch_query
from tpch_torch.sql import TPC_H_Q1_SQL


@dataclass(frozen=True)
class DummyLogicalPlan:
    logical_plan: str = "logical"
    logical_opt: str = "optimized"
    physical_plan: str = "physical"


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("N", "O", 20.0, 200.0, 0.10, 0.20, "1998-09-03"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
    ("N", "O", 30.0, 300.0, 0.05, 0.00, "1997-12-31"),
]


def test_run_sql_executes_q1_from_sql_text_through_default_frontend():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    result = run_sql(con, TPC_H_Q1_SQL, device="cpu")

    assert result.query_id == 1
    assert result.rows[0]["l_returnflag"] == "A"
    assert result.rows[1]["count_order"] == 2


def test_validate_sql_compares_q1_with_duckdb_baseline():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    result = validate_sql(con, TPC_H_Q1_SQL, device="cpu")

    assert result.query_id == 1
    assert result.row_count == 2
    assert result.max_abs_error < 1e-9


def test_load_sql_requires_exactly_one_source(tmp_path):
    con = duckdb.connect()

    try:
        load_sql(con)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    sql_file = tmp_path / "q.sql"
    sql_file.write_text("select 1")
    assert load_sql(con, sql_file=sql_file) == "select 1"


def test_validate_sql_compares_q6_with_duckdb_baseline():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    sql = """
        select sum(l_extendedprice * l_discount) as revenue
        from lineitem
        where l_shipdate >= date '1994-01-01'
          and l_shipdate < date '1995-01-01'
          and l_discount between 0.05 and 0.07
          and l_quantity < 24
    """

    result = validate_sql(con, sql, device="cpu")

    assert result.query_id == 6
    assert result.row_count == 1
    assert result.max_abs_error < 1e-6


def test_run_sql_with_sirius_frontend_skips_substrait_export(monkeypatch):
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    calls = []

    def export_plan(connection, sql):
        calls.append(("substrait", sql))
        raise AssertionError("substrait export should not be called")

    def export_logical(connection, sql):
        calls.append(("logical", sql))
        return DummyLogicalPlan()

    monkeypatch.setattr("tpch_torch.frontend.substrait.export_substrait_json", export_plan)
    monkeypatch.setattr("tpch_torch.frontend.sirius.export_duckdb_logical_plan", export_logical)

    result = run_sql_with_frontend(con, TPC_H_Q1_SQL, device="cpu", frontend="sirius")

    assert result.query_id == 1
    assert calls == [("logical", TPC_H_Q1_SQL)]


def test_validate_sql_with_sirius_frontend_compares_with_duckdb_baseline():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    result = validate_sql_with_frontend(con, TPC_H_Q1_SQL, device="cpu", frontend="sirius")

    assert result.query_id == 1
    assert result.max_abs_error < 1e-9


def test_identifies_tpch_queries_blocked_by_strict_substrait_export():
    con = duckdb.connect()

    for query_id in (2, 4, 16, 17, 20, 21, 22):
        assert identify_tpch_query(get_tpch_query(con, query_id)) == query_id
