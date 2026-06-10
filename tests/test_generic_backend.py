import duckdb
import pytest

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.generic_sql import parse_generic_sql
from tpch_torch.substrait import UnsupportedPlanError


def _make_table():
    con = duckdb.connect()
    con.execute("create table t(a integer, b double, c varchar)")
    con.execute("insert into t values (1, 1.5, 'x'), (1, 2.5, 'y'), (2, 3.0, 'z')")
    return con


def test_generic_backend_executes_count_star():
    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql("select count(*) as n from t"), device="cpu")

    assert rows == [{"n": 3}]


def test_generic_backend_executes_grouped_sum_ordered():
    sql = "select a, sum(b) as total from t group by a order by a"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [{"a": 1, "total": 4.0}, {"a": 2, "total": 3.0}]


def test_generic_backend_executes_filter_projection_and_order():
    sql = "select a, b * 2 as twice from t where b >= 2 order by a"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [{"a": 1, "twice": 5.0}, {"a": 2, "twice": 6.0}]


def test_generic_backend_rejects_missing_source_column():
    con = _make_table()
    plan = parse_generic_sql("select missing from t")

    with pytest.raises(UnsupportedPlanError, match="missing column"):
        execute_generic_sql_plan(con, plan, device="cpu")
