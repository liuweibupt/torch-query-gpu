import duckdb
import pytest
import torch

from tpch_torch.backend.generic import execute_generic_sql_plan
from tpch_torch.generic_sql import parse_generic_sql
from tpch_torch.errors import UnsupportedPlanError


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


def test_generic_backend_executes_string_filter_and_projection():
    rows = execute_generic_sql_plan(
        _make_table(),
        parse_generic_sql("select c from t where c = 'y'"),
        device="cpu",
    )

    assert rows == [{"c": "y"}]


def test_generic_backend_executes_extended_scalar_aggregates():
    sql = "select min(b) as lo, max(b) as hi, avg(b) as mean_b, count(c) as c_count from t"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [{"lo": 1.5, "hi": 3.0, "mean_b": 7.0 / 3.0, "c_count": 3}]


def test_generic_backend_executes_extended_grouped_aggregates():
    sql = "select a, min(b) as lo, max(b) as hi, avg(b) as mean_b, count(c) as c_count from t group by a order by a"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [
        {"a": 1, "lo": 1.5, "hi": 2.5, "mean_b": 2.0, "c_count": 2},
        {"a": 2, "lo": 3.0, "hi": 3.0, "mean_b": 3.0, "c_count": 1},
    ]


def test_generic_backend_executes_in_like_or_not_filters():
    sql = "select c from t where not a = 1 or c in ('x', 'z') and c like 'z%' order by c"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [{"c": "z"}]


def test_generic_backend_executes_descending_order_by_with_limit():
    sql = "select a, b from t order by b desc limit 2"

    rows = execute_generic_sql_plan(_make_table(), parse_generic_sql(sql), device="cpu")

    assert rows == [{"a": 2, "b": 3.0}, {"a": 1, "b": 2.5}]


def test_fetch_generic_tensor_table_uses_columnar_numpy_fetch():
    import numpy as np
    from tpch_torch.backend.generic import _fetch_generic_tensor_table

    class FakeResult:
        def fetchnumpy(self):
            return {
                "a": np.array([1, 2, 3], dtype=np.int64),
                "c": np.array(["x", "y", "x"]),
            }

        def fetchall(self):
            raise AssertionError("generic table fetch must not materialize rows with fetchall")

    class FakeConnection:
        def execute(self, sql):
            assert sql == "select a, c from t"
            return FakeResult()

    table = _fetch_generic_tensor_table(FakeConnection(), "t", ("a", "c"), "cpu")

    assert table.columns["a"].tolist() == [1, 2, 3]
    assert table.columns["a"].dtype == torch.int64
    assert table.columns["c"].tolist() == [0, 1, 0]
    assert table.dictionaries["c"] == ("x", "y")
