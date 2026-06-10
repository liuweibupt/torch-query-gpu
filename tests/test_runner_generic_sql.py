import duckdb

from tpch_torch.runner import run_sql, validate_sql


def _make_table():
    con = duckdb.connect()
    con.execute("create table t(a integer, b double)")
    con.execute("insert into t values (1, 1.5), (1, 2.5), (2, 3.0)")
    return con


def test_validate_sql_accepts_generic_count_query():
    result = validate_sql(_make_table(), "select count(*) as n from t", device="cpu")

    assert result.query_id is None
    assert result.row_count == 1
    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"n": 3}]


def test_validate_sql_accepts_generic_grouped_sum_query():
    sql = "select a, sum(b) as total from t group by a order by a"

    result = validate_sql(_make_table(), sql, device="cpu")

    assert result.query_id is None
    assert result.row_count == 2
    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"a": 1, "total": 4.0}, {"a": 2, "total": 3.0}]


def test_run_sql_accepts_generic_projection_filter_query():
    sql = "select a, b * 2 as twice from t where b >= 2 order by a"

    result = run_sql(_make_table(), sql, device="cpu")

    assert result.query_id is None
    assert result.rows == [{"a": 1, "twice": 5.0}, {"a": 2, "twice": 6.0}]


def test_run_sql_reports_backend_unsupported_generic_join():
    con = duckdb.connect()
    con.execute("create table t(id integer)")
    con.execute("create table u(id integer)")

    import pytest
    from tpch_torch.substrait import UnsupportedPlanError

    with pytest.raises(UnsupportedPlanError, match="generic SQL is not executable by PyTorch backend"):
        run_sql(con, "select * from t join u on t.id = u.id", device="cpu")
