import duckdb

from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.runner import load_sql, run_sql, validate_sql
from tpch_torch.sql import TPC_H_Q1_SQL


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("N", "O", 20.0, 200.0, 0.10, 0.20, "1998-09-03"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
    ("N", "O", 30.0, 300.0, 0.05, 0.00, "1997-12-31"),
]


def test_run_sql_executes_q1_from_sql_text_through_substrait():
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
