import duckdb
import pytest

from tpch_torch.duckdb_bridge import generate_tpch
from tpch_torch.runner import validate_sql, validate_sql_with_frontend
from tpch_torch.sql import get_tpch_query

STRICT_SUBSTRAIT_EXPORTABLE_QUERIES = (1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19)
ALL_TPCH_QUERIES = tuple(range(1, 23))


@pytest.fixture(scope="module")
def tpch_con():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("query_id", STRICT_SUBSTRAIT_EXPORTABLE_QUERIES)
def test_strict_substrait_exportable_tpch_query_validates_through_default_frontend(tpch_con, query_id):
    sql = get_tpch_query(tpch_con, query_id)

    result = validate_sql(tpch_con, sql, device="cpu")

    assert result.query_id == query_id
    assert result.max_abs_error <= 1e-2


@pytest.mark.parametrize("query_id", ALL_TPCH_QUERIES)
def test_all_tpch_queries_validate_through_sirius_frontend(tpch_con, query_id):
    sql = get_tpch_query(tpch_con, query_id)

    result = validate_sql_with_frontend(tpch_con, sql, device="cpu", frontend="sirius")

    assert result.query_id == query_id
    assert result.max_abs_error <= 1e-2


def test_all_tpch_queries_do_not_call_query_template_executors(tpch_con, monkeypatch):
    for query_id in range(2, 23):
        module = __import__(f"tpch_torch.queries.q{query_id:02d}", fromlist=[f"execute_q{query_id}"])
        monkeypatch.setattr(
            module,
            f"execute_q{query_id}",
            lambda *args, _query_id=query_id, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"q{_query_id:02d} template executor called")
            ),
        )

    for query_id in range(1, 23):
        sql = get_tpch_query(tpch_con, query_id)
        result = validate_sql_with_frontend(tpch_con, sql, device="cpu", frontend="sirius")
        assert result.query_id == query_id
        assert result.max_abs_error <= 1e-2
