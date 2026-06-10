import duckdb
import pytest

from tpch_torch.duckdb_bridge import generate_tpch
from tpch_torch.runner import validate_sql, validate_sql_with_frontend, validate_sql_with_plan_source
from tpch_torch.sql import get_tpch_query

SUPPORTED_DUCKDB_EXPORTABLE_QUERIES = (1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19)
ALL_TPCH_QUERIES = tuple(range(1, 23))


@pytest.fixture(scope="module")
def tpch_con():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("query_id", SUPPORTED_DUCKDB_EXPORTABLE_QUERIES)
def test_duckdb_exportable_tpch_query_validates_through_pytorch(tpch_con, query_id):
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
