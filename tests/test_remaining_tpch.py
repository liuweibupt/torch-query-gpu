import duckdb
import pytest

from tpch_torch.duckdb_bridge import generate_tpch
from tpch_torch.runner import validate_sql_with_frontend
from tpch_torch.sql import get_tpch_query


@pytest.fixture(scope="module")
def tpch_con_remaining():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("query_id", (2, 4, 16, 17, 20, 21, 22))
def test_remaining_tpch_query_validates_with_sirius_frontend(tpch_con_remaining, query_id):
    sql = get_tpch_query(tpch_con_remaining, query_id)

    result = validate_sql_with_frontend(
        tpch_con_remaining,
        sql,
        device="cpu",
        frontend="sirius",
    )

    assert result.query_id == query_id
    assert result.max_abs_error <= 1e-2
