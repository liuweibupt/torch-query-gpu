import duckdb
import pytest

from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.queries.q06 import execute_q6


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1994-01-01"),
    ("N", "O", 23.0, 200.0, 0.07, 0.20, "1994-12-31"),
    ("A", "F", 24.0, 300.0, 0.06, 0.08, "1994-06-01"),
    ("R", "F", 1.0, 400.0, 0.04, 0.00, "1994-06-01"),
    ("R", "F", 1.0, 500.0, 0.06, 0.00, "1995-01-01"),
]


def test_execute_q6_sums_discounted_revenue_for_canonical_predicate():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    rows = execute_q6(con, device="cpu")

    assert rows == [{"revenue": pytest.approx(19.0)}]


def test_execute_q6_compressed_mask_path_matches_plain_path():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    assert execute_q6(con, device="cpu", use_compressed_masks=True) == execute_q6(con, device="cpu")
