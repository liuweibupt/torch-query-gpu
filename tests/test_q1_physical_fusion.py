import duckdb
import pytest
import torch

from tpch_torch.backend.physical_fusion import _execute_q1_fused
from tpch_torch.duckdb_bridge import clear_lineitem_tensor_table_cache, create_lineitem_fixture


Q1_FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
]


def test_q1_fused_aggregates_with_masked_bincount_without_payload_gather(monkeypatch):
    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    clear_lineitem_tensor_table_cache(con)

    def fail_index_select(*_args, **_kwargs):
        raise AssertionError("Q1 fused aggregation should avoid selected-row payload gathers")

    monkeypatch.setattr(torch.Tensor, "index_select", fail_index_select)

    rows = _execute_q1_fused(con, "cpu")

    assert rows == [
        {
            "l_returnflag": "A",
            "l_linestatus": "F",
            "sum_qty": pytest.approx(5.0),
            "sum_base_price": pytest.approx(50.0),
            "sum_disc_price": pytest.approx(50.0),
            "sum_charge": pytest.approx(54.0),
            "avg_qty": pytest.approx(5.0),
            "avg_price": pytest.approx(50.0),
            "avg_disc": pytest.approx(0.0),
            "count_order": 1,
        },
        {
            "l_returnflag": "N",
            "l_linestatus": "O",
            "sum_qty": pytest.approx(10.0),
            "sum_base_price": pytest.approx(100.0),
            "sum_disc_price": pytest.approx(95.0),
            "sum_charge": pytest.approx(104.5),
            "avg_qty": pytest.approx(10.0),
            "avg_price": pytest.approx(100.0),
            "avg_disc": pytest.approx(0.05),
            "count_order": 1,
        },
    ]
