import pytest

import tpch_torch.queries.q01 as q01_module
from tpch_torch.queries.q01 import execute_q1
from tpch_torch.storage import TensorTable, table_from_rows
from tpch_torch.substrait import Q1Plan


@pytest.fixture
def q1_plan() -> Q1Plan:
    return Q1Plan(
        table_name="lineitem",
        shipdate_cutoff_yyyymmdd=19980902,
        required_columns=(
            "l_returnflag",
            "l_linestatus",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_shipdate",
        ),
        group_keys=("l_returnflag", "l_linestatus"),
        order_keys=("l_returnflag", "l_linestatus"),
    )


def test_table_from_rows_encodes_dates_and_strings():
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-02",
            }
        ],
        device="cpu",
    )

    assert isinstance(table, TensorTable)
    assert table.columns["l_shipdate"].item() == 19980902
    assert table.decode_value("l_returnflag", 0) == "N"
    assert table.decode_value("l_linestatus", 0) == "O"


def test_execute_q1_matches_expected_aggregates(q1_plan):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-02",
            },
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 20.0,
                "l_extendedprice": 200.0,
                "l_discount": 0.10,
                "l_tax": 0.20,
                "l_shipdate": "1998-09-03",
            },
            {
                "l_returnflag": "A",
                "l_linestatus": "F",
                "l_quantity": 5.0,
                "l_extendedprice": 50.0,
                "l_discount": 0.0,
                "l_tax": 0.08,
                "l_shipdate": "1998-01-01",
            },
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 30.0,
                "l_extendedprice": 300.0,
                "l_discount": 0.05,
                "l_tax": 0.00,
                "l_shipdate": "1997-12-31",
            },
        ],
        device="cpu",
    )

    rows = execute_q1(table, q1_plan)

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
            "sum_qty": pytest.approx(40.0),
            "sum_base_price": pytest.approx(400.0),
            "sum_disc_price": pytest.approx(380.0),
            "sum_charge": pytest.approx(389.5),
            "avg_qty": pytest.approx(20.0),
            "avg_price": pytest.approx(200.0),
            "avg_disc": pytest.approx(0.05),
            "count_order": 2,
        },
    ]


def test_execute_q1_does_not_use_general_unique_grouping(q1_plan, monkeypatch):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-02",
            },
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 20.0,
                "l_extendedprice": 200.0,
                "l_discount": 0.10,
                "l_tax": 0.20,
                "l_shipdate": "1998-09-02",
            },
            {
                "l_returnflag": "A",
                "l_linestatus": "F",
                "l_quantity": 5.0,
                "l_extendedprice": 50.0,
                "l_discount": 0.0,
                "l_tax": 0.08,
                "l_shipdate": "1998-01-01",
            },
        ],
        device="cpu",
    )

    def fail_unique_grouping(*args, **kwargs):
        raise AssertionError("Q1 should avoid general unique grouping for dictionary keys")

    monkeypatch.setattr(q01_module, "composite_group_ids", fail_unique_grouping, raising=False)

    rows = execute_q1(table, q1_plan)

    assert [
        (row["l_returnflag"], row["l_linestatus"], row["count_order"]) for row in rows
    ] == [
        ("A", "F", 1),
        ("N", "O", 2),
    ]


def test_execute_q1_returns_empty_rows_without_filtering_shipdate(q1_plan):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-03",
            }
        ],
        device="cpu",
    )

    rows = execute_q1(table, q1_plan)

    assert rows == []


def test_q1_filter_does_not_materialize_shipdate_payload(q1_plan):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-02",
            }
        ],
        device="cpu",
    )

    filtered = q01_module._filter_q1(table, q1_plan)

    assert "l_shipdate" not in filtered


def test_q1_filter_computes_selected_row_index_once(q1_plan, monkeypatch):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_tax": 0.10,
                "l_shipdate": "1998-09-02",
            },
            {
                "l_returnflag": "R",
                "l_linestatus": "F",
                "l_quantity": 5.0,
                "l_extendedprice": 50.0,
                "l_discount": 0.0,
                "l_tax": 0.08,
                "l_shipdate": "1998-09-03",
            },
        ],
        device="cpu",
    )
    calls = []
    original_nonzero = q01_module.torch.nonzero

    def recording_nonzero(*args, **kwargs):
        calls.append(args[0])
        return original_nonzero(*args, **kwargs)

    monkeypatch.setattr(q01_module.torch, "nonzero", recording_nonzero)

    filtered = q01_module._filter_q1(table, q1_plan)

    assert len(calls) == 1
    assert filtered["l_quantity"].tolist() == [10.0]


def test_execute_q1_rejects_missing_required_column(q1_plan):
    table = table_from_rows(
        [
            {
                "l_returnflag": "N",
                "l_linestatus": "O",
                "l_quantity": 10.0,
                "l_extendedprice": 100.0,
                "l_discount": 0.05,
                "l_shipdate": "1998-09-02",
            }
        ],
        device="cpu",
    )

    with pytest.raises(KeyError, match="l_tax"):
        execute_q1(table, q1_plan)
