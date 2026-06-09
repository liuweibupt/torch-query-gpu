import json

import duckdb
import pytest

from tpch_torch.duckdb_bridge import (
    DuckDBSubstraitError,
    connect_database,
    create_lineitem_fixture,
    export_substrait_json,
    fetch_lineitem_tensor_table,
    run_duckdb_q1,
)
from tpch_torch.substrait import Q1Plan
from tpch_torch.validate import validate_q1




def q1_fixture_plan() -> Q1Plan:
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


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("N", "O", 20.0, 200.0, 0.10, 0.20, "1998-09-03"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
    ("N", "O", 30.0, 300.0, 0.05, 0.00, "1997-12-31"),
]


def test_fetch_lineitem_tensor_table_reads_duckdb_fixture():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    table = fetch_lineitem_tensor_table(con, device="cpu")

    assert len(table) == 4
    assert table.columns["l_shipdate"].tolist() == [19980902, 19980903, 19980101, 19971231]
    assert table.decode_value("l_returnflag", 0) == "A"


def test_run_duckdb_q1_returns_baseline_rows():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    rows = run_duckdb_q1(con)

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


def test_validate_q1_matches_duckdb_baseline():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    result = validate_q1(con, device="cpu", plan=q1_fixture_plan())

    assert result.row_count == 2
    assert result.max_abs_error < 1e-9


def test_connect_database_opens_path(tmp_path):
    db_path = tmp_path / "fixture.duckdb"
    con = connect_database(db_path)
    con.execute("create table marker(value integer)")
    con.close()

    reopened = connect_database(db_path)
    assert reopened.execute("select count(*) from marker").fetchone()[0] == 0


def test_export_substrait_json_uses_duckdb_extension_or_reports_unavailable():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    try:
        plan_json = export_substrait_json(con)
    except DuckDBSubstraitError as exc:
        pytest.skip(f"DuckDB Substrait extension unavailable: {exc}")

    assert isinstance(plan_json, dict)
    assert "relations" in plan_json
    json.dumps(plan_json)
