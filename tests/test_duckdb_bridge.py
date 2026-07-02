import inspect
import json

import duckdb
import pytest
import torch

from tpch_torch.duckdb_bridge import (
    DuckDBSubstraitError,
    connect_database,
    create_lineitem_fixture,
    export_substrait_json,
    fetch_lineitem_tensor_table,
    generate_tpch,
)
from tpch_torch.relational import run_duckdb_sql
from tpch_torch.sql import TPC_H_Q1_SQL

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



class _FakeLineitemConnection:
    def __init__(self, fail_after: int | None = 1):
        self.execute_count = 0
        self.fail_after = fail_after

    def execute(self, _sql):
        self.execute_count += 1
        if self.fail_after is not None and self.execute_count > self.fail_after:
            raise AssertionError("hot Q1 should reuse resident lineitem tensors")
        return self

    def fetchnumpy(self):
        import numpy as np

        return {
            "l_returnflag": np.array([1, 0], dtype=np.int64),
            "l_linestatus": np.array([1, 0], dtype=np.int64),
            "l_quantity": np.array([10.0, 5.0], dtype=np.float64),
            "l_extendedprice": np.array([100.0, 50.0], dtype=np.float64),
            "l_discount": np.array([0.05, 0.0], dtype=np.float64),
            "l_tax": np.array([0.10, 0.08], dtype=np.float64),
            "l_shipdate": np.array([19980902, 19980101], dtype=np.int32),
        }


def test_fetch_lineitem_tensor_table_reuses_cached_tensors_for_same_connection_and_device():
    con = _FakeLineitemConnection()

    first = fetch_lineitem_tensor_table(con, device="cpu")
    second = fetch_lineitem_tensor_table(con, device="cpu")

    assert second is first
    assert con.execute_count == 1


def test_fetch_lineitem_tensor_table_cache_normalizes_equivalent_cpu_devices():
    con = _FakeLineitemConnection()

    first = fetch_lineitem_tensor_table(con, device="cpu")
    second = fetch_lineitem_tensor_table(con, device=torch.device("cpu"))

    assert second is first
    assert con.execute_count == 1

def test_clear_lineitem_tensor_table_cache_forces_refetch():
    from tpch_torch.duckdb_bridge import clear_lineitem_tensor_table_cache

    con = _FakeLineitemConnection(fail_after=None)

    first = fetch_lineitem_tensor_table(con, device="cpu")
    clear_lineitem_tensor_table_cache(con)
    second = fetch_lineitem_tensor_table(con, device="cpu")

    assert second is not first
    assert con.execute_count == 2


def test_fetch_lineitem_tensor_table_uses_preencoded_string_columns(monkeypatch):
    from tpch_torch import duckdb_bridge

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    def fail_generic_encoder(*args, **kwargs):
        raise AssertionError("Q1 lineitem fetch should not build object string tensors")

    monkeypatch.setattr(duckdb_bridge, "table_from_columnar", fail_generic_encoder, raising=False)

    table = fetch_lineitem_tensor_table(con, device="cpu")

    assert table.dictionaries["l_returnflag"] == ("A", "N", "R")
    assert table.dictionaries["l_linestatus"] == ("F", "O")
    assert table.columns["l_returnflag"].tolist() == [1, 1, 0, 1]
    assert table.columns["l_linestatus"].tolist() == [1, 1, 0, 1]


def test_run_duckdb_sql_returns_q1_baseline_rows():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    rows = run_duckdb_sql(con, TPC_H_Q1_SQL)

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


def test_connect_database_opens_path(tmp_path):
    db_path = tmp_path / "fixture.duckdb"
    con = connect_database(db_path)
    con.execute("create table marker(value integer)")
    con.close()

    reopened = connect_database(db_path)
    assert reopened.execute("select count(*) from marker").fetchone()[0] == 0


def test_export_substrait_json_requires_explicit_sql_argument():
    parameter = inspect.signature(export_substrait_json).parameters["sql"]

    assert parameter.default is inspect.Signature.empty


def test_export_substrait_json_uses_duckdb_extension_or_reports_unavailable():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    try:
        plan_json = export_substrait_json(con, TPC_H_Q1_SQL)
    except DuckDBSubstraitError as exc:
        pytest.skip(f"DuckDB Substrait extension unavailable: {exc}")

    assert isinstance(plan_json, dict)
    assert "relations" in plan_json
    json.dumps(plan_json)


def test_export_substrait_json_reports_missing_explicit_extension(monkeypatch, tmp_path):
    missing_extension = tmp_path / "missing_substrait.duckdb_extension"
    monkeypatch.setenv("TQG_SUBSTRAIT_EXTENSION", str(missing_extension))
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)

    with pytest.raises(DuckDBSubstraitError) as exc_info:
        export_substrait_json(con, TPC_H_Q1_SQL)

    message = str(exc_info.value)
    assert "TQG_SUBSTRAIT_EXTENSION" in message
    assert str(missing_extension) in message


def test_load_substrait_extension_uses_default_install_when_env_unset(monkeypatch):
    from tpch_torch import duckdb_bridge

    class RecordingConnection:
        def __init__(self):
            self.calls = []

        def install_extension(self, name, repository=None):
            self.calls.append(("install_extension", name, repository))

        def load_extension(self, name):
            self.calls.append(("load_extension", name))

    monkeypatch.delenv("TQG_SUBSTRAIT_EXTENSION", raising=False)
    con = RecordingConnection()

    duckdb_bridge._load_substrait_extension(con)

    assert con.calls == [
        ("install_extension", "substrait", "community"),
        ("load_extension", "substrait"),
    ]


def test_load_substrait_extension_wraps_runtime_error(monkeypatch):
    from tpch_torch import duckdb_bridge

    class FailingLoadConnection:
        def install_extension(self, name, repository=None):
            return None

        def load_extension(self, name):
            raise RuntimeError("Resource temporarily unavailable")

    monkeypatch.delenv("TQG_SUBSTRAIT_EXTENSION", raising=False)

    with pytest.raises(DuckDBSubstraitError) as exc_info:
        duckdb_bridge._load_substrait_extension(FailingLoadConnection())

    assert "Resource temporarily unavailable" in str(exc_info.value)


def test_generate_tpch_sets_default_helper_threads(monkeypatch):
    class RecordingConnection:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))
            return self

    monkeypatch.delenv("TQG_DUCKDB_THREADS", raising=False)
    con = RecordingConnection()

    generate_tpch(con, scale_factor=0.01)

    assert con.calls[:4] == [
        ("install tpch", None),
        ("load tpch", None),
        ("pragma threads=1", None),
        ("call dbgen(sf = ?)", [0.01]),
    ]
