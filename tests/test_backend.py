import duckdb
import pytest

from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.ir import TQPPlan
from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.backend import PyTorchBackend


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
]


def test_pytorch_backend_executes_q1_tqp_plan():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = TQPPlan(query_id=1, source_sql=TPC_H_Q1_SQL, frontend="sirius")

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert [row["l_returnflag"] for row in rows] == ["A", "N"]


def test_pytorch_backend_rejects_unknown_query_id():
    plan = TQPPlan(query_id=99, source_sql="select 1", frontend="sirius")

    with pytest.raises(UnsupportedPlanError, match="TPC-H Q99"):
        PyTorchBackend().execute(duckdb.connect(), plan, device="cpu")


def test_pytorch_backend_executes_generic_tqp_plan():
    from tpch_torch.generic_sql import parse_generic_sql

    con = duckdb.connect()
    con.execute("create table t(a integer)")
    con.execute("insert into t values (1), (2)")
    plan = TQPPlan(
        query_id=None,
        source_sql="select count(*) as n from t",
        frontend="sirius",
        generic_plan=parse_generic_sql("select count(*) as n from t"),
    )

    assert PyTorchBackend().execute(con, plan, device="cpu") == [{"n": 2}]


def test_pytorch_backend_passes_compressed_mask_option_to_q6(monkeypatch):
    import tpch_torch.backend.pytorch as backend_module

    calls = []

    def execute_q6(con, *, device, use_compressed_masks=False):
        calls.append((device, use_compressed_masks))
        return [{"revenue": 1.0}]

    monkeypatch.setitem(backend_module._EXECUTOR_BY_QUERY, 6, "q06")
    monkeypatch.setattr("tpch_torch.queries.q06.execute_q6", execute_q6)
    plan = TQPPlan(query_id=6, source_sql="select -- q6", frontend="sirius")

    rows = PyTorchBackend().execute(duckdb.connect(), plan, device="cpu", use_compressed_masks=True)

    assert rows == [{"revenue": 1.0}]
    assert calls == [("cpu", True)]
