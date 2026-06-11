import duckdb
import pytest

from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.backend import PyTorchBackend


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
]


def _compiled_tpch_graph(query_id: int, sql: str) -> TQPOperatorGraph:
    node = TQPOperatorNode(
        node_id="compiled_tpch",
        kind=OperatorKind.COMPILED_TPCH,
        name="COMPILED_TPCH",
        metadata={"query_id": query_id},
    )
    return TQPOperatorGraph(source_sql=sql, query_id=query_id, root_id="compiled_tpch", nodes=(node,))



def test_pytorch_backend_executes_q1_tqp_plan():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = TQPPlan(query_id=1, source_sql=TPC_H_Q1_SQL, frontend="sirius", operator_graph=_compiled_tpch_graph(1, TPC_H_Q1_SQL))

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert [row["l_returnflag"] for row in rows] == ["A", "N"]


def test_pytorch_backend_rejects_tpch_without_operator_graph():
    plan = TQPPlan(query_id=99, source_sql="select 1", frontend="sirius")

    with pytest.raises(UnsupportedPlanError, match="requires a frontend-lowered TQP operator graph"):
        PyTorchBackend().execute(duckdb.connect(), plan, device="cpu")


def test_pytorch_backend_executes_generic_tqp_plan():
    from tpch_torch.generic_sql import parse_generic_sql

    con = duckdb.connect()
    con.execute("create table t(a integer)")
    con.execute("insert into t values (1), (2)")
    sql = "select count(*) as n from t"
    graph = TQPOperatorGraph(
        source_sql=sql,
        query_id=None,
        root_id="n0",
        nodes=(TQPOperatorNode(node_id="n0", kind=OperatorKind.SCAN, name="SEQ_SCAN"),),
    )
    plan = TQPPlan(
        query_id=None,
        source_sql=sql,
        frontend="sirius",
        generic_plan=parse_generic_sql(sql),
        operator_graph=graph,
    )

    assert PyTorchBackend().execute(con, plan, device="cpu") == [{"n": 2}]


def test_pytorch_backend_passes_compressed_mask_option_to_q6(monkeypatch):
    calls = []

    def execute_q6(con, *, device, use_compressed_masks=False):
        calls.append((device, use_compressed_masks))
        return [{"revenue": 1.0}]

    monkeypatch.setattr("tpch_torch.queries.q06.execute_q6", execute_q6)
    plan = TQPPlan(
        query_id=6,
        source_sql="select -- q6",
        frontend="sirius",
        operator_graph=_compiled_tpch_graph(6, "select -- q6"),
    )

    rows = PyTorchBackend().execute(duckdb.connect(), plan, device="cpu", use_compressed_masks=True)

    assert rows == [{"revenue": 1.0}]
    assert calls == [("cpu", True)]


def test_pytorch_backend_executes_tpch_through_operator_graph(monkeypatch):
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode

    calls = []

    class DummyGraphExecutor:
        def execute(self, con, plan, *, device="cpu", use_compressed_masks=False):
            calls.append((plan.query_id, device, use_compressed_masks))
            return [{"ok": True}]

    node = TQPOperatorNode(
        node_id="n0",
        kind=OperatorKind.COMPILED_TPCH,
        name="COMPILED_TPCH",
        metadata={"query_id": 3},
    )
    graph = TQPOperatorGraph(
        source_sql="select -- q3",
        query_id=3,
        root_id="n0",
        nodes=(node,),
    )
    plan = TQPPlan(query_id=3, source_sql="select -- q3", frontend="sirius", operator_graph=graph)
    monkeypatch.setattr("tpch_torch.backend.pytorch.PyTorchGraphExecutor", DummyGraphExecutor)

    rows = PyTorchBackend().execute(duckdb.connect(), plan, device="cpu", use_compressed_masks=True)

    assert rows == [{"ok": True}]
    assert calls == [(3, "cpu", True)]
