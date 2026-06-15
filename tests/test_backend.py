import duckdb
import pytest
import torch

from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.ir import TQPPlan
from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.sql import TPC_H_Q1_SQL
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.backend import PyTorchBackend
from tpch_torch.compressed import PlainMask


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
]

Q6_FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1994-01-01"),
    ("N", "O", 23.0, 200.0, 0.07, 0.20, "1994-12-31"),
    ("A", "F", 24.0, 300.0, 0.06, 0.08, "1994-06-01"),
]

TPC_H_Q6_SQL = """
select
    sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1994-01-01'
  and l_shipdate < date '1995-01-01'
  and l_discount between 0.05 and 0.07
  and l_quantity < 24
""".strip()


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
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert [row["l_returnflag"] for row in rows] == ["A", "N"]


def test_pytorch_backend_rejects_tpch_without_operator_graph():
    plan = TQPPlan(query_id=99, source_sql="select 1", frontend="sirius")

    with pytest.raises(UnsupportedPlanError, match="requires a frontend-lowered TQP operator graph"):
        PyTorchBackend().execute(duckdb.connect(), plan, device="cpu")


def test_pytorch_backend_executes_legacy_generic_tqp_plan_without_operator_graph():
    from tpch_torch.generic_sql import parse_generic_sql

    con = duckdb.connect()
    con.execute("create table t(a integer)")
    con.execute("insert into t values (1), (2)")
    sql = "select count(*) as n from t"
    plan = TQPPlan(
        query_id=None,
        source_sql=sql,
        frontend="sirius",
        generic_plan=parse_generic_sql(sql),
    )

    assert PyTorchBackend().execute(con, plan, device="cpu") == [{"n": 2}]


def test_pytorch_backend_prefers_physical_graph_over_legacy_generic_parser(monkeypatch):
    import tpch_torch.backend.graph as graph_backend

    con = duckdb.connect()
    con.execute("create table t(a integer)")
    con.execute("insert into t values (1), (2)")
    sql = "select count(*) as n from t"
    plan = compile_sirius_plan(con, sql)
    monkeypatch.setattr(
        graph_backend,
        "execute_generic_sql_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy generic parser path used")),
    )

    assert PyTorchBackend().execute(con, plan, device="cpu") == [{"n": 2}]


def test_pytorch_backend_passes_compressed_mask_option_to_q6(monkeypatch):
    con = duckdb.connect()
    con.execute(
        "create table lineitem("
        "l_quantity double, l_extendedprice double, l_discount double, l_shipdate date"
        ")"
    )
    con.execute("insert into lineitem values (10.0, 100.0, 0.06, date '1994-02-01')")
    graph = TQPOperatorGraph(
        source_sql="select -- q6",
        query_id=6,
        root_id="n0",
        nodes=(TQPOperatorNode(node_id="n0", kind=OperatorKind.SCAN, name="SEQ_SCAN"),),
    )
    plan = TQPPlan(query_id=6, source_sql="select -- q6", frontend="sirius", operator_graph=graph)
    calls = []

    def compressed_mask(table):
        calls.append(tuple(sorted(table.columns)))
        return PlainMask(torch.tensor([True], dtype=torch.bool))

    monkeypatch.setattr("tpch_torch.backend.graph._q6_compressed_mask", compressed_mask)

    rows = PyTorchBackend().execute(con, plan, device="cpu", use_compressed_masks=True)

    assert rows == [{"revenue": 6.0}]
    assert calls == [("l_discount", "l_extendedprice", "l_quantity", "l_shipdate")]


def test_q6_default_graph_execution_uses_physical_plan_interpreter(monkeypatch):
    import tpch_torch.backend.graph as graph_backend
    from tpch_torch.frontend import compile_sirius_plan

    con = duckdb.connect()
    create_lineitem_fixture(con, Q6_FIXTURE_ROWS)
    plan = compile_sirius_plan(con, TPC_H_Q6_SQL)
    calls = []
    execute_physical_plan = graph_backend.execute_physical_plan

    def tracked_execute_physical_plan(con_arg, graph, *, device="cpu"):
        calls.append((graph.query_id, graph.root.kind, device))
        return execute_physical_plan(con_arg, graph, device=device)

    monkeypatch.setattr(graph_backend, "execute_physical_plan", tracked_execute_physical_plan)
    monkeypatch.setattr(
        graph_backend,
        "_execute_q6_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("q6 direct primitive path used")),
    )

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert calls == [(6, OperatorKind.AGGREGATE, "cpu")]
    assert rows == [{"revenue": 19.0}]


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


def test_q1_graph_execution_does_not_call_template(monkeypatch):
    from tpch_torch.frontend import compile_sirius_plan

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)
    monkeypatch.setattr(
        "tpch_torch.queries.q01.execute_q1",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("q01 template called")),
    )

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert [row["l_returnflag"] for row in rows] == ["A", "N"]


def test_q1_graph_execution_uses_physical_plan_interpreter(monkeypatch):
    import tpch_torch.backend.graph as graph_backend
    from tpch_torch.frontend import compile_sirius_plan

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)
    calls = []
    execute_physical_plan = graph_backend.execute_physical_plan

    def tracked_execute_physical_plan(con_arg, graph, *, device="cpu"):
        calls.append((graph.query_id, graph.root.kind, device))
        return execute_physical_plan(con_arg, graph, device=device)

    monkeypatch.setattr(graph_backend, "execute_physical_plan", tracked_execute_physical_plan)
    monkeypatch.setattr(
        graph_backend,
        "_execute_q1_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("q1 direct primitive path used")),
        raising=False,
    )

    rows = PyTorchBackend().execute(con, plan, device="cpu")

    assert calls == [(1, OperatorKind.PROJECT, "cpu")]
    assert [row["l_returnflag"] for row in rows] == ["A", "N"]
