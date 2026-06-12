import duckdb
import pytest

from tpch_torch.duckdb_bridge import generate_tpch
from tpch_torch.runner import run_sql, validate_sql, validate_sql_with_frontend
from tpch_torch.sql import get_tpch_query


def _join_con():
    con = duckdb.connect()
    con.execute("create table t(a integer, id integer, amount double)")
    con.execute("create table u(id integer, name varchar)")
    con.execute("insert into t values (1, 10, 1.5), (2, 10, 2.5), (3, 20, 3.0)")
    con.execute("insert into u values (10, 'x'), (20, 'y')")
    return con


def test_physical_plan_executes_generic_inner_join_without_generic_parser(monkeypatch):
    import tpch_torch.backend.graph as graph_backend

    monkeypatch.setattr(
        graph_backend,
        "execute_generic_sql_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old generic parser path used")),
    )

    result = run_sql(_join_con(), "select a, name from t join u on t.id = u.id order by a", device="cpu")

    assert result.query_id is None
    assert result.rows == [{"a": 1, "name": "x"}, {"a": 2, "name": "x"}, {"a": 3, "name": "y"}]


def test_physical_plan_executes_join_group_order_limit_query():
    sql = """
        select name, sum(amount) as total
        from t join u on t.id = u.id
        where amount > 1.5
        group by name
        order by total desc
        limit 1
    """

    result = validate_sql(_join_con(), sql, device="cpu")

    assert result.query_id is None
    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"name": "y", "total": 3.0}]


def test_physical_plan_executes_join_projection_alias_expression(monkeypatch):
    import tpch_torch.backend.graph as graph_backend

    monkeypatch.setattr(
        graph_backend,
        "execute_generic_sql_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old generic parser path used")),
    )
    sql = """
        select name, amount * 2 as twice
        from t join u on t.id = u.id
        where amount > 1.5
        order by twice desc
    """

    result = run_sql(_join_con(), sql, device="cpu")

    assert result.rows == [{"name": "y", "twice": 6.0}, {"name": "x", "twice": 5.0}]


def test_physical_plan_executes_final_aggregate_expression():
    con = duckdb.connect()
    con.execute("create table r(x double, y double)")
    con.execute("insert into r values (2.0, 4.0), (4.0, 6.0)")

    result = validate_sql(con, "select 100.0 * sum(x) / sum(y) as ratio from r", device="cpu")

    assert result.query_id is None
    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"ratio": 60.0}]


@pytest.fixture(scope="module")
def tpch_con_physical():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("query_id,module_name,func_name", [
    (12, "tpch_torch.backend.tpch_graph_q12", "execute_q12_graph"),
    (14, "tpch_torch.backend.tpch_graph_q14", "execute_q14_graph"),
    (19, "tpch_torch.backend.tpch_graph_q19", "execute_q19_graph"),
])
def test_migrated_tpch_query_uses_physical_plan_not_recipe(
    tpch_con_physical,
    query_id,
    module_name,
    func_name,
    monkeypatch,
):
    module = __import__(module_name, fromlist=[func_name])
    monkeypatch.setattr(
        module,
        func_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query recipe path used")),
    )

    result = validate_sql_with_frontend(
        tpch_con_physical,
        get_tpch_query(tpch_con_physical, query_id),
        device="cpu",
        frontend="sirius",
    )

    assert result.query_id == query_id
    assert result.max_abs_error <= 1e-2
