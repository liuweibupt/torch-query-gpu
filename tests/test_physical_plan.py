import duckdb
import pytest
import torch

from tpch_torch.duckdb_bridge import create_lineitem_fixture, generate_tpch
from tpch_torch.frontend import compile_sirius_plan
from tpch_torch.runner import run_sql, validate_sql, validate_sql_with_frontend
from tpch_torch.sql import TPC_H_Q1_SQL, get_tpch_query


Q1_FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
]


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


def test_physical_inner_join_indices_stays_in_tensor_path(monkeypatch):
    from tpch_torch.backend.physical import _inner_join_indices

    expected_left = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
    expected_right = torch.tensor([0, 1, 3, 0, 1], dtype=torch.int64)

    def fail_tolist(_tensor):
        raise AssertionError("join indices must not materialize keys through Tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    left_rows, right_rows = _inner_join_indices(
        torch.tensor([2, 1, 2, 3], dtype=torch.int64),
        torch.tensor([2, 2, 4, 1], dtype=torch.int64),
    )

    assert torch.equal(left_rows, expected_left)
    assert torch.equal(right_rows, expected_right)


def test_physical_inner_join_indices_uses_sorted_unique_build_fast_path(monkeypatch):
    from tpch_torch.backend.physical import _inner_join_indices

    def fail_argsort(*_args, **_kwargs):
        raise AssertionError("sorted unique build-side keys should not call torch.argsort")

    def fail_repeat_interleave(*_args, **_kwargs):
        raise AssertionError("sorted unique build-side keys should not need repeat_interleave")

    monkeypatch.setattr(torch, "argsort", fail_argsort)
    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    left_rows, right_rows = _inner_join_indices(
        torch.tensor([3, 1, 2, 5, 2], dtype=torch.int64),
        torch.tensor([1, 2, 3, 4], dtype=torch.int64),
    )

    assert torch.equal(left_rows, torch.tensor([0, 1, 2, 4], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([2, 0, 1, 1], dtype=torch.int64))


def test_physical_semi_and_anti_join_use_membership_probe_without_pair_expansion(monkeypatch):
    import tpch_torch.backend.physical_join as physical_join
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_pair_expansion(*_args, **_kwargs):
        raise AssertionError("semi/anti joins should use membership probing, not pair expansion")

    monkeypatch.setattr(physical_join, "join_indices_for_conditions", fail_pair_expansion)

    left = PhysicalTable(
        "left",
        {"id": PhysicalValue(torch.tensor([1, 2, 3, 4], dtype=torch.int64))},
        ("id",),
        4,
    )
    right = PhysicalTable(
        "right",
        {"id": PhysicalValue(torch.tensor([2, 2, 4, 5], dtype=torch.int64))},
        ("id",),
        4,
    )

    semi_rows = physical_join.semi_join_indices(left, right, (("id", "id"),))
    anti_rows = physical_join.anti_join_indices(left, right, (("id", "id"),))

    assert torch.equal(semi_rows, torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(anti_rows, torch.tensor([0, 2], dtype=torch.int64))


def test_physical_grouped_aggregate_uses_unique_consecutive_for_sorted_keys(monkeypatch):
    from tpch_torch.backend.physical_aggregate import AggregateSpec, execute_grouped_aggregate
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_unique(*_args, **_kwargs):
        raise AssertionError("sorted grouped aggregate should use unique_consecutive, not generic unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    child = PhysicalTable(
        "sorted_input",
        {
            "k1": PhysicalValue(torch.tensor([1, 1, 1, 2, 2, 3], dtype=torch.int64)),
            "k2": PhysicalValue(torch.tensor([1, 1, 2, 1, 1, 1], dtype=torch.int64)),
            "v": PhysicalValue(torch.tensor([5.0, 7.0, 11.0, 13.0, 17.0, 19.0])),
        },
        ("k1", "k2", "v"),
        6,
    )

    result = execute_grouped_aggregate(
        child,
        ("k1", "k2"),
        (AggregateSpec("sum", "v", ("sum_v",)), AggregateSpec("count_star", None, ("count_order",))),
    )

    assert result.value_named("k1").require_tensor().tolist() == [1, 1, 2, 3]
    assert result.value_named("k2").require_tensor().tolist() == [1, 2, 1, 1]
    assert result.value_named("sum_v").require_tensor().tolist() == [12.0, 11.0, 30.0, 19.0]
    assert result.value_named("count_order").require_tensor().tolist() == [2, 1, 2, 1]


def test_physical_grouped_aggregate_uses_dense_dictionary_group_ids(monkeypatch):
    from tpch_torch.backend.physical_aggregate import AggregateSpec, execute_grouped_aggregate
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_unique(*_args, **_kwargs):
        raise AssertionError("dictionary grouped aggregate should avoid generic unique")

    def fail_unique_consecutive(*_args, **_kwargs):
        raise AssertionError("dictionary grouped aggregate should avoid unique_consecutive")

    monkeypatch.setattr(torch, "unique", fail_unique)
    monkeypatch.setattr(torch, "unique_consecutive", fail_unique_consecutive)

    child = PhysicalTable(
        "lineitem_chunk",
        {
            "l_returnflag": PhysicalValue(
                torch.tensor([1, 0, 1, 2], dtype=torch.int64),
                dictionary=("A", "N", "R"),
            ),
            "l_linestatus": PhysicalValue(
                torch.tensor([1, 0, 1, 0], dtype=torch.int64),
                dictionary=("F", "O"),
            ),
            "v": PhysicalValue(torch.tensor([5.0, 7.0, 11.0, 13.0])),
        },
        ("l_returnflag", "l_linestatus", "v"),
        4,
    )

    result = execute_grouped_aggregate(
        child,
        ("l_returnflag", "l_linestatus"),
        (AggregateSpec("sum", "v", ("sum_v",)),),
    )

    assert result.value_named("l_returnflag").require_tensor().tolist() == [0, 1, 2]
    assert result.value_named("l_linestatus").require_tensor().tolist() == [0, 1, 0]
    assert result.value_named("sum_v").require_tensor().tolist() == [7.0, 16.0, 13.0]


def test_physical_expression_folds_same_column_literal_or(monkeypatch):
    from tpch_torch.backend.physical_expr import evaluate_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_logical_or(*_args, **_kwargs):
        raise AssertionError("same-column literal OR should use membership mask")

    monkeypatch.setattr(torch, "logical_or", fail_logical_or)

    table = PhysicalTable(
        "lineitem",
        {
            "l_shipmode": PhysicalValue(
                torch.tensor([0, 2, 5, 6], dtype=torch.int64),
                dictionary=("AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"),
            )
        },
        ("l_shipmode",),
        4,
    )

    result = evaluate_expression(table, "(l_shipmode = 'MAIL') OR (l_shipmode = 'SHIP')")

    assert torch.equal(result.require_tensor(), torch.tensor([False, True, True, False]))


def test_physical_expression_singleton_numeric_in_uses_fast_membership(monkeypatch):
    from tpch_torch.backend.physical_expr import evaluate_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_isin(*_args, **_kwargs):
        raise AssertionError("singleton numeric IN should not call torch.isin")

    monkeypatch.setattr(torch, "isin", fail_isin)

    table = PhysicalTable(
        "t",
        {"a": PhysicalValue(torch.tensor([1, 2, 1], dtype=torch.int64))},
        ("a",),
        3,
    )

    result = evaluate_expression(table, "a IN (1)")

    assert torch.equal(result.require_tensor(), torch.tensor([True, False, True]))


def test_physical_expression_disambiguates_symmetric_duplicate_name_predicate():
    from tpch_torch.backend.physical_expr import evaluate_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    left = PhysicalValue(torch.tensor([0, 0, 1, 2], dtype=torch.int64), dictionary=("FRANCE", "GERMANY", "BRAZIL"))
    right = PhysicalValue(torch.tensor([1, 2, 0, 1], dtype=torch.int64), dictionary=("FRANCE", "GERMANY", "BRAZIL"))
    table = PhysicalTable(
        "join",
        {"n_name": left, "n_name__1": right},
        ("n_name", "n_name__1"),
        4,
    )

    result = evaluate_expression(
        table,
        "(((n_name = 'FRANCE') AND (n_name = 'GERMANY')) OR "
        "((n_name = 'GERMANY') AND (n_name = 'FRANCE')))",
    )

    assert torch.equal(result.require_tensor(), torch.tensor([True, False, True, False]))


def test_physical_projection_keeps_qualified_duplicate_select_aliases_separate():
    from tpch_torch.backend.physical_projection import projection_output_name
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    left = PhysicalValue(torch.tensor([0], dtype=torch.int64), dictionary=("FRANCE", "GERMANY"))
    right = PhysicalValue(torch.tensor([1], dtype=torch.int64), dictionary=("FRANCE", "GERMANY"))
    child = PhysicalTable(
        "join",
        {"n_name": left, "n1.n_name": left, "n_name__1": right, "n2.n_name": right},
        ("n_name", "n_name__1"),
        1,
    )

    _, aliases = projection_output_name(
        child,
        "supp_nation",
        0,
        left,
        {"supp_nation": "n1.n_name", "cust_nation": "n2.n_name"},
    )

    assert "supp_nation" in aliases
    assert "cust_nation" not in aliases




def test_physical_plan_executes_multi_branch_searched_case_group_by():
    con = duckdb.connect()
    con.execute("create table lineitem(l_quantity double)")
    con.execute("insert into lineitem values (5.0), (15.0), (25.0), (35.0)")

    result = validate_sql(
        con,
        """
        select case
                 when l_quantity < 10 then 1
                 when l_quantity < 30 then 2
                 else 3
               end as bucket,
               count(*) as n
        from lineitem
        group by bucket
        order by bucket
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"bucket": 1, "n": 1},
        {"bucket": 2, "n": 2},
        {"bucket": 3, "n": 1},
    ]


def test_physical_plan_executes_simple_case_group_by():
    con = duckdb.connect()
    con.execute("create table lineitem(l_returnflag varchar)")
    con.execute("insert into lineitem values ('A'), ('A'), ('N'), ('R')")

    result = validate_sql(
        con,
        """
        select case l_returnflag
                 when 'A' then 1
                 when 'N' then 2
                 else 3
               end as bucket,
               count(*) as n
        from lineitem
        group by bucket
        order by bucket
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"bucket": 1, "n": 2},
        {"bucket": 2, "n": 1},
        {"bucket": 3, "n": 1},
    ]


def test_physical_plan_executes_order_by_limit_duckdb_topn_rowid_shape():
    con = duckdb.connect()
    con.execute("create table lineitem(l_orderkey integer, l_quantity double)")
    con.execute(
        "insert into lineitem values (1, 5.0), (2, 35.0), (3, 15.0), (4, 45.0), (5, 25.0)"
    )

    result = validate_sql(
        con,
        """
        select l_orderkey, l_quantity
        from lineitem
        order by l_quantity desc
        limit 3
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"l_orderkey": 4, "l_quantity": 45.0},
        {"l_orderkey": 2, "l_quantity": 35.0},
        {"l_orderkey": 5, "l_quantity": 25.0},
    ]



def test_physical_topn_limit_uses_torch_topk(monkeypatch):
    original_topk = torch.topk
    calls = []

    def tracked_topk(*args, **kwargs):
        calls.append((args, kwargs))
        return original_topk(*args, **kwargs)

    monkeypatch.setattr(torch, "topk", tracked_topk)
    con = duckdb.connect()
    con.execute("create table lineitem(l_orderkey integer, l_quantity double)")
    con.execute("insert into lineitem values (1, 5.0), (2, 35.0), (3, 15.0), (4, 45.0), (5, 25.0)")

    result = validate_sql(
        con,
        """
        select l_orderkey, l_quantity
        from lineitem
        order by l_quantity desc
        limit 3
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert calls

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


def test_physical_scalar_sum_on_empty_input_returns_null():
    from tpch_torch.backend.physical import execute_physical_plan

    con = duckdb.connect()
    con.execute("create table r(x double)")
    plan = compile_sirius_plan(con, "select sum(x) as total from r")

    rows = execute_physical_plan(con, plan.operator_graph, device="cpu")

    assert rows == [{"total": None}]


def test_physical_arithmetic_propagates_null_validity():
    from tpch_torch.backend.physical_expr import evaluate_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    table = PhysicalTable(
        "aggregate",
        {
            "sum(#0)": PhysicalValue(
                torch.tensor([0.0], dtype=torch.float64),
                valid=torch.tensor([False], dtype=torch.bool),
            )
        },
        ("sum(#0)",),
        1,
    )

    value = evaluate_expression(table, "sum(#0) / 7.0")

    assert value.cell(0) is None


def test_physical_filter_treats_null_predicate_as_false():
    from tpch_torch.backend.physical import PhysicalPlanExecutor
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    table = PhysicalTable(
        "input",
        {
            "keep": PhysicalValue(
                torch.tensor([True, True], dtype=torch.bool),
                valid=torch.tensor([True, False], dtype=torch.bool),
            )
        },
        ("keep",),
        2,
    )

    mask = PhysicalPlanExecutor._filter_mask(table.value_named("keep"))

    assert torch.equal(mask, torch.tensor([True, False]))


def test_physical_plan_accepts_sum_no_overflow_aggregate_alias():
    from tpch_torch.backend.physical import _aggregate_specs
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    node = TQPOperatorNode(
        node_id="agg",
        kind=OperatorKind.AGGREGATE,
        name="PERFECT_HASH_GROUP_BY",
        metadata={"Aggregates": ["sum_no_overflow(#0)"]},
    )
    child = PhysicalTable(
        "input",
        {"amount": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64))},
        ("amount",),
        2,
    )

    specs = _aggregate_specs(node, child)

    assert [(spec.function, spec.argument) for spec in specs] == [("sum", "#0")]


def test_q1_physical_plan_uses_graph_lowered_fusion(monkeypatch):
    from tpch_torch.backend.physical import execute_physical_plan
    import tpch_torch.backend.physical_fusion as physical_fusion

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    plan = compile_sirius_plan(con, TPC_H_Q1_SQL)
    calls = []
    sentinel = [{"ok": 1}]

    def fused(con_arg, graph, device, scan_ranges=None):
        calls.append((graph.query_id, graph.root.kind, device, scan_ranges))
        return sentinel

    monkeypatch.setattr(physical_fusion, "try_execute_fused_physical_plan", fused)

    rows = execute_physical_plan(con, plan.operator_graph, device="cpu")

    assert rows == sentinel
    assert calls == [(1, plan.operator_graph.root.kind, "cpu", {})]


def test_q1_fused_physical_plan_matches_duckdb_fixture_without_generic_unique(monkeypatch):
    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)

    def fail_unique(*_args, **_kwargs):
        raise AssertionError("Q1 fused grouping should use dense ids, not generic torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    result = validate_sql_with_frontend(con, TPC_H_Q1_SQL, device="cpu", frontend="sirius")

    assert result.query_id == 1
    assert result.max_abs_error == 0.0
    assert [row["l_returnflag"] for row in result.pytorch_rows] == ["A", "N"]


@pytest.fixture(scope="module")
def tpch_con_physical():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("query_id,module_name,func_name", [
    (2, "tpch_torch.backend.tpch_graph_q02", "execute_q2_graph"),
    (3, "tpch_torch.backend.tpch_graph_q03", "execute_q3_graph"),
    (4, "tpch_torch.backend.tpch_graph_q04", "execute_q4_graph"),
    (5, "tpch_torch.backend.tpch_graph_q05", "execute_q5_graph"),
    (7, "tpch_torch.backend.tpch_graph_q07", "execute_q7_graph"),
    (8, "tpch_torch.backend.tpch_graph_q08", "execute_q8_graph"),
    (9, "tpch_torch.backend.tpch_graph_q09", "execute_q9_graph"),
    (10, "tpch_torch.backend.tpch_graph_q10", "execute_q10_graph"),
    (11, "tpch_torch.backend.tpch_graph_q11", "execute_q11_graph"),
    (12, "tpch_torch.backend.tpch_graph_q12", "execute_q12_graph"),
    (13, "tpch_torch.backend.tpch_graph_q13", "execute_q13_graph"),
    (14, "tpch_torch.backend.tpch_graph_q14", "execute_q14_graph"),
    (15, "tpch_torch.backend.tpch_graph_q15", "execute_q15_graph"),
    (16, "tpch_torch.backend.tpch_graph_q16", "execute_q16_graph"),
    (17, "tpch_torch.backend.tpch_graph_q17", "execute_q17_graph"),
    (18, "tpch_torch.backend.tpch_graph_q18", "execute_q18_graph"),
    (19, "tpch_torch.backend.tpch_graph_q19", "execute_q19_graph"),
    (20, "tpch_torch.backend.tpch_graph_q20", "execute_q20_graph"),
    (21, "tpch_torch.backend.tpch_graph_q21", "execute_q21_graph"),
    (22, "tpch_torch.backend.tpch_graph_q22", "execute_q22_graph"),
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


def test_q17_physical_plan_preserves_positional_extendedprice_projection(monkeypatch):
    con = duckdb.connect()
    con.execute(
        """
        create table lineitem(
            l_partkey bigint,
            l_quantity decimal(15,2),
            l_extendedprice decimal(15,2)
        )
        """
    )
    con.execute(
        """
        create table part(
            p_partkey bigint,
            p_brand varchar,
            p_container varchar
        )
        """
    )
    con.execute("insert into part values (1, 'Brand#23', 'MED BOX')")
    con.execute(
        """
        insert into lineitem values
            (1, 1.00, 70.00),
            (1, 10.00, 1000.00),
            (1, 10.00, 1000.00)
        """
    )
    module = __import__("tpch_torch.backend.tpch_graph_q17", fromlist=["execute_q17_graph"])
    monkeypatch.setattr(
        module,
        "execute_q17_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query recipe path used")),
    )

    result = validate_sql_with_frontend(
        con,
        """
        SELECT
            sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM
            lineitem,
            part
        WHERE
            p_partkey = l_partkey
            AND p_brand = 'Brand#23'
            AND p_container = 'MED BOX'
            AND l_quantity < (
                SELECT
                    0.2 * avg(l_quantity)
                FROM
                    lineitem
                WHERE
                    l_partkey = p_partkey)
        """,
        device="cpu",
        frontend="sirius",
    )

    assert result.query_id == 17
    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"avg_yearly": 10.0}]


def _q3_shape_con():
    con = duckdb.connect()
    con.execute("create table lineitem(l_orderkey integer, l_extendedprice double, l_discount double)")
    con.execute("create table orders(o_custkey integer, o_orderkey integer, o_orderdate date, o_shippriority integer)")
    con.execute("create table customer(c_custkey integer, c_mktsegment varchar)")
    con.execute("insert into lineitem values (1, 100.0, 0.10), (2, 50.0, 0.00), (3, 30.0, 0.20)")
    con.execute("insert into orders values (10, 1, DATE '1995-03-10', 0), (20, 2, DATE '1995-03-12', 1), (30, 3, DATE '1995-03-14', 2)")
    con.execute("insert into customer values (10, 'BUILDING'), (20, 'BUILDING'), (30, 'AUTOMOBILE')")
    return con


def test_physical_join_preserves_required_join_key_without_leaking_join_only_key():
    from tpch_torch.backend.physical import execute_physical_plan
    from tpch_torch.relational import compare_rows, run_duckdb_sql

    sql = """
        select l_orderkey, o_orderdate, o_shippriority,
               sum(l_extendedprice * (1.0 - l_discount)) as revenue
        from lineitem
        join orders on l_orderkey = o_orderkey
        join customer on o_custkey = c_custkey
        where c_mktsegment = 'BUILDING'
        group by l_orderkey, o_orderdate, o_shippriority
        order by revenue desc, o_orderdate
    """
    con = _q3_shape_con()
    plan = compile_sirius_plan(con, sql)

    rows = execute_physical_plan(con, plan.operator_graph, device="cpu")

    assert compare_rows(run_duckdb_sql(con, sql), rows) == 0.0
    assert rows == [
        {"l_orderkey": 1, "o_orderdate": "1995-03-10", "o_shippriority": 0, "revenue": 90.0},
        {"l_orderkey": 2, "o_orderdate": "1995-03-12", "o_shippriority": 1, "revenue": 50.0},
    ]


def test_physical_plan_executes_right_join():
    sql = """
        select a, name
        from t right join u on t.id = u.id
        order by name
    """

    result = validate_sql(_join_con(), sql, device="cpu")

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"a": 1, "name": "x"}, {"a": 2, "name": "x"}, {"a": 3, "name": "y"}]


def test_physical_plan_executes_right_join_with_unmatched_preserved_rows(monkeypatch):
    import tpch_torch.backend.graph as graph_backend

    monkeypatch.setattr(
        graph_backend,
        "execute_generic_sql_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old generic parser path used")),
    )
    con = duckdb.connect()
    con.execute("create table l(id integer, amount double)")
    con.execute("create table r(id integer, name varchar)")
    con.execute("insert into l values (1, 10.0)")
    con.execute("insert into r values (1, 'matched'), (2, 'unmatched')")

    result = validate_sql(
        con,
        "select name, count(amount) as count_amount from l right join r on l.id = r.id group by name order by name",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"name": "matched", "count_amount": 1},
        {"name": "unmatched", "count_amount": 0},
    ]


def test_physical_plan_counts_zero_on_right_join_unmatched_join_key():
    con = duckdb.connect()
    con.execute("create table orders(o_custkey integer, o_orderkey integer)")
    con.execute("create table customer(c_custkey integer)")
    con.execute("insert into orders values (1, 10), (1, 11)")
    con.execute("insert into customer values (1), (2)")

    result = validate_sql(
        con,
        """
        select c_count, count(*) as custdist
        from (
            select c_custkey, count(o_orderkey) as c_count
            from orders right join customer on o_custkey = c_custkey
            group by c_custkey
        ) c_orders
        group by c_count
        order by c_count
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"c_count": 0, "custdist": 1},
        {"c_count": 2, "custdist": 1},
    ]


def test_physical_plan_executes_semi_join_without_recipe(monkeypatch):
    import tpch_torch.backend.graph as graph_backend

    monkeypatch.setattr(
        graph_backend,
        "execute_generic_sql_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old generic parser path used")),
    )
    con = duckdb.connect()
    con.execute("create table l(id integer, amount double)")
    con.execute("create table r(id integer)")
    con.execute("insert into l values (1, 10.0), (2, 20.0), (3, 30.0)")
    con.execute("insert into r values (1), (1), (3)")

    result = validate_sql(
        con,
        "select id, amount from l where id in (select id from r) order by id",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"id": 1, "amount": 10.0}, {"id": 3, "amount": 30.0}]


def test_physical_join_right_semi_preserves_matching_right_rows():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    left = PhysicalTable(
        "left",
        {"k": PhysicalValue(torch.tensor([1, 1, 3], dtype=torch.int64))},
        ("k",),
        3,
    )
    right = PhysicalTable(
        "right",
        {
            "rk": PhysicalValue(torch.tensor([1, 2, 3], dtype=torch.int64)),
            "name": PhysicalValue(torch.tensor([0, 1, 2], dtype=torch.int64), dictionary=("a", "b", "c")),
        },
        ("rk", "name"),
        3,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "RIGHT_SEMI", "Conditions": "k = rk"},
    )

    result = _execute_join_node(node, left, right, "", ())

    assert result.order == right.order
    assert [result.columns["name"].cell(i) for i in range(result.row_count)] == ["a", "c"]


def test_physical_join_right_anti_preserves_unmatched_right_rows():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    left = PhysicalTable(
        "left",
        {"k": PhysicalValue(torch.tensor([1, 1, 3], dtype=torch.int64))},
        ("k",),
        3,
    )
    right = PhysicalTable(
        "right",
        {
            "rk": PhysicalValue(torch.tensor([1, 2, 3], dtype=torch.int64)),
            "name": PhysicalValue(torch.tensor([0, 1, 2], dtype=torch.int64), dictionary=("a", "b", "c")),
        },
        ("rk", "name"),
        3,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "RIGHT_ANTI", "Conditions": "k = rk"},
    )

    result = _execute_join_node(node, left, right, "", ())

    assert result.order == right.order
    assert [result.columns["name"].cell(i) for i in range(result.row_count)] == ["b"]


def test_physical_delim_right_join_keeps_outer_keys_before_subquery_payload():
    from tpch_torch.backend.physical_delim import execute_delim_join_result
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    outer = PhysicalTable(
        "partsupp",
        {
            "ps_partkey": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64)),
            "ps_availqty": PhysicalValue(torch.tensor([100, 200], dtype=torch.int64)),
            "ps_suppkey": PhysicalValue(torch.tensor([10, 20], dtype=torch.int64)),
        },
        ("ps_partkey", "ps_availqty", "ps_suppkey"),
        2,
    )
    subquery = PhysicalTable(
        "subquery",
        {
            "threshold": PhysicalValue(torch.tensor([5.0, 6.0], dtype=torch.float64)),
            "ps_partkey": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64)),
            "ps_suppkey": PhysicalValue(torch.tensor([10, 20], dtype=torch.int64)),
        },
        ("threshold", "ps_partkey", "ps_suppkey"),
        2,
    )
    node = TQPOperatorNode(
        node_id="delim",
        kind=OperatorKind.JOIN,
        name="RIGHT_DELIM_JOIN",
        metadata={
            "Join Type": "RIGHT",
            "Conditions": [
                "ps_partkey IS NOT DISTINCT FROM ps_partkey",
                "ps_suppkey IS NOT DISTINCT FROM ps_suppkey",
            ],
        },
    )

    table = execute_delim_join_result(node, outer, subquery, "select s_name from supplier", ("#2",))

    assert table.order[:4] == ("ps_availqty", "ps_partkey", "ps_suppkey", "threshold")
    assert table.value_at(2).cell(0) == 10


def test_physical_delim_semi_join_matches_correlated_delim_key_not_payload_key():
    from tpch_torch.backend.physical_delim import execute_delim_join_result
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    outer = PhysicalTable(
        "outer",
        {
            "s_name": PhysicalValue(torch.tensor([0], dtype=torch.int64), dictionary=("Supplier#000000074",)),
            "l_suppkey": PhysicalValue(torch.tensor([74], dtype=torch.int64)),
            "l_orderkey": PhysicalValue(torch.tensor([97], dtype=torch.int64)),
        },
        ("s_name", "l_suppkey", "l_orderkey"),
        1,
    )
    subquery = PhysicalTable(
        "subquery",
        {
            "l_suppkey": PhysicalValue(torch.tensor([32], dtype=torch.int64)),
            "l_suppkey__1": PhysicalValue(torch.tensor([74], dtype=torch.int64)),
            "l_orderkey": PhysicalValue(torch.tensor([97], dtype=torch.int64)),
            "l_orderkey__3": PhysicalValue(torch.tensor([97], dtype=torch.int64)),
            "delim.l_suppkey": PhysicalValue(torch.tensor([74], dtype=torch.int64)),
            "delim.l_orderkey": PhysicalValue(torch.tensor([97], dtype=torch.int64)),
        },
        ("l_suppkey", "l_suppkey__1", "l_orderkey", "l_orderkey__3"),
        1,
    )
    node = TQPOperatorNode(
        node_id="delim",
        kind=OperatorKind.JOIN,
        name="RIGHT_DELIM_JOIN",
        metadata={
            "Join Type": "RIGHT_SEMI",
            "Conditions": [
                "l_orderkey IS NOT DISTINCT FROM l_orderkey",
                "l_suppkey IS NOT DISTINCT FROM l_suppkey",
            ],
        },
    )

    result = execute_delim_join_result(node, outer, subquery, "", ())

    assert result.row_count == 1
    assert result.columns["s_name"].cell(0) == "Supplier#000000074"


def test_physical_join_conditions_parse_not_distinct_equality():
    from tpch_torch.backend.physical_join_exec import join_conditions
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="RIGHT_DELIM_JOIN",
        metadata={
            "Join Type": "RIGHT",
            "Conditions": [
                "ps_partkey IS NOT DISTINCT FROM ps_partkey",
                "ps_suppkey IS NOT DISTINCT FROM ps_suppkey",
            ],
        },
    )

    assert join_conditions(node) == (("ps_partkey", "ps_partkey"), ("ps_suppkey", "ps_suppkey"))


def test_physical_join_conditions_ignore_non_equi_residual_for_hash_keys():
    from tpch_torch.backend.physical_join_exec import join_conditions, residual_conditions
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "INNER", "Conditions": ["l_orderkey = l_orderkey", "l_suppkey != l_suppkey"]},
    )

    assert join_conditions(node) == (("l_orderkey", "l_orderkey"),)
    assert residual_conditions(node) == ("l_suppkey != l_suppkey",)


def test_physical_dependent_join_dummy_wrapper_keeps_payload_side():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    payload = PhysicalTable(
        "payload",
        {
            "o_orderkey": PhysicalValue(torch.tensor([1, 3], dtype=torch.int64)),
            "value": PhysicalValue(torch.tensor([10.0, 30.0], dtype=torch.float64)),
        },
        ("o_orderkey", "value"),
        2,
    )
    dummy = PhysicalTable(
        "dummy",
        {"__rowid__": PhysicalValue(torch.tensor([1], dtype=torch.int64))},
        ("__rowid__",),
        1,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "RIGHT_SEMI", "Conditions": "o_orderkey IS NOT DISTINCT FROM o_orderkey"},
    )

    result = _execute_join_node(node, payload, dummy, "", ())

    assert result.order == payload.order
    assert [result.columns["value"].cell(i) for i in range(result.row_count)] == [10.0, 30.0]


def test_physical_dependent_right_anti_dummy_wrapper_keeps_payload_side():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    payload = PhysicalTable(
        "payload",
        {
            "o_orderkey": PhysicalValue(torch.tensor([1, 3], dtype=torch.int64)),
            "value": PhysicalValue(torch.tensor([10.0, 30.0], dtype=torch.float64)),
        },
        ("o_orderkey", "value"),
        2,
    )
    dummy = PhysicalTable(
        "dummy",
        {"__rowid__": PhysicalValue(torch.tensor([1], dtype=torch.int64))},
        ("__rowid__",),
        1,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "RIGHT_ANTI", "Conditions": "o_orderkey IS NOT DISTINCT FROM o_orderkey"},
    )

    result = _execute_join_node(node, payload, dummy, "", ())

    assert result.order == payload.order
    assert [result.columns["value"].cell(i) for i in range(result.row_count)] == [10.0, 30.0]


def test_physical_dependent_right_join_dummy_wrapper_keeps_payload_side():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    payload = PhysicalTable(
        "payload",
        {
            "ps_partkey": PhysicalValue(torch.tensor([1, 3], dtype=torch.int64)),
            "subquery_value": PhysicalValue(torch.tensor([10.0, 30.0], dtype=torch.float64)),
        },
        ("subquery_value", "ps_partkey"),
        2,
    )
    dummy = PhysicalTable(
        "dummy",
        {"__rowid__": PhysicalValue(torch.tensor([1], dtype=torch.int64))},
        ("__rowid__",),
        1,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "RIGHT", "Conditions": "ps_partkey IS NOT DISTINCT FROM ps_partkey"},
    )

    result = _execute_join_node(node, payload, dummy, "", ())

    assert result.order == payload.order
    assert [result.columns["subquery_value"].cell(i) for i in range(result.row_count)] == [10.0, 30.0]


def test_physical_inner_join_refreshes_alias_for_uniquified_equivalent_key():
    from tpch_torch.backend.physical import _execute_join_node
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
    from tpch_torch.operator_graph import OperatorKind, TQPOperatorNode

    left = PhysicalTable(
        "left",
        {"ps_partkey__4": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64))},
        ("ps_partkey__4",),
        2,
    )
    right = PhysicalTable(
        "right",
        {"p_partkey": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64))},
        ("p_partkey",),
        2,
    )
    node = TQPOperatorNode(
        node_id="join",
        kind=OperatorKind.JOIN,
        name="HASH_JOIN",
        metadata={"Join Type": "INNER", "Conditions": "ps_partkey = p_partkey"},
    )

    result = _execute_join_node(node, left, right, "", ())

    assert result.value_named("p_partkey") is result.columns["ps_partkey__4"]


def test_sirius_frontend_continues_to_physical_graph_when_generic_parser_errors():
    con = duckdb.connect()
    con.execute("create table l(id integer, amount double)")
    con.execute("create table r(id integer)")

    plan = compile_sirius_plan(con, "select id, amount from l where id in (select id from r) order by id")

    assert plan.generic_plan is None
    assert "ValueError" in (plan.generic_error or "")
    assert plan.operator_graph is not None


def test_physical_projection_maps_single_child_column_alias_to_position_ref():
    con = duckdb.connect()
    con.execute("create table customer(c_custkey integer)")
    con.execute("create table orders(o_custkey integer, o_orderkey integer, o_comment varchar)")
    con.execute("insert into customer values (1), (2)")
    con.execute("insert into orders values (1, 10, 'plain'), (1, 11, 'special requests')")

    result = validate_sql(
        con,
        """
        select c_count, count(*) as custdist
        from (
            select c_custkey, count(o_orderkey)
            from customer left outer join orders
              on c_custkey = o_custkey and o_comment not like '%special%requests%'
            group by c_custkey
        ) as c_orders(c_custkey, c_count)
        group by c_count
        order by c_count
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"c_count": 0, "custdist": 1}, {"c_count": 1, "custdist": 1}]




def test_physical_plan_executes_having_with_alias_and_count_expression():
    con = duckdb.connect()
    con.execute("create table lineitem(l_returnflag varchar, l_quantity double)")
    con.execute("insert into lineitem values ('A', 10.0), ('A', 30.0), ('N', 100.0), ('R', 5.0)")

    result = validate_sql(
        con,
        """
        select l_returnflag, sum(l_quantity) as total_qty, count(*) as n
        from lineitem
        group by l_returnflag
        having total_qty / count(*) > 20
        order by l_returnflag
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"l_returnflag": "N", "total_qty": 100.0, "n": 1},
    ]

def test_physical_filter_resolves_aggregate_expression_alias():
    con = duckdb.connect()
    con.execute("create table lineitem(l_orderkey integer, l_quantity double)")
    con.execute("insert into lineitem values (1, 100.0), (1, 250.0), (2, 10.0)")

    result = validate_sql(
        con,
        """
        select l_orderkey
        from lineitem
        group by l_orderkey
        having sum(l_quantity) > 300.0
        order by l_orderkey
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"l_orderkey": 1}]


def test_physical_expression_ignores_numeric_cast_wrapper():
    con = duckdb.connect()
    con.execute("create table r(x double, y integer)")
    con.execute("insert into r values (2.5, 4), (3.0, 5)")

    result = validate_sql(con, "select sum(cast(x as decimal(34,2)) * cast(y as decimal(34,0))) as total from r", device="cpu")

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"total": 25.0}]


def test_physical_plan_executes_multi_column_inner_join():
    con = duckdb.connect()
    con.execute("create table l(a integer, b integer, value double)")
    con.execute("create table r(a integer, b integer, name varchar)")
    con.execute("insert into l values (1, 1, 10.0), (1, 2, 20.0), (2, 1, 30.0)")
    con.execute("insert into r values (1, 2, 'x'), (2, 1, 'y'), (2, 2, 'z')")

    result = validate_sql(
        con,
        "select value, name from l join r on l.a = r.a and l.b = r.b order by value",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"value": 20.0, "name": "x"}, {"value": 30.0, "name": "y"}]



def test_physical_projection_resolves_child_alias_chain():
    con = duckdb.connect()
    con.execute("create table r(a integer, b integer)")
    con.execute("insert into r values (1, 10), (2, 20)")

    result = validate_sql(
        con,
        "select renamed from (select a as renamed, b from r) s order by renamed",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"renamed": 1}, {"renamed": 2}]


def test_physical_expression_executes_not_like_operator():
    con = duckdb.connect()
    con.execute("create table r(comment varchar)")
    con.execute("insert into r values ('plain request'), ('very special requests'), ('other')")

    result = validate_sql(
        con,
        "select count(*) as count_star from r where comment not like '%special%requests%'",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"count_star": 2}]


def test_physical_join_preserves_key_needed_by_parent_join():
    con = duckdb.connect()
    con.execute("create table customer(c_custkey integer, c_nationkey integer)")
    con.execute("create table nation(n_nationkey integer, n_name varchar)")
    con.execute("create table supplier(s_nationkey integer, s_name varchar)")
    con.execute("insert into customer values (1, 10), (2, 20)")
    con.execute("insert into nation values (10, 'ALGERIA'), (20, 'BRAZIL')")
    con.execute("insert into supplier values (10, 'Supplier#000000001'), (20, 'Supplier#000000002')")

    result = validate_sql(
        con,
        """
        select s_name, n_name
        from customer
        join nation on c_nationkey = n_nationkey
        join supplier on c_nationkey = s_nationkey
        order by s_name
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"s_name": "Supplier#000000001", "n_name": "ALGERIA"},
        {"s_name": "Supplier#000000002", "n_name": "BRAZIL"},
    ]


def test_physical_projection_resolves_nested_select_aliases_and_extract_year():
    con = duckdb.connect()
    con.execute("create table r(name varchar, event_date date, price double, discount double)")
    con.execute("insert into r values ('FRANCE', DATE '1995-04-01', 100.0, 0.10)")

    result = validate_sql(
        con,
        """
        select nation_alias, event_year, volume
        from (
            select name as nation_alias,
                   extract(year from event_date) as event_year,
                   price * (1 - discount) as volume
            from r
        ) s
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"nation_alias": "FRANCE", "event_year": 1995, "volume": 90.0}]


def test_physical_projection_rewrites_aggregate_expression_without_select_alias():
    con = duckdb.connect()
    con.execute("create table r(x double, y integer)")
    con.execute("insert into r values (2.5, 4), (3.0, 5)")

    result = validate_sql(
        con,
        "select total * 0.0001 as scaled from (select sum(cast(x as decimal(34,2)) * cast(y as decimal(34,0))) as total from r) s",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"scaled": 0.0025}]


def test_physical_join_appends_parent_required_internal_keys_after_payload():
    from tpch_torch.backend.physical_join import combine_join_tables
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    left = PhysicalTable(
        "left",
        {
            "k1": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64)),
            "v1": PhysicalValue(torch.tensor([10.0, 20.0], dtype=torch.float64)),
            "v2": PhysicalValue(torch.tensor([0.1, 0.2], dtype=torch.float64)),
            "k2": PhysicalValue(torch.tensor([3, 4], dtype=torch.int64)),
            "name": PhysicalValue(torch.tensor([0, 1], dtype=torch.int64), dictionary=("A", "B")),
        },
        ("k1", "v1", "v2", "k2", "name"),
        2,
    )
    right = PhysicalTable(
        "right",
        {
            "rk1": PhysicalValue(torch.tensor([1, 2], dtype=torch.int64)),
            "rk2": PhysicalValue(torch.tensor([3, 4], dtype=torch.int64)),
        },
        ("rk1", "rk2"),
        2,
    )

    joined = combine_join_tables(
        left,
        right,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        ("k1", "k2"),
        ("rk1", "rk2"),
        "select name, sum(v1 * (1 - v2)) from left join right on k1 = rk1 and k2 = rk2 group by name",
        ("k1 = later", "k2 = later"),
    )

    assert joined.order == ("v1", "v2", "name", "k1", "k2")


def test_physical_plan_resolves_self_join_table_alias_outputs():
    con = duckdb.connect()
    con.execute("create table nation(n_nationkey integer, n_name varchar)")
    con.execute("create table pair(supp_nationkey integer, cust_nationkey integer)")
    con.execute("insert into nation values (1, 'FRANCE'), (2, 'GERMANY')")
    con.execute("insert into pair values (1, 2)")

    result = validate_sql(
        con,
        """
        select n1.n_name as supp_nation, n2.n_name as cust_nation
        from pair
        join nation n1 on supp_nationkey = n1.n_nationkey
        join nation n2 on cust_nationkey = n2.n_nationkey
        order by supp_nation, cust_nation
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"supp_nation": "FRANCE", "cust_nation": "GERMANY"}]


def test_physical_case_expression_resolves_prior_projection_alias():
    con = duckdb.connect()
    con.execute("create table r(nation varchar, price double, discount double)")
    con.execute("insert into r values ('BRAZIL', 100.0, 0.10), ('FRANCE', 40.0, 0.25)")

    result = validate_sql(
        con,
        """
        select nation,
               case when nation = 'BRAZIL' then volume else 0.0 end as brazil_volume,
               volume
        from (
            select nation, price * (1 - discount) as volume
            from r
        ) s
        order by nation
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"nation": "BRAZIL", "brazil_volume": 90.0, "volume": 90.0},
        {"nation": "FRANCE", "brazil_volume": 0.0, "volume": 30.0},
    ]


def test_physical_projection_maps_multiple_aggregate_calls_to_child_columns():
    con = duckdb.connect()
    con.execute("create table r(k integer, nation varchar, amount double)")
    con.execute("insert into r values (1995, 'BRAZIL', 2.0), (1995, 'FRANCE', 3.0)")

    result = validate_sql(
        con,
        """
        select k,
               sum(case when nation = 'BRAZIL' then amount else 0.0 end) / sum(amount) as share
        from r
        group by k
        order by k
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"k": 1995, "share": 0.4}]


def test_physical_projection_rewrites_aggregate_call_without_dropping_trailing_arithmetic():
    from tpch_torch.backend.physical_projection import projection_value_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    table = PhysicalTable(
        "aggregate",
        {
            "sum(CAST(x AS DECIMAL(34,2)) * CAST(y AS DECIMAL(34,0)))": PhysicalValue(
                torch.tensor([25.0], dtype=torch.float64)
            )
        },
        ("sum(CAST(x AS DECIMAL(34,2)) * CAST(y AS DECIMAL(34,0)))",),
        1,
    )
    aliases = {
        "scaled": "total * 0.0001",
        "total": "sum(cast(x as decimal(34,2)) * cast(y as decimal(34,0)))",
    }

    assert projection_value_expression(aliases, table, "scaled") == "(#0) * 0.0001"


def test_physical_plan_executes_scalar_subquery_first_aggregate():
    con = duckdb.connect()
    con.execute("create table r(x double)")
    con.execute("insert into r values (2.0), (4.0)")

    result = validate_sql(
        con,
        "select x from r where x > (select avg(x) from r) order by x",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"x": 4.0}]


def test_physical_projection_refreshes_positional_alias_after_reordering():
    con = duckdb.connect()
    con.execute("create table r(k integer, v double)")
    con.execute("insert into r values (1, 10.0), (2, 20.0)")

    result = validate_sql(
        con,
        """
        select value
        from (
            select k, sum(v) as value
            from r
            group by k
        ) s
        where value > 15.0
        order by value
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"value": 20.0}]


def test_physical_projection_position_ref_ignores_stale_positional_alias():
    from tpch_torch.backend.physical_projection import projection_value_expression
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    first = PhysicalValue(torch.tensor([1], dtype=torch.int64))
    second = PhysicalValue(torch.tensor([2], dtype=torch.int64))
    table = PhysicalTable(
        "projection",
        {"semantic": first, "other": second, "#0": second},
        ("semantic", "other"),
        1,
    )

    assert projection_value_expression({}, table, "#0") == "#0"


def test_physical_projection_position_ref_prefers_semantic_output_name():
    from tpch_torch.backend.physical_projection import projection_output_name
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    value = PhysicalValue(torch.tensor([1], dtype=torch.int64))
    table = PhysicalTable(
        "projection",
        {"#0": value, "semantic": value, "projection.semantic": value},
        ("#0",),
        1,
    )

    name, aliases = projection_output_name(table, "#0", 0, value, {})

    assert name == "semantic"
    assert aliases == ("semantic", "projection.semantic", "#0")


def test_physical_projection_position_ref_inherits_equivalent_key_aliases():
    from tpch_torch.backend.physical_projection import projection_output_name
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    sibling = PhysicalValue(torch.tensor([1], dtype=torch.int64))
    selected = PhysicalValue(torch.tensor([1], dtype=torch.int64))
    table = PhysicalTable(
        "join",
        {
            "p_partkey": sibling,
            "ps_partkey__4": sibling,
            "ps_partkey__4__3": selected,
        },
        ("p_partkey", "ps_partkey__4__3"),
        1,
    )

    _name, aliases = projection_output_name(table, "#1", 0, selected, {})

    assert "p_partkey" in aliases
