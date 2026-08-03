import duckdb
import pytest

from tpch_torch.backend.physical_partitionable import PartitionConfig, row_ranges
from tpch_torch.duckdb_bridge import create_lineitem_fixture
from tpch_torch.runner import compile_tqp_plan
from tpch_torch.sql import TPC_H_Q1_SQL


Q6_SQL = """
select sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1994-01-01'
  and l_shipdate < date '1995-01-01'
  and l_discount between 0.05 and 0.07
  and l_quantity < 24
""".strip()


FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1994-01-01"),
    ("N", "O", 23.0, 200.0, 0.07, 0.20, "1994-12-31"),
    ("A", "F", 24.0, 300.0, 0.06, 0.08, "1994-06-01"),
    ("R", "F", 1.0, 400.0, 0.04, 0.00, "1994-06-01"),
    ("R", "F", 1.0, 500.0, 0.06, 0.00, "1995-01-01"),
]


Q1_FIXTURE_ROWS = [
    ("N", "O", 10.0, 100.0, 0.05, 0.10, "1998-09-02"),
    ("N", "O", 20.0, 200.0, 0.10, 0.20, "1998-09-03"),
    ("A", "F", 5.0, 50.0, 0.00, 0.08, "1998-01-01"),
    ("N", "O", 30.0, 300.0, 0.05, 0.00, "1997-12-31"),
]


def test_row_ranges_cover_input_without_overlap():
    assert row_ranges(10, 4) == ((0, 4), (4, 8), (8, 10))


def test_partition_config_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        PartitionConfig(table="lineitem", chunk_size=0)


def test_partitionable_q6_matches_default_physical_execution():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.runner import run_sql_with_frontend

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = compile_tqp_plan(con, Q6_SQL, "sirius")

    partitioned = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )
    default = run_sql_with_frontend(con, Q6_SQL, device="cpu", frontend="sirius")

    assert partitioned == default.rows
    assert partitioned == [{"revenue": pytest.approx(19.0)}]


def test_partitionable_q1_matches_default_physical_execution():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.runner import run_sql_with_frontend

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    plan = compile_tqp_plan(con, TPC_H_Q1_SQL, "sirius")

    partitioned = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )
    default = run_sql_with_frontend(con, TPC_H_Q1_SQL, device="cpu", frontend="sirius")

    assert partitioned == default.rows
    assert [row["count_order"] for row in partitioned] == [1, 2]


def test_partitionable_execution_rejects_non_aggregate_graph():
    from tpch_torch.backend.graph import PyTorchGraphExecutor

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    sql = "select l_quantity from lineitem"
    plan = compile_tqp_plan(con, sql, "sirius")

    with pytest.raises(Exception, match="aggregate"):
        PyTorchGraphExecutor().execute(
            con,
            plan,
            device="cpu",
            partition_config=PartitionConfig(table="lineitem", chunk_size=2),
        )


def test_partitionable_execution_rejects_sort_not_covered_by_group_keys():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.errors import UnsupportedPlanError

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    sql = """
    select l_returnflag, sum(l_quantity) as total_qty
    from lineitem
    group by l_returnflag
    order by total_qty desc
    """.strip()
    plan = compile_tqp_plan(con, sql, "sirius")

    with pytest.raises(UnsupportedPlanError, match="ORDER BY"):
        PyTorchGraphExecutor().execute(
            con,
            plan,
            device="cpu",
            partition_config=PartitionConfig(table="lineitem", chunk_size=2),
        )


def test_partitionable_execution_rejects_descending_group_key_order():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.errors import UnsupportedPlanError

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    sql = """
    select l_returnflag, sum(l_quantity) as total_qty
    from lineitem
    group by l_returnflag
    order by l_returnflag desc
    """.strip()
    plan = compile_tqp_plan(con, sql, "sirius")

    with pytest.raises(UnsupportedPlanError, match="ascending"):
        PyTorchGraphExecutor().execute(
            con,
            plan,
            device="cpu",
            partition_config=PartitionConfig(table="lineitem", chunk_size=2),
        )


def test_partitionable_q1_uses_batch_pipeline_not_per_chunk_physical_executor(monkeypatch):
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical import PhysicalPlanExecutor

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    plan = compile_tqp_plan(con, TPC_H_Q1_SQL, "sirius")

    def fail_executor_execute(self):
        raise AssertionError("partitionable aggregate execution should use batch pipeline")

    monkeypatch.setattr(PhysicalPlanExecutor, "execute", fail_executor_execute)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )

    assert [row["count_order"] for row in rows] == [1, 2]


def test_partitionable_q1_pushes_scan_filter_and_prunes_shipdate(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor

    con = duckdb.connect()
    create_lineitem_fixture(con, Q1_FIXTURE_ROWS)
    plan = compile_tqp_plan(con, TPC_H_Q1_SQL, "sirius")
    observed = []
    original_stream = physical.fetch_physical_table_stream

    def tracked_stream(*args, **kwargs):
        observed.append(
            {
                "fetched_columns": args[2],
                "scan_filters": kwargs.get("scan_filters", ()),
            }
        )
        yield from original_stream(*args, **kwargs)

    monkeypatch.setattr(physical, "fetch_physical_table_stream", tracked_stream)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )

    assert [row["count_order"] for row in rows] == [1, 2]
    assert "l_shipdate" not in observed[0]["fetched_columns"]
    assert observed[0]["scan_filters"] == ("l_shipdate<='1998-09-02'::DATE",)


def test_partitionable_q6_merges_scan_filters_and_prunes_filter_only_columns(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = compile_tqp_plan(con, Q6_SQL, "sirius")
    observed = []
    original_stream = physical.fetch_physical_table_stream

    def tracked_stream(*args, **kwargs):
        observed.append(
            {
                "fetched_columns": args[2],
                "scan_filters": kwargs.get("scan_filters", ()),
            }
        )
        yield from original_stream(*args, **kwargs)

    monkeypatch.setattr(physical, "fetch_physical_table_stream", tracked_stream)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )

    assert rows == [{"revenue": pytest.approx(19.0)}]
    assert observed[0]["fetched_columns"] == ("l_discount", "l_extendedprice")
    assert observed[0]["scan_filters"] == (
        "l_shipdate>='1994-01-01'::DATE AND l_shipdate<'1995-01-01'::DATE",
        "l_discount>=0.05 AND l_discount<=0.07",
        "l_quantity<24.0",
    )


def test_partitionable_q6_uses_batch_pipeline_not_per_chunk_physical_executor(monkeypatch):
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical import PhysicalPlanExecutor

    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    plan = compile_tqp_plan(con, Q6_SQL, "sirius")

    def fail_executor_execute(self):
        raise AssertionError("partitionable aggregate execution should use batch pipeline")

    monkeypatch.setattr(PhysicalPlanExecutor, "execute", fail_executor_execute)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="lineitem", chunk_size=2),
    )

    assert rows == [{"revenue": pytest.approx(19.0)}]
