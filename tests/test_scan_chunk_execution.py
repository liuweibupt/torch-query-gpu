import duckdb
import pytest

from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.runner import compile_tqp_plan, run_sql_with_frontend


def _create_amount_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("create table t(id bigint, amount bigint)")
    con.execute("insert into t values (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)")


def test_scan_chunk_config_rejects_invalid_chunk_size():
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    with pytest.raises(ValueError, match="chunk_size"):
        ScanChunkConfig(table="t", chunk_size=0)


def test_chunked_scan_filter_project_matches_default_and_preserves_chunk_metadata(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    sql = "select id, amount + 1 as inc from t where amount >= 20"
    plan = compile_tqp_plan(con, sql, "sirius")
    default = run_sql_with_frontend(con, sql, device="cpu", frontend="sirius")
    calls = []
    original_stream = physical.fetch_physical_table_stream

    def fail_offset_fetch(*args, **kwargs):
        raise AssertionError("chunk execution should use Arrow stream scan, not OFFSET/LIMIT fetch")

    def tracked_stream(*args, **kwargs):
        for table in original_stream(*args, **kwargs):
            meta = table.batch.batch_meta
            calls.append((meta.chunk_size, meta.chunk_index, meta.source_offset))
            yield table

    monkeypatch.setattr(physical, "fetch_physical_table", fail_offset_fetch)
    monkeypatch.setattr(physical, "fetch_physical_table_stream", tracked_stream)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
    )

    assert rows == default.rows == [
        {"id": 2, "inc": 21},
        {"id": 3, "inc": 31},
        {"id": 4, "inc": 41},
        {"id": 5, "inc": 51},
    ]
    assert calls == [(2, 0, 0), (2, 1, 2)]


def test_scan_chunk_pushes_filters_and_prunes_filter_only_columns(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    plan = compile_tqp_plan(con, "select id from t where amount >= 30", "sirius")
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
        scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
    )

    assert rows == [{"id": 3}, {"id": 4}, {"id": 5}]
    assert observed == [{"fetched_columns": ("id",), "scan_filters": ("amount>=30",)}]


def test_scan_chunk_pushes_filter_node_into_scan_source(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig
    from tpch_torch.backend.physical_pipeline import FilterBatchOperator

    con = duckdb.connect()
    _create_amount_table(con)
    sql = "select id from (select id, amount + 1 as inc from t) s where inc >= 31"
    plan = compile_tqp_plan(con, sql, "sirius")
    observed = []
    original_stream = physical.fetch_physical_table_stream

    def fail_filter_next_batch(self):
        raise AssertionError("base-table FILTER nodes should be merged into scan_filters")

    def tracked_stream(*args, **kwargs):
        observed.append(
            {
                "fetched_columns": args[2],
                "scan_filters": kwargs.get("scan_filters", ()),
            }
        )
        yield from original_stream(*args, **kwargs)

    monkeypatch.setattr(FilterBatchOperator, "next_batch", fail_filter_next_batch)
    monkeypatch.setattr(physical, "fetch_physical_table_stream", tracked_stream)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
    )

    assert rows == [{"id": 3}, {"id": 4}, {"id": 5}]
    assert observed == [
        {
            "fetched_columns": ("id", "amount"),
            "scan_filters": ("((amount + 1) >= 31)",),
        }
    ]


def test_scan_chunk_execution_uses_batch_pipeline_instead_of_per_chunk_physical_executor(monkeypatch):
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical import PhysicalPlanExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    sql = "select id, amount + 1 as inc from t where amount >= 20"
    plan = compile_tqp_plan(con, sql, "sirius")

    def fail_executor_execute(self):
        raise AssertionError("chunk execution should use BatchOperator.next_batch")

    monkeypatch.setattr(PhysicalPlanExecutor, "execute", fail_executor_execute)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
    )

    assert rows == [
        {"id": 2, "inc": 21},
        {"id": 3, "inc": 31},
        {"id": 4, "inc": 41},
        {"id": 5, "inc": 51},
    ]


def test_scan_chunk_pipeline_supports_literal_only_projection():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    plan = compile_tqp_plan(con, "select 1 as one from t", "sirius")

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
    )

    assert rows == [{"one": 1}, {"one": 1}, {"one": 1}, {"one": 1}, {"one": 1}]


def test_scan_chunk_execution_rejects_aggregate_plan_with_partition_hint():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    plan = compile_tqp_plan(con, "select sum(amount) as total from t", "sirius")

    with pytest.raises(UnsupportedPlanError, match="PartitionConfig"):
        PyTorchGraphExecutor().execute(
            con,
            plan,
            device="cpu",
            scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
        )


def test_scan_chunk_execution_rejects_join_plan():
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_chunked import ScanChunkConfig

    con = duckdb.connect()
    _create_amount_table(con)
    con.execute("create table u(id bigint, payload bigint)")
    con.execute("insert into u values (2, 200), (4, 400)")
    plan = compile_tqp_plan(con, "select t.id, u.payload from t join u on t.id = u.id", "sirius")

    with pytest.raises(UnsupportedPlanError, match="join"):
        PyTorchGraphExecutor().execute(
            con,
            plan,
            device="cpu",
            scan_chunk_config=ScanChunkConfig(table="t", chunk_size=2),
        )


def test_partitionable_execution_preserves_configured_scan_chunk_metadata(monkeypatch):
    import tpch_torch.backend.physical as physical
    from tpch_torch.backend.graph import PyTorchGraphExecutor
    from tpch_torch.backend.physical_partitionable import PartitionConfig

    con = duckdb.connect()
    _create_amount_table(con)
    plan = compile_tqp_plan(con, "select sum(amount) as total from t", "sirius")
    calls = []
    original_stream = physical.fetch_physical_table_stream

    def fail_offset_fetch(*args, **kwargs):
        raise AssertionError("partitionable execution should use Arrow stream scan, not OFFSET/LIMIT fetch")

    def tracked_stream(*args, **kwargs):
        for table in original_stream(*args, **kwargs):
            meta = table.batch.batch_meta
            calls.append((meta.chunk_size, meta.chunk_index, meta.source_offset))
            yield table

    monkeypatch.setattr(physical, "fetch_physical_table", fail_offset_fetch)
    monkeypatch.setattr(physical, "fetch_physical_table_stream", tracked_stream)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="t", chunk_size=2),
    )

    assert rows == [{"total": 150}]
    assert calls == [(2, 0, 0), (2, 1, 2), (2, 2, 4)]
