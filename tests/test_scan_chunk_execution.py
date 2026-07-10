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
    original = physical.fetch_physical_table

    def tracked_fetch(*args, **kwargs):
        table = original(*args, **kwargs)
        meta = table.batch.batch_meta
        calls.append((kwargs.get("scan_range"), meta.chunk_size, meta.chunk_index, meta.source_offset))
        return table

    monkeypatch.setattr(physical, "fetch_physical_table", tracked_fetch)

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
    assert calls == [((0, 2), 2, 0, 0), ((2, 4), 2, 1, 2), ((4, 5), 2, 2, 4)]


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
    original = physical.fetch_physical_table

    def tracked_fetch(*args, **kwargs):
        table = original(*args, **kwargs)
        meta = table.batch.batch_meta
        calls.append((kwargs.get("scan_range"), meta.chunk_size, meta.chunk_index, meta.source_offset))
        return table

    monkeypatch.setattr(physical, "fetch_physical_table", tracked_fetch)

    rows = PyTorchGraphExecutor().execute(
        con,
        plan,
        device="cpu",
        partition_config=PartitionConfig(table="t", chunk_size=2),
    )

    assert rows == [{"total": 150}]
    assert calls == [((0, 2), 2, 0, 0), ((2, 4), 2, 1, 2), ((4, 5), 2, 2, 4)]
