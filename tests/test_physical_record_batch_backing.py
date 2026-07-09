import duckdb
import torch

from tpch_torch.backend.physical_scan import fetch_physical_table
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.record_batch import BatchMeta, ColumnStorage, ColumnType, TensorRecordBatch


def test_physical_scan_returns_tensor_record_batch_with_scan_range_chunk_metadata():
    con = duckdb.connect()
    con.execute("create table t(id bigint, amount decimal(15,2), label varchar)")
    con.execute("insert into t values (1, 10.00, 'a'), (2, 20.00, 'b'), (3, 30.00, 'a'), (4, 40.00, 'c')")

    table = fetch_physical_table(
        con,
        "t",
        ("id", "amount", "label"),
        ("id", "amount", "label"),
        "cpu",
        scan_range=(1, 3),
    )

    assert table.batch is not None
    assert table.batch.row_count == 2
    assert table.batch.batch_meta == BatchMeta(
        row_count=2,
        chunk_size=2,
        chunk_index=0,
        source_offset=1,
        device=torch.device("cpu"),
    )
    assert table.batch.types["amount"].duckdb_type_repr == "DECIMAL(15,2)"
    assert table.batch.types["label"].duckdb_type_id == "VARCHAR"
    assert table.value_named("amount").require_tensor().tolist() == [2000, 3000]
    assert table.value_named("label").dictionary == ("a", "b")


def test_physical_table_filter_gather_project_keep_tensor_record_batch_backing():
    batch = TensorRecordBatch.from_storages(
        columns={
            "id": ColumnStorage.fixed(torch.tensor([1, 2, 3], dtype=torch.int64)),
            "v": ColumnStorage.fixed(torch.tensor([10, 20, 30], dtype=torch.int64)),
        },
        types={"id": ColumnType.int64("id"), "v": ColumnType.int64("v")},
        batch_meta=BatchMeta(row_count=3, chunk_size=128, chunk_index=2, source_offset=256, device=torch.device("cpu")),
    )
    table = PhysicalTable.from_batch("t", batch, order=("id", "v"))

    filtered = table.filter(torch.tensor([False, True, True]))
    gathered = filtered.gather(torch.tensor([1, 0], dtype=torch.int64))
    projected = PhysicalTable.projected(
        "projection",
        [("v", gathered.value_named("v"), ("v",))],
        gathered.row_count,
    )

    assert gathered.batch is not None
    assert gathered.batch.row_count == 2
    assert gathered.batch.batch_meta.chunk_size == 128
    assert gathered.batch.batch_meta.chunk_index == 2
    assert gathered.value_named("v").require_tensor().tolist() == [30, 20]
    assert projected.batch is not None
    assert projected.batch.columns["v"].tolist() == [30, 20]


def test_fetch_physical_table_chunks_emits_tensor_record_batches_with_chunk_indices():
    from tpch_torch.backend.physical_scan import fetch_physical_table_chunks

    con = duckdb.connect()
    con.execute("create table t(id bigint)")
    con.execute("insert into t values (10), (20), (30), (40), (50)")

    chunks = tuple(fetch_physical_table_chunks(con, "t", ("id",), ("id",), "cpu", chunk_size=2))

    assert [chunk.value_named("id").require_tensor().tolist() for chunk in chunks] == [[10, 20], [30, 40], [50]]
    assert [chunk.batch.batch_meta.chunk_size for chunk in chunks] == [2, 2, 2]
    assert [chunk.batch.batch_meta.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.batch.batch_meta.source_offset for chunk in chunks] == [0, 2, 4]
    assert [chunk.batch.row_count for chunk in chunks] == [2, 2, 1]
