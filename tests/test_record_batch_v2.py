import torch

from tpch_torch.record_batch import (
    AllocationOwner,
    BatchMeta,
    ColumnStorage,
    ColumnType,
    LogicalDType,
    StorageKind,
    TensorRecordBatch,
)


def test_tensor_record_batch_v2_preserves_duckdb_type_storage_chunk_and_owner():
    amount_type = ColumnType(
        name="amount",
        duckdb_type_id="DECIMAL",
        duckdb_type_repr="DECIMAL(15,2)",
        logical_dtype=LogicalDType.DECIMAL,
        nullable=True,
        precision=15,
        scale=2,
    )
    owner = AllocationOwner.torch()
    storage = ColumnStorage.decimal64(
        torch.tensor([100, 250, 999], dtype=torch.int64),
        validity=torch.tensor([True, True, False]),
        owner=owner,
    )
    batch = TensorRecordBatch.from_storages(
        columns={"amount": storage},
        types={"amount": amount_type},
        batch_meta=BatchMeta(
            row_count=3,
            chunk_size=1024,
            chunk_index=7,
            source_offset=4096,
            device=torch.device("cpu"),
        ),
    )

    filtered = batch.filter(torch.tensor([False, True, True]))
    gathered = filtered.gather(torch.tensor([1, 0], dtype=torch.int64))

    assert gathered.row_count == 2
    assert gathered.batch_meta.chunk_size == 1024
    assert gathered.batch_meta.chunk_index == 7
    assert gathered.batch_meta.source_offset == 4096
    assert gathered.types["amount"].duckdb_type_repr == "DECIMAL(15,2)"
    assert gathered.storage["amount"].kind == StorageKind.DECIMAL64
    assert gathered.storage["amount"].owner == owner
    assert gathered.columns["amount"].tolist() == [999, 250]
    assert gathered.validity["amount"].tolist() == [False, True]


def test_utf8_offsets_storage_distinguishes_empty_string_from_null_and_gathers():
    text_type = ColumnType.varchar("comment", nullable=True)
    storage = ColumnStorage.utf8_offsets(["ab", "", None, "é"], device="cpu")
    batch = TensorRecordBatch.from_storages(
        columns={"comment": storage},
        types={"comment": text_type},
        batch_meta=BatchMeta(row_count=4, chunk_size=4, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )

    gathered = batch.gather(torch.tensor([3, 2, 1, 0], dtype=torch.int64))

    assert gathered.storage["comment"].kind == StorageKind.UTF8_OFFSETS
    assert gathered.storage["comment"].validity.tolist() == [True, False, True, True]
    assert gathered.storage["comment"].decode_utf8() == ["é", None, "", "ab"]
    offsets = gathered.storage["comment"].children["offsets"]
    assert offsets.dtype == torch.int64
    assert offsets.tolist()[0] == 0
    assert offsets.tolist()[-1] == int(gathered.storage["comment"].children["chars"].numel())
