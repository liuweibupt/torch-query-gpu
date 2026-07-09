import torch

from tpch_torch.batch_aggregate import grouped_sum_batch
from tpch_torch.batch_join import inner_join_indices_batch
from tpch_torch.record_batch import BatchMeta, ColumnStorage, ColumnType, TensorRecordBatch


def test_typed_batch_inner_join_supports_key_and_payload_metadata():
    left = TensorRecordBatch.from_storages(
        columns={
            "k": ColumnStorage.fixed(torch.tensor([1, 2, 3], dtype=torch.int64)),
            "payload": ColumnStorage.fixed(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)),
        },
        types={"k": ColumnType.int64("k"), "payload": ColumnType.fp32("payload")},
        batch_meta=BatchMeta(row_count=3, chunk_size=3, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )
    right = TensorRecordBatch.from_storages(
        columns={"k": ColumnStorage.fixed(torch.tensor([2, 3], dtype=torch.int64))},
        types={"k": ColumnType.int64("k")},
        batch_meta=BatchMeta(row_count=2, chunk_size=2, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )

    left_rows, right_rows = inner_join_indices_batch(left, right, left_keys=("k",), right_keys=("k",))

    assert torch.equal(left_rows, torch.tensor([1, 2], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))
    assert left_rows.device == left.batch_meta.device


def test_typed_batch_grouped_sum_preserves_decimal_type_and_batch_metadata():
    batch = TensorRecordBatch.from_storages(
        columns={
            "k": ColumnStorage.fixed(torch.tensor([1, 1, 2], dtype=torch.int64)),
            "amount": ColumnStorage.decimal64(torch.tensor([100, 250, 333], dtype=torch.int64)),
        },
        types={"k": ColumnType.int64("k"), "amount": ColumnType.decimal("amount", precision=12, scale=2)},
        batch_meta=BatchMeta(row_count=3, chunk_size=128, chunk_index=4, source_offset=512, device=torch.device("cpu")),
    )

    result = grouped_sum_batch(batch, group_keys=("k",), sum_columns=("amount",))

    assert result.row_count == 2
    assert result.batch_meta.chunk_size == 128
    assert result.batch_meta.chunk_index == 4
    assert result.types["sum_amount"].scale == 2
    assert result.columns["k"].tolist() == [1, 2]
    assert result.columns["sum_amount"].tolist() == [350, 333]
