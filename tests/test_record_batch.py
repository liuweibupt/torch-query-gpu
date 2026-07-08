from decimal import Decimal

import torch

from tpch_torch.record_batch import ColumnMeta, LogicalDType, TensorRecordBatch


def test_column_meta_decimal_requires_int64_storage_and_scale():
    meta = ColumnMeta.decimal(precision=12, scale=2, nullable=True)

    assert meta.logical_dtype == LogicalDType.DECIMAL
    assert meta.torch_dtype == torch.int64
    assert meta.precision == 12
    assert meta.scale == 2
    assert meta.nullable is True
    assert meta.decode_scalar(12345) == Decimal("123.45")


def test_tensor_record_batch_filter_gather_and_project_preserve_metadata():
    amount = ColumnMeta.decimal(precision=10, scale=2, nullable=True)
    batch = TensorRecordBatch(
        columns={
            "id": torch.tensor([1, 2, 3], dtype=torch.int64),
            "amount": torch.tensor([100, 250, 999], dtype=torch.int64),
        },
        meta={"id": ColumnMeta.int64(), "amount": amount},
        validity={"amount": torch.tensor([True, True, False])},
    )

    filtered = batch.filter(torch.tensor([False, True, True]))
    gathered = filtered.gather(torch.tensor([1, 0], dtype=torch.int64))
    projected = gathered.project(("amount",))

    assert projected.row_count == 2
    assert projected.columns["amount"].tolist() == [999, 250]
    assert projected.validity["amount"].tolist() == [False, True]
    assert projected.meta["amount"] == amount
