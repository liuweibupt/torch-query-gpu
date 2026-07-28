from decimal import Decimal

import torch

from tpch_torch.backend.expression_plan import add, col, compile_projection, div, lit, mul, project_expressions
from tpch_torch.record_batch import BatchMeta, ColumnStorage, ColumnType, LogicalDType, TensorRecordBatch


def _fixed_batch():
    return TensorRecordBatch.from_storages(
        columns={
            "a": ColumnStorage.fixed(torch.tensor([1, 2, 3], dtype=torch.int64)),
            "b": ColumnStorage.fixed(torch.tensor([10, 20, 30], dtype=torch.int64)),
        },
        types={"a": ColumnType.int64("a"), "b": ColumnType.int64("b")},
        batch_meta=BatchMeta(row_count=3, chunk_size=3, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )


def test_projection_plan_constant_folds_literals_and_executes_on_batch_device():
    batch = _fixed_batch()

    result = project_expressions(batch, {"five": add(lit(2), lit(3))})

    assert result.columns["five"].device == batch.batch_meta.device
    assert result.columns["five"].dtype == torch.int64
    assert result.columns["five"].tolist() == [5, 5, 5]
    assert result.types["five"].logical_dtype == LogicalDType.INT64


def test_projection_plan_reuses_common_subexpressions_for_multiple_outputs():
    batch = _fixed_batch()
    shared = add(col("a"), col("b"))

    plan = compile_projection(batch, {"x": mul(shared, lit(2)), "y": mul(shared, lit(3))})
    result = plan.execute(batch)

    add_primitives = [primitive for primitive in plan.primitives if primitive.op == "add"]
    assert len(add_primitives) == 1
    assert result.columns["x"].tolist() == [22, 44, 66]
    assert result.columns["y"].tolist() == [33, 66, 99]


def test_projection_plan_aligns_decimal_scales_in_ast_lowering():
    batch = TensorRecordBatch.from_storages(
        columns={
            "a": ColumnStorage.decimal64(torch.tensor([1234, 2000], dtype=torch.int64)),
            "b": ColumnStorage.decimal64(torch.tensor([100, 250], dtype=torch.int64)),
        },
        types={
            "a": ColumnType.decimal("a", precision=10, scale=2),
            "b": ColumnType.decimal("b", precision=10, scale=4),
        },
        batch_meta=BatchMeta(row_count=2, chunk_size=2, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )

    result = project_expressions(batch, {"sum_ab": add(col("a"), col("b"))})

    assert result.types["sum_ab"].logical_dtype == LogicalDType.DECIMAL
    assert result.types["sum_ab"].scale == 4
    assert result.columns["sum_ab"].tolist() == [123500, 200250]


def test_projection_plan_materializes_decimal_literals_with_scale():
    batch = TensorRecordBatch.from_storages(
        columns={"amount": ColumnStorage.decimal64(torch.tensor([1234, 2000], dtype=torch.int64))},
        types={"amount": ColumnType.decimal("amount", precision=10, scale=2)},
        batch_meta=BatchMeta(row_count=2, chunk_size=2, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )

    result = project_expressions(batch, {"plus_tax": add(col("amount"), lit(Decimal("0.0500")))})

    assert result.types["plus_tax"].logical_dtype == LogicalDType.DECIMAL
    assert result.types["plus_tax"].scale == 4
    assert result.columns["plus_tax"].dtype == torch.int64
    assert result.columns["plus_tax"].tolist() == [123900, 200500]


def test_projection_plan_decimal_division_uses_real_values_not_scaled_payloads():
    batch = TensorRecordBatch.from_storages(
        columns={
            "amount": ColumnStorage.decimal64(torch.tensor([1234, 2000], dtype=torch.int64)),
            "rate": ColumnStorage.decimal64(torch.tensor([10, 25], dtype=torch.int64)),
        },
        types={
            "amount": ColumnType.decimal("amount", precision=10, scale=2),
            "rate": ColumnType.decimal("rate", precision=10, scale=1),
        },
        batch_meta=BatchMeta(row_count=2, chunk_size=2, chunk_index=0, source_offset=0, device=torch.device("cpu")),
    )

    result = project_expressions(batch, {"ratio": div(col("amount"), col("rate"))})

    assert result.types["ratio"].logical_dtype == LogicalDType.FP64
    assert result.columns["ratio"].dtype == torch.float64
    assert torch.allclose(result.columns["ratio"], torch.tensor([12.34, 8.0], dtype=torch.float64))
