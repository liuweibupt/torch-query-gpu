import torch

from tpch_torch.backend.physical_aggregate import AggregateSpec, execute_grouped_aggregate
from tpch_torch.backend.physical_hash_join import hash_join_indices_for_values
from tpch_torch.backend.physical_join import inner_join_indices, inner_join_indices_for_values
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.record_batch import ColumnMeta, LogicalDType


def test_join_supports_int64_fp32_fp64_and_decimal_keys():
    for dtype in (torch.int64, torch.float32, torch.float64):
        left_rows, right_rows = inner_join_indices(
            torch.tensor([1, 2, 3], dtype=dtype),
            torch.tensor([2, 3], dtype=dtype),
        )
        assert torch.equal(left_rows, torch.tensor([1, 2], dtype=torch.int64))
        assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))

    decimal_left = PhysicalValue(
        torch.tensor([1234, 500, 1250], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=2),
    )
    decimal_right = PhysicalValue(
        torch.tensor([50, 125], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=1),
    )
    left_rows, right_rows = inner_join_indices_for_values(decimal_left, decimal_right)
    assert torch.equal(left_rows, torch.tensor([1, 2], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))


def test_grouped_sum_supports_int64_fp32_fp64_and_decimal_values():
    for dtype in (torch.int64, torch.float32, torch.float64):
        child = PhysicalTable(
            "t",
            {
                "k": PhysicalValue(torch.tensor([1, 1, 2], dtype=torch.int64)),
                "v": PhysicalValue(torch.tensor([1, 2, 3], dtype=dtype)),
            },
            ("k", "v"),
            3,
        )
        result = execute_grouped_aggregate(child, ("k",), (AggregateSpec("sum", "v", ("sum_v",)),))
        assert result.value_named("sum_v").require_tensor().dtype == dtype
        assert result.value_named("sum_v").require_tensor().tolist() == [3, 3]

    decimal_meta = ColumnMeta.decimal(precision=12, scale=2)
    child = PhysicalTable(
        "t",
        {
            "k": PhysicalValue(torch.tensor([1, 1, 2], dtype=torch.int64)),
            "v": PhysicalValue(torch.tensor([100, 250, 333], dtype=torch.int64), meta=decimal_meta),
        },
        ("k", "v"),
        3,
    )
    result = execute_grouped_aggregate(child, ("k",), (AggregateSpec("sum", "v", ("sum_v",)),))
    summed = result.value_named("sum_v")
    assert summed.meta == decimal_meta
    assert summed.require_tensor().tolist() == [350, 333]


def test_grouped_min_max_preserve_decimal_metadata():
    decimal_meta = ColumnMeta.decimal(precision=12, scale=2)
    child = PhysicalTable(
        "t",
        {
            "k": PhysicalValue(torch.tensor([1, 1, 2], dtype=torch.int64)),
            "v": PhysicalValue(torch.tensor([300, 250, 333], dtype=torch.int64), meta=decimal_meta),
        },
        ("k", "v"),
        3,
    )

    result = execute_grouped_aggregate(
        child,
        ("k",),
        (
            AggregateSpec("min", "v", ("min_v",)),
            AggregateSpec("max", "v", ("max_v",)),
        ),
    )

    assert result.value_named("min_v").meta == decimal_meta
    assert result.value_named("max_v").meta == decimal_meta
    assert result.value_named("min_v").require_tensor().tolist() == [250, 333]
    assert result.value_named("max_v").require_tensor().tolist() == [300, 333]


def test_grouped_avg_decimal_returns_fp64_real_values():
    decimal_meta = ColumnMeta.decimal(precision=12, scale=2)
    child = PhysicalTable(
        "t",
        {
            "k": PhysicalValue(torch.tensor([1, 1, 2], dtype=torch.int64)),
            "v": PhysicalValue(torch.tensor([100, 250, 333], dtype=torch.int64), meta=decimal_meta),
        },
        ("k", "v"),
        3,
    )

    result = execute_grouped_aggregate(child, ("k",), (AggregateSpec("avg", "v", ("avg_v",)),))
    averaged = result.value_named("avg_v")

    assert averaged.meta == ColumnMeta.fp64()
    assert averaged.require_tensor().dtype == torch.float64
    assert averaged.require_tensor().tolist() == [1.75, 3.33]


def test_hash_join_indices_for_values_matches_decimal_scale_aligned_keys():
    left = PhysicalValue(
        torch.tensor([100, 125, 130], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=2),
    )
    right = PhysicalValue(
        torch.tensor([10, 13], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=1),
    )

    left_rows, right_rows = hash_join_indices_for_values(left, right)

    assert torch.equal(left_rows, torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))
