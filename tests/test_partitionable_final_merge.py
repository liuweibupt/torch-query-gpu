import pytest
import torch

from tpch_torch.backend.physical_partitionable_final import (
    FinalAggregateColumn,
    FinalAggregatePlan,
    merge_partitioned_aggregate_tables,
)
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def _partial_table(keys, avg_values, count_values):
    return PhysicalTable(
        "partial",
        {
            "k": PhysicalValue(torch.tensor(keys, dtype=torch.int64), dictionary=("B", "A")),
            "avg_v": PhysicalValue(torch.tensor(avg_values, dtype=torch.float64)),
            "count_v": PhysicalValue(torch.tensor(count_values, dtype=torch.int64)),
        },
        ("k", "avg_v", "count_v"),
        len(keys),
    )


def test_tensor_final_merge_weighted_avg_and_dictionary_sort():
    plan = FinalAggregatePlan(
        group_columns=("k",),
        aggregates=(
            FinalAggregateColumn("avg_v", "avg"),
            FinalAggregateColumn("count_v", "count"),
        ),
        count_column="count_v",
        sort_by_group_keys=True,
    )
    merged = merge_partitioned_aggregate_tables(
        (
            _partial_table([1, 0], [10.0, 7.0], [2, 1]),
            _partial_table([1, 0], [20.0, 9.0], [1, 3]),
        ),
        plan,
    )

    assert merged.value_named("k").dictionary == ("B", "A")
    assert merged.value_named("k").require_tensor().tolist() == [1, 0]
    assert merged.value_named("count_v").require_tensor().tolist() == [3, 4]
    assert merged.value_named("avg_v").require_tensor().tolist() == pytest.approx([40.0 / 3.0, 8.5])


def test_tensor_final_merge_rejects_changed_dictionary():
    plan = FinalAggregatePlan(
        group_columns=("k",),
        aggregates=(FinalAggregateColumn("count_v", "count"),),
        count_column="count_v",
        sort_by_group_keys=False,
    )
    left = _partial_table([0], [1.0], [1])
    right = PhysicalTable(
        "partial",
        {
            "k": PhysicalValue(torch.tensor([0], dtype=torch.int64), dictionary=("A", "B")),
            "avg_v": PhysicalValue(torch.tensor([1.0], dtype=torch.float64)),
            "count_v": PhysicalValue(torch.tensor([1], dtype=torch.int64)),
        },
        ("k", "avg_v", "count_v"),
        1,
    )

    with pytest.raises(Exception, match="dictionary"):
        merge_partitioned_aggregate_tables((left, right), plan)


def test_tensor_final_merge_preserves_null_sum_semantics():
    plan = FinalAggregatePlan(
        group_columns=(),
        aggregates=(FinalAggregateColumn("sum_v", "sum"),),
        count_column=None,
        sort_by_group_keys=False,
    )
    partial = PhysicalTable(
        "partial",
        {
            "sum_v": PhysicalValue(
                torch.tensor([0.0, 0.0], dtype=torch.float64),
                valid=torch.tensor([False, False]),
            )
        },
        ("sum_v",),
        2,
    )

    merged = merge_partitioned_aggregate_tables((partial,), plan)

    assert merged.value_named("sum_v").cell(0) is None


def test_tensor_final_merge_sum_min_max_count_without_groups():
    plan = FinalAggregatePlan(
        group_columns=(),
        aggregates=(
            FinalAggregateColumn("sum_v", "sum"),
            FinalAggregateColumn("min_v", "min"),
            FinalAggregateColumn("max_v", "max"),
            FinalAggregateColumn("count_v", "count"),
        ),
        count_column="count_v",
        sort_by_group_keys=False,
    )
    first = PhysicalTable(
        "partial",
        {
            "sum_v": PhysicalValue(torch.tensor([10.0], dtype=torch.float64)),
            "min_v": PhysicalValue(torch.tensor([3.0], dtype=torch.float64)),
            "max_v": PhysicalValue(torch.tensor([7.0], dtype=torch.float64)),
            "count_v": PhysicalValue(torch.tensor([2], dtype=torch.int64)),
        },
        ("sum_v", "min_v", "max_v", "count_v"),
        1,
    )
    second = PhysicalTable(
        "partial",
        {
            "sum_v": PhysicalValue(torch.tensor([5.0], dtype=torch.float64)),
            "min_v": PhysicalValue(torch.tensor([2.0], dtype=torch.float64)),
            "max_v": PhysicalValue(torch.tensor([8.0], dtype=torch.float64)),
            "count_v": PhysicalValue(torch.tensor([1], dtype=torch.int64)),
        },
        ("sum_v", "min_v", "max_v", "count_v"),
        1,
    )

    merged = merge_partitioned_aggregate_tables((first, second), plan)

    assert merged.value_named("sum_v").require_tensor().tolist() == [15.0]
    assert merged.value_named("min_v").require_tensor().tolist() == [2.0]
    assert merged.value_named("max_v").require_tensor().tolist() == [8.0]
    assert merged.value_named("count_v").require_tensor().tolist() == [3]
