import torch

from tpch_torch.backend.physical_aggregate import (
    AggregateSpec,
    execute_grouped_aggregate,
    execute_ungrouped_aggregate,
)
from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_join import (
    inner_join_indices,
    inner_join_indices_for_values,
    join_indices_for_conditions,
    semi_join_indices,
)
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def test_inner_join_indices_preserves_fractional_float_keys():
    left_rows, right_rows = inner_join_indices(
        torch.tensor([1.2, 1.8, 2.0, 2.2], dtype=torch.float64),
        torch.tensor([1.8, 2.2], dtype=torch.float64),
    )

    assert torch.equal(left_rows, torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))


def test_sorted_unique_join_metadata_preserves_fractional_float_keys():
    left = PhysicalValue(torch.tensor([1.2, 1.8, 2.0, 2.2], dtype=torch.float64))
    right = PhysicalValue(
        torch.tensor([1.8, 2.2], dtype=torch.float64),
        sorted_non_decreasing=True,
        unique=True,
    )

    left_rows, right_rows = inner_join_indices_for_values(left, right)

    assert torch.equal(left_rows, torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0, 1], dtype=torch.int64))


def test_semi_join_indices_preserves_fractional_float_keys():
    left = PhysicalTable(
        "left",
        {"k": PhysicalValue(torch.tensor([1.2, 1.8, 2.0], dtype=torch.float64))},
        ("k",),
        3,
    )
    right = PhysicalTable(
        "right",
        {"k": PhysicalValue(torch.tensor([1.8], dtype=torch.float64))},
        ("k",),
        1,
    )

    rows = semi_join_indices(left, right, (("k", "k"),))

    assert torch.equal(rows, torch.tensor([1], dtype=torch.int64))


def test_composite_join_filters_fractional_float_conditions_without_truncation():
    left = PhysicalTable(
        "left",
        {
            "k1": PhysicalValue(torch.tensor([1, 1, 1], dtype=torch.int64)),
            "k2": PhysicalValue(torch.tensor([1.2, 1.8, 1.2], dtype=torch.float64)),
        },
        ("k1", "k2"),
        3,
    )
    right = PhysicalTable(
        "right",
        {
            "rk1": PhysicalValue(torch.tensor([1], dtype=torch.int64)),
            "rk2": PhysicalValue(torch.tensor([1.8], dtype=torch.float64)),
        },
        ("rk1", "rk2"),
        1,
    )

    left_rows, right_rows = join_indices_for_conditions(left, right, (("k1", "rk1"), ("k2", "rk2")))

    assert torch.equal(left_rows, torch.tensor([1], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([0], dtype=torch.int64))


def test_grouped_integer_min_max_uses_integer_safe_initializers():
    child = PhysicalTable(
        "input",
        {
            "k": PhysicalValue(torch.tensor([1, 1, 2], dtype=torch.int64)),
            "v": PhysicalValue(torch.tensor([7, -3, 5], dtype=torch.int64)),
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

    assert result.value_named("min_v").require_tensor().tolist() == [-3, 5]
    assert result.value_named("max_v").require_tensor().tolist() == [7, 5]


def test_null_aware_boolean_or_and_not_follow_sql_three_valued_logic():
    table = PhysicalTable(
        "input",
        {
            "unknown": PhysicalValue(
                torch.tensor([False], dtype=torch.bool),
                valid=torch.tensor([False], dtype=torch.bool),
            ),
            "truthy": PhysicalValue(torch.tensor([True], dtype=torch.bool)),
            "falsy": PhysicalValue(torch.tensor([False], dtype=torch.bool)),
        },
        ("unknown", "truthy", "falsy"),
        1,
    )

    assert evaluate_expression(table, "unknown OR truthy").cell(0) is True
    assert evaluate_expression(table, "unknown OR falsy").cell(0) is None
    assert evaluate_expression(table, "unknown AND truthy").cell(0) is None
    assert evaluate_expression(table, "unknown AND falsy").cell(0) is False
    assert evaluate_expression(table, "NOT unknown").cell(0) is None


def test_case_when_treats_unknown_condition_as_not_matched():
    table = PhysicalTable(
        "input",
        {
            "maybe": PhysicalValue(
                torch.tensor([True], dtype=torch.bool),
                valid=torch.tensor([False], dtype=torch.bool),
            )
        },
        ("maybe",),
        1,
    )

    result = evaluate_expression(table, "CASE WHEN maybe THEN 1 ELSE 2 END")

    assert result.cell(0) == 2


def test_string_comparison_and_in_propagate_null_validity():
    table = PhysicalTable(
        "input",
        {
            "s": PhysicalValue(
                torch.tensor([0, 1], dtype=torch.int64),
                dictionary=("A", "B"),
                valid=torch.tensor([False, True], dtype=torch.bool),
            )
        },
        ("s",),
        2,
    )

    equals = evaluate_expression(table, "s = 'A'")
    in_list = evaluate_expression(table, "s IN ('A', 'B')")

    assert equals.cell(0) is None
    assert equals.cell(1) is False
    assert in_list.cell(0) is None
    assert in_list.cell(1) is True


def test_scalar_aggregates_ignore_invalid_values_and_return_null_when_all_invalid():
    child = PhysicalTable(
        "input",
        {
            "v": PhysicalValue(
                torch.tensor([10.0, 999.0, 888.0], dtype=torch.float64),
                valid=torch.tensor([True, False, False], dtype=torch.bool),
            )
        },
        ("v",),
        3,
    )

    result = execute_ungrouped_aggregate(
        child,
        (
            AggregateSpec("sum", "v", ("sum_v",)),
            AggregateSpec("avg", "v", ("avg_v",)),
            AggregateSpec("min", "v", ("min_v",)),
            AggregateSpec("max", "v", ("max_v",)),
            AggregateSpec("count", "v", ("count_v",)),
        ),
    )

    assert result.value_named("sum_v").cell(0) == 10.0
    assert result.value_named("avg_v").cell(0) == 10.0
    assert result.value_named("min_v").cell(0) == 10.0
    assert result.value_named("max_v").cell(0) == 10.0
    assert result.value_named("count_v").cell(0) == 1

    all_invalid = PhysicalTable(
        "input",
        {
            "v": PhysicalValue(
                torch.tensor([999.0], dtype=torch.float64),
                valid=torch.tensor([False], dtype=torch.bool),
            )
        },
        ("v",),
        1,
    )
    null_sum = execute_ungrouped_aggregate(all_invalid, (AggregateSpec("sum", "v", ("sum_v",)),))

    assert null_sum.value_named("sum_v").cell(0) is None


def test_grouped_aggregates_ignore_invalid_values_per_group():
    child = PhysicalTable(
        "input",
        {
            "k": PhysicalValue(torch.tensor([1, 1, 2, 2], dtype=torch.int64)),
            "v": PhysicalValue(
                torch.tensor([10.0, 999.0, 777.0, 888.0], dtype=torch.float64),
                valid=torch.tensor([True, False, False, False], dtype=torch.bool),
            ),
        },
        ("k", "v"),
        4,
    )

    result = execute_grouped_aggregate(
        child,
        ("k",),
        (
            AggregateSpec("sum", "v", ("sum_v",)),
            AggregateSpec("avg", "v", ("avg_v",)),
            AggregateSpec("min", "v", ("min_v",)),
            AggregateSpec("max", "v", ("max_v",)),
            AggregateSpec("count", "v", ("count_v",)),
        ),
    )

    assert result.value_named("sum_v").cell(0) == 10.0
    assert result.value_named("avg_v").cell(0) == 10.0
    assert result.value_named("min_v").cell(0) == 10.0
    assert result.value_named("max_v").cell(0) == 10.0
    assert result.value_named("count_v").cell(0) == 1
    assert result.value_named("sum_v").cell(1) is None
    assert result.value_named("avg_v").cell(1) is None
    assert result.value_named("min_v").cell(1) is None
    assert result.value_named("max_v").cell(1) is None
    assert result.value_named("count_v").cell(1) == 0


def test_count_distinct_ignores_invalid_values():
    child = PhysicalTable(
        "input",
        {
            "v": PhysicalValue(
                torch.tensor([1, 2, 2], dtype=torch.int64),
                valid=torch.tensor([True, False, False], dtype=torch.bool),
            )
        },
        ("v",),
        3,
    )

    result = execute_ungrouped_aggregate(child, (AggregateSpec("count", "v", ("count_distinct_v",), distinct=True),))

    assert result.value_named("count_distinct_v").cell(0) == 1
