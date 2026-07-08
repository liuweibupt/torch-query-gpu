from decimal import Decimal

import torch

from tpch_torch.backend.physical_expr import evaluate_expression
from tpch_torch.backend.physical_join import try_execute_scalar_nested_loop_join
from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue
from tpch_torch.record_batch import ColumnMeta, LogicalDType


def _decimal_value(values, scale=2, precision=12):
    return PhysicalValue(
        torch.tensor(values, dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=precision, scale=scale),
    )


def test_physical_value_decodes_decimal_cell_and_preserves_meta_on_filter_gather():
    value = _decimal_value([1234, 500], scale=2)

    filtered = value.filter(torch.tensor([True, False]))
    gathered = value.gather(torch.tensor([1, 0], dtype=torch.int64))

    assert value.cell(0) == Decimal("12.34")
    assert filtered.meta == value.meta
    assert gathered.meta == value.meta


def test_decimal_add_subtract_and_multiply_keep_scaled_int64_metadata():
    table = PhysicalTable(
        "t",
        {
            "a": _decimal_value([1234], scale=2),
            "b": _decimal_value([50], scale=1),
        },
        ("a", "b"),
        1,
    )

    added = evaluate_expression(table, "a + b")
    subtracted = evaluate_expression(table, "a - b")
    multiplied = evaluate_expression(table, "a * b")

    assert added.meta.logical_dtype == LogicalDType.DECIMAL
    assert added.meta.scale == 2
    assert added.require_tensor().tolist() == [1734]
    assert subtracted.require_tensor().tolist() == [734]
    assert multiplied.meta.scale == 3
    assert multiplied.require_tensor().tolist() == [61700]


def test_decimal_division_returns_fp64():
    table = PhysicalTable(
        "t",
        {"a": _decimal_value([1000]), "b": _decimal_value([400])},
        ("a", "b"),
        1,
    )

    divided = evaluate_expression(table, "a / b")

    assert divided.meta is None or divided.meta.logical_dtype == LogicalDType.FP64
    assert divided.require_tensor().dtype == torch.float64
    assert divided.cell(0) == 2.5


def test_decimal_comparisons_align_numeric_literals():
    table = PhysicalTable(
        "t",
        {"discount": _decimal_value([4, 5, 7, 8], scale=2)},
        ("discount",),
        4,
    )

    selected = evaluate_expression(table, "discount >= 0.05 AND discount <= 0.07")

    assert selected.require_tensor().tolist() == [False, True, True, False]


def test_decimal_arithmetic_aligns_numeric_literals():
    table = PhysicalTable(
        "t",
        {
            "price": _decimal_value([10000], scale=2),
            "discount": _decimal_value([5], scale=2),
        },
        ("price", "discount"),
        1,
    )

    discounted = evaluate_expression(table, "price * (1.0 - discount)")

    assert discounted.meta.logical_dtype == LogicalDType.DECIMAL
    assert discounted.meta.scale == 4
    assert discounted.require_tensor().tolist() == [950000]
    assert discounted.cell(0) == Decimal("95.0000")


def test_decimal_case_with_numeric_literal_preserves_scale():
    table = PhysicalTable(
        "t",
        {
            "selected": PhysicalValue(torch.tensor([True, False], dtype=torch.bool)),
            "amount": _decimal_value([1234, 5678], scale=2),
        },
        ("selected", "amount"),
        2,
    )

    value = evaluate_expression(table, "case when selected then amount else 0.0 end")

    assert value.meta.logical_dtype == LogicalDType.DECIMAL
    assert value.meta.scale == 2
    assert value.require_tensor().tolist() == [1234, 0]


def test_decimal_comparison_supports_fp64_tensor():
    table = PhysicalTable(
        "t",
        {
            "quantity": _decimal_value([100, 250], scale=2),
            "threshold": PhysicalValue(torch.tensor([1.5, 2.0], dtype=torch.float64)),
        },
        ("quantity", "threshold"),
        2,
    )

    selected = evaluate_expression(table, "quantity > threshold")

    assert selected.require_tensor().tolist() == [False, True]


def test_decimal_comparison_supports_int64_tensor():
    table = PhysicalTable(
        "t",
        {
            "available": PhysicalValue(torch.tensor([10, 2], dtype=torch.int64), meta=ColumnMeta.int64()),
            "threshold": _decimal_value([500, 300], scale=2),
        },
        ("available", "threshold"),
        2,
    )

    selected = evaluate_expression(table, "available > threshold")

    assert selected.require_tensor().tolist() == [True, False]


def test_scalar_subquery_compare_uses_decimal_real_values():
    left = PhysicalTable(
        "customer",
        {"c_acctbal": _decimal_value([10000, 600000], scale=2)},
        ("c_acctbal",),
        2,
    )
    right = PhysicalTable(
        "subquery",
        {"SUBQUERY": PhysicalValue(torch.tensor([5000.0], dtype=torch.float64), meta=ColumnMeta.fp64())},
        ("SUBQUERY",),
        1,
    )

    filtered = try_execute_scalar_nested_loop_join(left, right, "CAST(c_acctbal AS DOUBLE) > SUBQUERY")

    assert filtered is not None
    assert filtered.row_count == 1
    assert filtered.value_named("c_acctbal").require_tensor().tolist() == [600000]
