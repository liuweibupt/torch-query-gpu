import pytest
import torch

from tpch_torch.operators import (
    gather_by_mask,
    grouped_max,
    grouped_mean,
    grouped_min,
    logical_and_all,
    logical_or_all,
    membership_mask,
    topk_indices,
)


def test_logical_and_all_combines_boolean_masks_without_mutating_inputs():
    first = torch.tensor([True, True, False, True])
    second = torch.tensor([True, False, True, True])
    third = torch.tensor([False, True, True, True])

    result = logical_and_all((first, second, third))

    assert result.tolist() == [False, False, False, True]
    assert first.tolist() == [True, True, False, True]


def test_logical_or_all_combines_boolean_masks_without_mutating_inputs():
    first = torch.tensor([False, False, False])
    second = torch.tensor([False, True, False])

    result = logical_or_all((first, second))

    assert result.tolist() == [False, True, False]
    assert first.tolist() == [False, False, False]


def test_logical_mask_helpers_reject_empty_or_non_boolean_masks():
    with pytest.raises(ValueError, match="at least one mask"):
        logical_and_all(())
    with pytest.raises(TypeError, match="boolean"):
        logical_or_all((torch.tensor([1, 0]),))


def test_membership_mask_uses_equality_for_singleton_sets(monkeypatch):
    def fail_isin(*_args, **_kwargs):
        raise AssertionError("singleton membership should not call torch.isin")

    monkeypatch.setattr(torch, "isin", fail_isin)

    result = membership_mask(torch.tensor([1, 2, 1, 3]), (1,))

    assert result.tolist() == [True, False, True, False]


def test_gather_by_mask_selects_rows_from_first_dimension():
    values = torch.tensor([[1, 10], [2, 20], [3, 30], [4, 40]])
    mask = torch.tensor([True, False, True, False])

    result = gather_by_mask(values, mask)

    assert result.tolist() == [[1, 10], [3, 30]]


def test_gather_by_mask_requires_boolean_mask_matching_first_dimension():
    with pytest.raises(TypeError, match="boolean"):
        gather_by_mask(torch.tensor([1, 2]), torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="first dimension"):
        gather_by_mask(torch.tensor([[1], [2]]), torch.tensor([True]))


def test_grouped_min_max_and_mean_reduce_by_group_ids():
    values = torch.tensor([5.0, 2.0, 7.0, 3.0, 11.0])
    group_ids = torch.tensor([0, 1, 0, 1, 2])

    assert grouped_min(values, group_ids, 3).tolist() == [5.0, 2.0, 11.0]
    assert grouped_max(values, group_ids, 3).tolist() == [7.0, 3.0, 11.0]
    assert grouped_mean(values, group_ids, 3).tolist() == [6.0, 2.5, 11.0]


def test_grouped_min_max_mean_raise_for_empty_groups():
    values = torch.tensor([5.0, 2.0])
    group_ids = torch.tensor([0, 2])

    with pytest.raises(ValueError, match="empty group"):
        grouped_min(values, group_ids, 3)
    with pytest.raises(ValueError, match="empty group"):
        grouped_max(values, group_ids, 3)
    with pytest.raises(ValueError, match="empty group"):
        grouped_mean(values, group_ids, 3)


def test_grouped_reductions_validate_group_ids():
    with pytest.raises(ValueError, match="same length"):
        grouped_min(torch.tensor([1.0, 2.0]), torch.tensor([0]), 1)
    with pytest.raises(ValueError, match="out of range"):
        grouped_max(torch.tensor([1.0]), torch.tensor([2]), 2)


def test_topk_indices_returns_indices_in_value_order():
    values = torch.tensor([4.0, 9.0, 1.0, 7.0])

    assert topk_indices(values, 2, descending=True).tolist() == [1, 3]
    assert topk_indices(values, 2, descending=False).tolist() == [2, 0]


def test_topk_indices_rejects_invalid_k_and_non_vector_values():
    with pytest.raises(ValueError, match="non-negative"):
        topk_indices(torch.tensor([1.0]), -1)
    with pytest.raises(ValueError, match="cannot exceed"):
        topk_indices(torch.tensor([1.0]), 2)
    with pytest.raises(ValueError, match="1-D"):
        topk_indices(torch.tensor([[1.0]]), 1)


def test_low_cardinality_group_ids_encode_dense_composite_keys():
    from tpch_torch.operators import low_cardinality_group_ids

    group_ids, group_count = low_cardinality_group_ids(
        (torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 1, 2])),
        (2, 3),
    )

    assert group_count == 6
    assert group_ids.tolist() == [1, 2, 4, 5]


def test_grouped_sum_and_count_bincount_reduce_dense_ids():
    from tpch_torch.operators import grouped_count_bincount, grouped_sum_bincount

    group_ids = torch.tensor([1, 1, 2, 5])
    values = torch.tensor([1.5, 2.5, 10.0, 3.0])

    assert grouped_sum_bincount(values, group_ids, 6).tolist() == [0.0, 4.0, 10.0, 0.0, 0.0, 3.0]
    assert grouped_count_bincount(group_ids, 6).tolist() == [0, 2, 1, 0, 0, 1]


def test_grouped_sum_bincount_preserves_integer_value_dtype():
    from tpch_torch.operators import grouped_sum_bincount

    result = grouped_sum_bincount(
        torch.tensor([2, 3, 5], dtype=torch.int64),
        torch.tensor([0, 0, 1], dtype=torch.int64),
        3,
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [5, 5, 0]
