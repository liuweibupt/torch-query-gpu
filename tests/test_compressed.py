import pytest
import torch

from tpch_torch.compressed import (
    RLERanges,
    complement_rle,
    plain_to_rle,
    range_intersect,
    range_union,
    rle_to_index,
)


def test_plain_to_rle_encodes_true_runs_from_boolean_mask():
    mask = torch.tensor([False, True, True, False, True, False, True, True])

    ranges = plain_to_rle(mask)

    assert ranges.starts.tolist() == [1, 4, 6]
    assert ranges.ends.tolist() == [2, 4, 7]


def test_plain_to_rle_returns_empty_ranges_for_empty_or_all_false_masks():
    empty = plain_to_rle(torch.tensor([], dtype=torch.bool))
    all_false = plain_to_rle(torch.tensor([False, False], dtype=torch.bool))

    assert empty.starts.numel() == 0
    assert all_false.ends.numel() == 0


def test_rle_to_index_expands_inclusive_ranges():
    ranges = RLERanges(
        starts=torch.tensor([2, 6], dtype=torch.int64),
        ends=torch.tensor([3, 8], dtype=torch.int64),
    )

    assert rle_to_index(ranges).tolist() == [2, 3, 6, 7, 8]


def test_range_intersect_returns_overlapping_segments():
    left = RLERanges(starts=torch.tensor([2]), ends=torch.tensor([7]))
    right = RLERanges(starts=torch.tensor([1, 4, 6]), ends=torch.tensor([3, 5, 8]))

    result = range_intersect(left, right)

    assert result.starts.tolist() == [2, 4, 6]
    assert result.ends.tolist() == [3, 5, 7]


def test_range_intersect_handles_one_range_overlapping_multiple_ranges():
    left = RLERanges(starts=torch.tensor([1, 6]), ends=torch.tensor([4, 10]))
    right = RLERanges(starts=torch.tensor([3]), ends=torch.tensor([8]))

    result = range_intersect(left, right)

    assert result.starts.tolist() == [3, 6]
    assert result.ends.tolist() == [4, 8]


def test_range_union_merges_overlapping_and_adjacent_segments():
    left = RLERanges(starts=torch.tensor([1, 8]), ends=torch.tensor([3, 9]))
    right = RLERanges(starts=torch.tensor([4, 6, 11]), ends=torch.tensor([5, 7, 11]))

    result = range_union(left, right)

    assert result.starts.tolist() == [1, 11]
    assert result.ends.tolist() == [9, 11]


def test_complement_rle_returns_unselected_ranges_within_row_count():
    ranges = RLERanges(starts=torch.tensor([2, 5]), ends=torch.tensor([3, 6]))

    result = complement_rle(ranges, row_count=8)

    assert result.starts.tolist() == [0, 4, 7]
    assert result.ends.tolist() == [1, 4, 7]


def test_complement_rle_handles_empty_and_full_ranges():
    device = torch.device("cpu")
    empty = RLERanges.empty(device=device)
    full = RLERanges(starts=torch.tensor([0]), ends=torch.tensor([2]))

    assert complement_rle(empty, row_count=3).starts.tolist() == [0]
    assert complement_rle(empty, row_count=3).ends.tolist() == [2]
    assert complement_rle(full, row_count=3).starts.numel() == 0


def test_rle_ranges_reject_malformed_ranges():
    with pytest.raises(ValueError, match="same length"):
        RLERanges(starts=torch.tensor([0, 2]), ends=torch.tensor([1]))
    with pytest.raises(ValueError, match="start <= end"):
        RLERanges(starts=torch.tensor([2]), ends=torch.tensor([1]))
    with pytest.raises(ValueError, match="sorted"):
        RLERanges(starts=torch.tensor([3, 1]), ends=torch.tensor([4, 2]))
    with pytest.raises(ValueError, match="non-overlapping"):
        RLERanges(starts=torch.tensor([1, 2]), ends=torch.tensor([3, 4]))


def test_compressed_helpers_reject_invalid_inputs():
    with pytest.raises(TypeError, match="boolean"):
        plain_to_rle(torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="row_count"):
        complement_rle(RLERanges.empty(device=torch.device("cpu")), row_count=-1)
