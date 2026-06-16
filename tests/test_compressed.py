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


def test_idx_in_rle_returns_index_positions_inside_ranges():
    from tpch_torch.compressed import idx_in_rle

    positions = torch.tensor([2, 4, 7, 9], dtype=torch.int64)
    ranges = RLERanges(starts=torch.tensor([0, 6]), ends=torch.tensor([2, 7]))

    assert idx_in_rle(positions, ranges).tolist() == [2, 7]


def test_rle_contain_idx_returns_ranges_that_contain_positions():
    from tpch_torch.compressed import rle_contain_idx

    positions = torch.tensor([2, 4, 7], dtype=torch.int64)
    ranges = RLERanges(starts=torch.tensor([0, 6, 10]), ends=torch.tensor([2, 7, 12]))

    result = rle_contain_idx(positions, ranges)

    assert result.starts.tolist() == [0, 6]
    assert result.ends.tolist() == [2, 7]


def test_idx_in_idx_intersects_sorted_index_positions():
    from tpch_torch.compressed import idx_in_idx

    left = torch.tensor([1, 2, 4, 7], dtype=torch.int64)
    right = torch.tensor([2, 3, 7, 9], dtype=torch.int64)

    assert idx_in_idx(left, right).tolist() == [2, 7]


def test_merge_sorted_idx_unions_sorted_index_positions():
    from tpch_torch.compressed import merge_sorted_idx

    left = torch.tensor([1, 2, 7], dtype=torch.int64)
    right = torch.tensor([2, 3, 7, 9], dtype=torch.int64)

    assert merge_sorted_idx(left, right).tolist() == [1, 2, 3, 7, 9]


def test_complement_index_returns_rle_ranges():
    from tpch_torch.compressed import complement_index

    positions = torch.tensor([1, 3, 4], dtype=torch.int64)

    result = complement_index(positions, row_count=6)

    assert result.starts.tolist() == [0, 2, 5]
    assert result.ends.tolist() == [0, 2, 5]


def test_rle_to_plain_materializes_boolean_mask():
    from tpch_torch.compressed import rle_to_plain

    ranges = RLERanges(starts=torch.tensor([1, 4]), ends=torch.tensor([2, 4]))

    assert rle_to_plain(ranges, row_count=6).tolist() == [False, True, True, False, True, False]


def test_index_helpers_reject_unsorted_or_out_of_bounds_inputs():
    from tpch_torch.compressed import complement_index, idx_in_idx, rle_to_plain

    with pytest.raises(ValueError, match="sorted"):
        idx_in_idx(torch.tensor([2, 1]), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="row_count"):
        complement_index(torch.tensor([3]), row_count=3)
    with pytest.raises(ValueError, match="row_count"):
        rle_to_plain(RLERanges(starts=torch.tensor([0]), ends=torch.tensor([2])), row_count=2)


def test_mask_column_dispatch_matches_plain_boolean_logic():
    from tpch_torch.compressed import IndexMask, RLEMask, mask_and, mask_not, mask_or, mask_to_plain

    left = RLEMask(plain_to_rle(torch.tensor([False, True, True, False, True])), row_count=5)
    right = IndexMask(torch.tensor([2, 3, 4], dtype=torch.int64), row_count=5)

    assert mask_to_plain(mask_and(left, right)).tolist() == [False, False, True, False, True]
    assert mask_to_plain(mask_or(left, right)).tolist() == [False, True, True, True, True]
    assert mask_to_plain(mask_not(right)).tolist() == [True, True, False, False, False]


def test_mask_column_dispatch_rejects_incompatible_masks():
    from tpch_torch.compressed import IndexMask, PlainMask, mask_and

    left = PlainMask(torch.tensor([True, False]))
    right = IndexMask(torch.tensor([0], dtype=torch.int64), row_count=3)

    with pytest.raises(ValueError, match="row_count"):
        mask_and(left, right)


def test_range_arange_generates_segmented_offsets():
    from tpch_torch.compressed import range_arange

    starts = torch.tensor([2, 10, 20], dtype=torch.int64)
    lengths = torch.tensor([3, 0, 2], dtype=torch.int64)

    result = range_arange(starts, lengths)

    assert result.tolist() == [2, 3, 4, 20, 21]


def test_compact_rle_removes_gaps_between_selected_runs():
    from tpch_torch.compressed import compact_rle

    ranges = RLERanges(starts=torch.tensor([2, 6, 10]), ends=torch.tensor([3, 8, 10]))

    result = compact_rle(ranges)

    assert result.starts.tolist() == [0, 2, 5]
    assert result.ends.tolist() == [1, 4, 5]


def test_rle_aggregates_use_run_lengths_without_expanding_positions(monkeypatch):
    from tpch_torch.compressed import rle_to_index
    from tpch_torch.compressed_aggregates import rle_count, rle_max, rle_mean, rle_min, rle_sum

    def fail_expand(*_args, **_kwargs):
        raise AssertionError("RLE aggregates should use run lengths, not row expansion")

    monkeypatch.setattr("tpch_torch.compressed.rle_to_index", fail_expand)

    ranges = RLERanges(starts=torch.tensor([2, 6, 10]), ends=torch.tensor([3, 8, 10]))
    values = torch.tensor([5.0, 2.0, 7.0])

    assert rle_count(ranges).item() == 6
    assert rle_sum(values, ranges).item() == 23.0
    assert rle_min(values, ranges).item() == 2.0
    assert rle_max(values, ranges).item() == 7.0
    assert rle_mean(values, ranges).item() == 23.0 / 6.0


def test_rle_aggregates_reject_misaligned_run_values():
    from tpch_torch.compressed_aggregates import rle_sum

    ranges = RLERanges(starts=torch.tensor([0, 3]), ends=torch.tensor([1, 4]))

    with pytest.raises(ValueError, match="one value per RLE run"):
        rle_sum(torch.tensor([1.0]), ranges)
