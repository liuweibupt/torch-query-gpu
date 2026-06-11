"""Tensor primitives for lightweight-compressed relational execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@dataclass(frozen=True)
class RLERanges:
    """Inclusive RLE mask ranges represented by sorted start/end tensors."""

    starts: torch.Tensor
    ends: torch.Tensor

    def __post_init__(self) -> None:
        _validate_range_tensor("starts", self.starts)
        _validate_range_tensor("ends", self.ends)
        if self.starts.shape != self.ends.shape:
            raise ValueError("RLE starts and ends must have the same length")
        if self.starts.device != self.ends.device:
            raise ValueError("RLE starts and ends must be on the same device")
        if self.starts.numel() == 0:
            return
        if bool(torch.any(self.starts > self.ends).cpu().item()):
            raise ValueError("RLE ranges require start <= end")
        if self.starts.numel() == 1:
            return
        if bool(torch.any(self.starts[1:] <= self.starts[:-1]).cpu().item()):
            raise ValueError("RLE ranges must be sorted by start position")
        if bool(torch.any(self.starts[1:] <= self.ends[:-1]).cpu().item()):
            raise ValueError("RLE ranges must be non-overlapping")

    @classmethod
    def empty(cls, device: torch.device | str) -> RLERanges:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return cls(starts=empty, ends=empty.clone())

    @property
    def device(self) -> torch.device:
        return self.starts.device

    @property
    def lengths(self) -> torch.Tensor:
        return self.ends - self.starts + 1




@dataclass(frozen=True)
class PlainMask:
    """Plain boolean mask column."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.dtype is not torch.bool:
            raise TypeError("PlainMask values must be boolean")
        if self.values.ndim != 1:
            raise ValueError("PlainMask values must be 1-D")

    @property
    def row_count(self) -> int:
        return int(self.values.numel())

    @property
    def device(self) -> torch.device:
        return self.values.device


@dataclass(frozen=True)
class RLEMask:
    """RLE encoded boolean mask column."""

    ranges: RLERanges
    row_count: int

    def __post_init__(self) -> None:
        _validate_row_count(self.row_count)
        if self.ranges.ends.numel() and bool(torch.any(self.ranges.ends >= self.row_count).cpu().item()):
            raise ValueError("RLEMask ranges cannot extend past row_count")

    @property
    def device(self) -> torch.device:
        return self.ranges.device


@dataclass(frozen=True)
class IndexMask:
    """Index encoded boolean mask column with sorted selected positions."""

    positions: torch.Tensor
    row_count: int

    def __post_init__(self) -> None:
        _validate_row_count(self.row_count)
        _validate_index_positions(self.positions, name="mask")
        if self.positions.numel() and bool(torch.any(self.positions >= self.row_count).cpu().item()):
            raise ValueError("IndexMask positions cannot extend past row_count")

    @property
    def device(self) -> torch.device:
        return self.positions.device


MaskColumn = PlainMask | RLEMask | IndexMask

def plain_to_rle(mask: torch.Tensor) -> RLERanges:
    """Encode a 1-D boolean mask as inclusive RLE ranges of true values."""

    if mask.dtype is not torch.bool:
        raise TypeError("plain_to_rle expects a boolean mask")
    if mask.ndim != 1:
        raise ValueError("plain_to_rle expects a 1-D mask")
    if mask.numel() == 0:
        return RLERanges.empty(mask.device)
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.int8, device=mask.device),
            mask.to(dtype=torch.int8),
            torch.zeros(1, dtype=torch.int8, device=mask.device),
        )
    )
    transitions = padded[1:] - padded[:-1]
    starts = torch.nonzero(transitions == 1).flatten().to(dtype=torch.int64)
    ends = (torch.nonzero(transitions == -1).flatten() - 1).to(dtype=torch.int64)
    return RLERanges(starts=starts, ends=ends)



def rle_to_index(ranges: RLERanges) -> torch.Tensor:
    """Expand inclusive RLE ranges into explicit sorted positions."""

    if ranges.starts.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=ranges.device)
    repeated_starts = torch.repeat_interleave(ranges.starts, ranges.lengths)
    offsets = torch.arange(int(ranges.lengths.sum().cpu().item()), device=ranges.device, dtype=torch.int64)
    range_offsets = torch.repeat_interleave(torch.cumsum(ranges.lengths, dim=0) - ranges.lengths, ranges.lengths)
    return repeated_starts + offsets - range_offsets



def rle_to_plain(ranges: RLERanges, row_count: int) -> torch.Tensor:
    """Materialize inclusive RLE mask ranges as a plain boolean mask."""

    _validate_row_count(row_count)
    if ranges.starts.numel() and bool(torch.any(ranges.ends >= row_count).cpu().item()):
        raise ValueError("RLE ranges cannot extend past row_count")
    mask = torch.zeros(row_count, dtype=torch.bool, device=ranges.device)
    if ranges.starts.numel() == 0:
        return mask
    mask[rle_to_index(ranges)] = True
    return mask


def range_intersect(left: RLERanges, right: RLERanges) -> RLERanges:
    """Return pairwise overlaps between two sorted inclusive RLE range sets."""

    _validate_compatible_ranges(left, right)
    if left.starts.numel() == 0 or right.starts.numel() == 0:
        return RLERanges.empty(left.device)
    if left.starts.numel() <= right.starts.numel():
        return _range_intersect_ordered(left, right)
    return _range_intersect_ordered(right, left)



def range_union(left: RLERanges, right: RLERanges) -> RLERanges:
    """Return the normalized union of two sorted inclusive RLE range sets."""

    _validate_compatible_ranges(left, right)
    if left.starts.numel() == 0:
        return right
    if right.starts.numel() == 0:
        return left
    starts = torch.cat((left.starts, right.starts))
    ends = torch.cat((left.ends, right.ends))
    order = torch.argsort(starts)
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    current_start = int(sorted_starts[0].cpu().item())
    current_end = int(sorted_ends[0].cpu().item())
    merged_starts: list[int] = []
    merged_ends: list[int] = []
    for index in range(1, int(sorted_starts.numel())):
        start = int(sorted_starts[index].cpu().item())
        end = int(sorted_ends[index].cpu().item())
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        merged_starts.append(current_start)
        merged_ends.append(current_end)
        current_start = start
        current_end = end
    merged_starts.append(current_start)
    merged_ends.append(current_end)
    return RLERanges(
        starts=torch.tensor(merged_starts, dtype=torch.int64, device=left.device),
        ends=torch.tensor(merged_ends, dtype=torch.int64, device=left.device),
    )



def complement_rle(ranges: RLERanges, row_count: int) -> RLERanges:
    """Return ranges not selected by `ranges` inside `[0, row_count)`."""

    _validate_row_count(row_count)
    if ranges.starts.numel() == 0:
        if row_count == 0:
            return RLERanges.empty(ranges.device)
        return RLERanges(
            starts=torch.tensor([0], dtype=torch.int64, device=ranges.device),
            ends=torch.tensor([row_count - 1], dtype=torch.int64, device=ranges.device),
        )
    if bool(torch.any(ranges.ends >= row_count).cpu().item()):
        raise ValueError("RLE ranges cannot extend past row_count")
    complement_starts: list[int] = []
    complement_ends: list[int] = []
    next_start = 0
    for start_tensor, end_tensor in zip(ranges.starts, ranges.ends):
        start = int(start_tensor.cpu().item())
        end = int(end_tensor.cpu().item())
        if next_start < start:
            complement_starts.append(next_start)
            complement_ends.append(start - 1)
        next_start = end + 1
    if next_start < row_count:
        complement_starts.append(next_start)
        complement_ends.append(row_count - 1)
    return RLERanges(
        starts=torch.tensor(complement_starts, dtype=torch.int64, device=ranges.device),
        ends=torch.tensor(complement_ends, dtype=torch.int64, device=ranges.device),
    )


def idx_in_rle(positions: torch.Tensor, ranges: RLERanges) -> torch.Tensor:
    """Return sorted index positions contained within any RLE range."""

    _validate_index_positions(positions, name="positions")
    if positions.numel() == 0 or ranges.starts.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=positions.device)
    _validate_compatible_index_and_ranges(positions, ranges)
    bins = torch.searchsorted(ranges.starts, positions, right=True) - 1
    valid_bins = bins >= 0
    safe_bins = torch.clamp(bins, min=0)
    selected = valid_bins & (positions <= ranges.ends[safe_bins])
    return positions[selected]


def rle_contain_idx(positions: torch.Tensor, ranges: RLERanges) -> RLERanges:
    """Return ranges that contain at least one index position."""

    _validate_index_positions(positions, name="positions")
    if positions.numel() == 0 or ranges.starts.numel() == 0:
        return RLERanges.empty(ranges.device)
    _validate_compatible_index_and_ranges(positions, ranges)
    bins_start = torch.searchsorted(positions, ranges.starts, right=False)
    bins_end = torch.searchsorted(positions, ranges.ends, right=True)
    selected = bins_end > bins_start
    return RLERanges(starts=ranges.starts[selected], ends=ranges.ends[selected])


def idx_in_idx(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return sorted positions present in both sorted index tensors."""

    _validate_index_positions(left, name="left")
    _validate_index_positions(right, name="right")
    _validate_compatible_indices(left, right)
    if left.numel() == 0 or right.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=left.device)
    positions = torch.searchsorted(right, left)
    in_bounds = positions < right.numel()
    safe_positions = torch.clamp(positions, max=max(int(right.numel()) - 1, 0))
    matched = in_bounds & (right[safe_positions] == left)
    return left[matched]


def merge_sorted_idx(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the sorted union of two sorted index tensors."""

    _validate_index_positions(left, name="left")
    _validate_index_positions(right, name="right")
    _validate_compatible_indices(left, right)
    if left.numel() == 0:
        return right
    if right.numel() == 0:
        return left
    return torch.unique(torch.cat((left, right)), sorted=True)


def complement_index(positions: torch.Tensor, row_count: int) -> RLERanges:
    """Return the complement of sorted index positions as RLE ranges."""

    _validate_row_count(row_count)
    _validate_index_positions(positions, name="positions")
    if positions.numel() and bool(torch.any(positions >= row_count).cpu().item()):
        raise ValueError("index positions cannot extend past row_count")
    selected = torch.zeros(row_count, dtype=torch.bool, device=positions.device)
    if positions.numel():
        selected[positions] = True
    return plain_to_rle(torch.logical_not(selected))




def mask_to_plain(mask: MaskColumn) -> torch.Tensor:
    """Materialize any encoded mask as a plain boolean tensor."""

    if isinstance(mask, PlainMask):
        return mask.values.clone()
    if isinstance(mask, RLEMask):
        return rle_to_plain(mask.ranges, mask.row_count)
    if isinstance(mask, IndexMask):
        values = torch.zeros(mask.row_count, dtype=torch.bool, device=mask.device)
        if mask.positions.numel():
            values[mask.positions] = True
        return values
    raise TypeError(f"unsupported mask type: {type(mask).__name__}")


def mask_to_index(mask: MaskColumn) -> torch.Tensor:
    """Materialize any encoded mask as sorted selected row positions."""

    if isinstance(mask, PlainMask):
        return torch.nonzero(mask.values).flatten().to(dtype=torch.int64)
    if isinstance(mask, RLEMask):
        return rle_to_index(mask.ranges)
    if isinstance(mask, IndexMask):
        return mask.positions.clone()
    raise TypeError(f"unsupported mask type: {type(mask).__name__}")


def mask_and(left: MaskColumn, right: MaskColumn) -> MaskColumn:
    """Return encoded logical AND for two compatible masks."""

    _validate_compatible_masks(left, right)
    if isinstance(left, PlainMask) and isinstance(right, PlainMask):
        return PlainMask(torch.logical_and(left.values, right.values))
    if isinstance(left, RLEMask) and isinstance(right, RLEMask):
        return RLEMask(range_intersect(left.ranges, right.ranges), left.row_count)
    if isinstance(left, IndexMask) and isinstance(right, IndexMask):
        return IndexMask(idx_in_idx(left.positions, right.positions), left.row_count)
    return _mask_and_mixed(left, right)


def mask_or(left: MaskColumn, right: MaskColumn) -> MaskColumn:
    """Return encoded logical OR for two compatible masks."""

    _validate_compatible_masks(left, right)
    if isinstance(left, PlainMask) and isinstance(right, PlainMask):
        return PlainMask(torch.logical_or(left.values, right.values))
    if isinstance(left, RLEMask) and isinstance(right, RLEMask):
        return RLEMask(range_union(left.ranges, right.ranges), left.row_count)
    if isinstance(left, IndexMask) and isinstance(right, IndexMask):
        return IndexMask(merge_sorted_idx(left.positions, right.positions), left.row_count)
    return _mask_or_mixed(left, right)


def mask_not(mask: MaskColumn) -> MaskColumn:
    """Return encoded logical NOT for a mask."""

    if isinstance(mask, PlainMask):
        return PlainMask(torch.logical_not(mask.values))
    if isinstance(mask, RLEMask):
        return RLEMask(complement_rle(mask.ranges, mask.row_count), mask.row_count)
    if isinstance(mask, IndexMask):
        return RLEMask(complement_index(mask.positions, mask.row_count), mask.row_count)
    raise TypeError(f"unsupported mask type: {type(mask).__name__}")


def _mask_and_mixed(left: MaskColumn, right: MaskColumn) -> MaskColumn:
    if isinstance(left, IndexMask) and isinstance(right, RLEMask):
        return IndexMask(idx_in_rle(left.positions, right.ranges), left.row_count)
    if isinstance(left, RLEMask) and isinstance(right, IndexMask):
        return IndexMask(idx_in_rle(right.positions, left.ranges), left.row_count)
    if isinstance(left, PlainMask) and isinstance(right, IndexMask):
        return IndexMask(right.positions[left.values[right.positions]], left.row_count)
    if isinstance(left, IndexMask) and isinstance(right, PlainMask):
        return IndexMask(left.positions[right.values[left.positions]], left.row_count)
    if isinstance(left, PlainMask) and isinstance(right, RLEMask):
        return IndexMask(idx_in_rle(mask_to_index(left), right.ranges), left.row_count)
    if isinstance(left, RLEMask) and isinstance(right, PlainMask):
        return IndexMask(idx_in_rle(mask_to_index(right), left.ranges), left.row_count)
    raise TypeError(f"unsupported mask types: {type(left).__name__}, {type(right).__name__}")


def _mask_or_mixed(left: MaskColumn, right: MaskColumn) -> MaskColumn:
    if isinstance(left, PlainMask):
        values = left.values.clone()
        values[mask_to_index(right)] = True
        return PlainMask(values)
    if isinstance(right, PlainMask):
        values = right.values.clone()
        values[mask_to_index(left)] = True
        return PlainMask(values)
    if isinstance(left, RLEMask) and isinstance(right, IndexMask):
        return IndexMask(merge_sorted_idx(rle_to_index(left.ranges), right.positions), left.row_count)
    if isinstance(left, IndexMask) and isinstance(right, RLEMask):
        return IndexMask(merge_sorted_idx(left.positions, rle_to_index(right.ranges)), left.row_count)
    raise TypeError(f"unsupported mask types: {type(left).__name__}, {type(right).__name__}")


def _validate_compatible_masks(left: MaskColumn, right: MaskColumn) -> None:
    if left.row_count != right.row_count:
        raise ValueError("mask row_count must match")
    if left.device != right.device:
        raise ValueError("masks must be on the same device")

def _validate_range_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"RLE {name} must be a 1-D tensor")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"RLE {name} must use an integer dtype")
    if tensor.numel() == 0:
        return
    if bool(torch.any(tensor < 0).cpu().item()):
        raise ValueError(f"RLE {name} must be non-negative")



def _validate_compatible_ranges(left: RLERanges, right: RLERanges) -> None:
    if left.device != right.device:
        raise ValueError("RLE ranges must be on the same device")


def _range_intersect_ordered(smaller: RLERanges, larger: RLERanges) -> RLERanges:
    bins = torch.searchsorted(larger.ends, smaller.starts, right=False)
    bine = torch.searchsorted(larger.starts, smaller.ends, right=True)
    counts = torch.clamp(bine - bins, min=0)
    if int(counts.sum().cpu().item()) == 0:
        return RLERanges.empty(smaller.device)
    smaller_idx = torch.repeat_interleave(
        torch.arange(smaller.starts.numel(), dtype=torch.int64, device=smaller.device), counts
    )
    larger_idx = _range_arange(bins, counts)
    starts = torch.maximum(smaller.starts[smaller_idx], larger.starts[larger_idx])
    ends = torch.minimum(smaller.ends[smaller_idx], larger.ends[larger_idx])
    overlap = starts <= ends
    return RLERanges(starts=starts[overlap], ends=ends[overlap])


def _range_arange(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    total_size = int(lengths.sum().cpu().item())
    if total_size == 0:
        return torch.empty(0, dtype=torch.int64, device=starts.device)
    offsets = torch.cumsum(lengths, dim=0) - lengths
    return (
        torch.repeat_interleave(starts, lengths)
        + torch.arange(total_size, dtype=torch.int64, device=starts.device)
        - torch.repeat_interleave(offsets, lengths)
    )


def _validate_row_count(row_count: int) -> None:
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")


def _validate_index_positions(positions: torch.Tensor, name: str) -> None:
    if positions.ndim != 1:
        raise ValueError(f"{name} index positions must be a 1-D tensor")
    if positions.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} index positions must use an integer dtype")
    if positions.numel() == 0:
        return
    if bool(torch.any(positions < 0).cpu().item()):
        raise ValueError(f"{name} index positions must be non-negative")
    if positions.numel() == 1:
        return
    if bool(torch.any(positions[1:] <= positions[:-1]).cpu().item()):
        raise ValueError(f"{name} index positions must be sorted and unique")


def _validate_compatible_indices(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.device != right.device:
        raise ValueError("index position tensors must be on the same device")


def _validate_compatible_index_and_ranges(positions: torch.Tensor, ranges: RLERanges) -> None:
    if positions.device != ranges.device:
        raise ValueError("index positions and RLE ranges must be on the same device")
