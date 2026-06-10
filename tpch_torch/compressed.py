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

    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")
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
