"""Aggregate primitives over RLE-encoded columns."""

from __future__ import annotations

import torch

from tpch_torch.compressed import RLERanges


def rle_count(ranges: RLERanges) -> torch.Tensor:
    """Count selected rows represented by RLE ranges."""

    return ranges.lengths.sum().to(dtype=torch.int64)


def rle_sum(values: torch.Tensor, ranges: RLERanges) -> torch.Tensor:
    """Sum one run value per RLE range using run lengths as weights."""

    _validate_run_values(values, ranges)
    return (values * ranges.lengths.to(dtype=values.dtype)).sum()


def rle_min(values: torch.Tensor, ranges: RLERanges) -> torch.Tensor:
    """Return the minimum run value."""

    _validate_non_empty_run_values(values, ranges)
    return values.min()


def rle_max(values: torch.Tensor, ranges: RLERanges) -> torch.Tensor:
    """Return the maximum run value."""

    _validate_non_empty_run_values(values, ranges)
    return values.max()


def rle_mean(values: torch.Tensor, ranges: RLERanges) -> torch.Tensor:
    """Return weighted mean over RLE run values."""

    total = rle_count(ranges).to(dtype=torch.float64)
    if int(total.cpu().item()) == 0:
        raise ValueError("RLE mean requires at least one selected row")
    return rle_sum(values.to(dtype=torch.float64), ranges) / total


def _validate_non_empty_run_values(values: torch.Tensor, ranges: RLERanges) -> None:
    _validate_run_values(values, ranges)
    if values.numel() == 0:
        raise ValueError("RLE aggregate requires at least one run")


def _validate_run_values(values: torch.Tensor, ranges: RLERanges) -> None:
    if values.ndim != 1:
        raise ValueError("RLE aggregate values must be a 1-D tensor")
    if values.device != ranges.device:
        raise ValueError("RLE aggregate values and ranges must be on the same device")
    if values.numel() != ranges.starts.numel():
        raise ValueError("RLE aggregate expects one value per RLE run")
    if not torch.is_floating_point(values) and values.dtype not in {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("RLE aggregate values must be numeric")
