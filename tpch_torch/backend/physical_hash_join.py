"""First tensor dictionary/probe join prototype for physical values."""

from __future__ import annotations

import torch

from tpch_torch.backend.physical_key_ops import comparable_value_tensors
from tpch_torch.backend.physical_join import inner_join_indices
from tpch_torch.backend.physical_types import PhysicalValue


def hash_join_indices_for_values(
    left_value: PhysicalValue,
    right_value: PhysicalValue,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equi-join indices through a tensor dictionary-encoded probe API."""

    left_key, right_key = comparable_value_tensors(left_value, right_value)
    left_ids, right_ids = _dictionary_encode_pair(left_key, right_key)
    return inner_join_indices(left_ids, right_ids)


def _dictionary_encode_pair(
    left_key: torch.Tensor,
    right_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left_key.ndim != 1 or right_key.ndim != 1:
        raise ValueError("hash join keys must be 1-D tensors")
    if left_key.device != right_key.device:
        raise ValueError("hash join keys must be on the same device")
    combined = torch.cat((left_key, right_key))
    if combined.numel() == 0:
        return left_key.to(dtype=torch.int64), right_key.to(dtype=torch.int64)
    _, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    left_count = left_key.numel()
    return inverse[:left_count].to(dtype=torch.int64), inverse[left_count:].to(dtype=torch.int64)
