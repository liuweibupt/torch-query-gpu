"""Triton atomic-CAS hash join primitives for unique build keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:  # pragma: no cover - exercised in environments with Triton installed.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - explicit runtime error is tested through wrapper.
    triton = None
    tl = None


DEFAULT_PROBE_GROUP_SIZE = 4
DEFAULT_HASH_FUNCTION_COUNT = 2
DEFAULT_BUILD_BLOCK_SIZE = 128
DEFAULT_MAX_PROBES = 128
_MIN_TABLE_CAPACITY = 16


@dataclass(frozen=True)
class TritonHashJoinConfig:
    """GPU hash join configuration matching the first TQP Triton prototype."""

    probe_group_size: int = DEFAULT_PROBE_GROUP_SIZE
    hash_function_count: int = DEFAULT_HASH_FUNCTION_COUNT
    build_block_size: int = DEFAULT_BUILD_BLOCK_SIZE
    max_probes: int = DEFAULT_MAX_PROBES

    def __post_init__(self) -> None:
        if self.probe_group_size != DEFAULT_PROBE_GROUP_SIZE:
            raise ValueError("Triton hash join currently uses a fixed 4-thread probe group")
        if self.hash_function_count != DEFAULT_HASH_FUNCTION_COUNT:
            raise ValueError("Triton hash join currently requires double hashing")
        if self.build_block_size <= 0:
            raise ValueError("build_block_size must be positive")
        if self.max_probes <= 0:
            raise ValueError("max_probes must be positive")


def triton_hash_join_indices(
    left_key: torch.Tensor,
    right_key: torch.Tensor,
    *,
    config: TritonHashJoinConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inner-join row indices for int64 CUDA keys using a Triton hash table.

    The first prototype supports unique build-side keys. Duplicate build keys are
    rejected explicitly because a full SQL multimap hash table needs payload
    chaining or prefix-sum output sizing.
    """

    config = config or TritonHashJoinConfig()
    _validate_inputs(left_key, right_key)
    _reject_duplicate_build_keys(right_key)
    if left_key.numel() == 0 or right_key.numel() == 0:
        return _empty_indices(left_key.device)
    capacity = _next_power_of_two(max(_MIN_TABLE_CAPACITY, int(right_key.numel()) * 2))
    max_probes = min(capacity, config.max_probes)
    states, table_keys, table_values, failures = _allocate_hash_table(right_key, capacity)
    _launch_build_kernel(right_key, states, table_keys, table_values, failures, capacity, max_probes, config)
    if bool(torch.any(failures != 0).cpu().item()):
        raise RuntimeError("Triton hash join build exceeded max_probes; increase table capacity/probes")
    matches = torch.full((left_key.numel(),), -1, dtype=torch.int64, device=left_key.device)
    _launch_probe_kernel(left_key, states, table_keys, table_values, matches, capacity, max_probes, config)
    left_rows = torch.nonzero(matches >= 0, as_tuple=False).flatten().to(dtype=torch.int64)
    return left_rows, matches.index_select(0, left_rows).to(dtype=torch.int64)


def _validate_inputs(left_key: torch.Tensor, right_key: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton hash join requires Triton to be installed")
    if left_key.ndim != 1 or right_key.ndim != 1:
        raise ValueError("Triton hash join keys must be 1-D tensors")
    if left_key.dtype != torch.int64 or right_key.dtype != torch.int64:
        raise TypeError("Triton hash join currently supports int64 keys")
    if not left_key.is_cuda or not right_key.is_cuda:
        raise RuntimeError("Triton hash join requires CUDA tensors; CPU fallback is not provided")
    if left_key.device != right_key.device:
        raise ValueError("Triton hash join keys must be on the same CUDA device")


def _reject_duplicate_build_keys(right_key: torch.Tensor) -> None:
    if torch.unique(right_key).numel() != right_key.numel():
        raise ValueError("Triton hash join currently requires unique build keys")


def _empty_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    return empty, empty


def _allocate_hash_table(
    right_key: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.zeros((capacity,), dtype=torch.int32, device=right_key.device)
    table_keys = torch.empty((capacity,), dtype=torch.int64, device=right_key.device)
    table_values = torch.full((capacity,), -1, dtype=torch.int64, device=right_key.device)
    failures = torch.zeros((right_key.numel(),), dtype=torch.int32, device=right_key.device)
    return states, table_keys, table_values, failures


def _launch_build_kernel(
    right_key: torch.Tensor,
    states: torch.Tensor,
    table_keys: torch.Tensor,
    table_values: torch.Tensor,
    failures: torch.Tensor,
    capacity: int,
    max_probes: int,
    config: TritonHashJoinConfig,
) -> None:
    grid = (triton.cdiv(right_key.numel(), config.build_block_size),)
    _hash_join_build_kernel[grid](
        right_key,
        states,
        table_keys,
        table_values,
        failures,
        right_key.numel(),
        capacity,
        BLOCK_SIZE=config.build_block_size,
        MAX_PROBES=max_probes,
    )


def _launch_probe_kernel(
    left_key: torch.Tensor,
    states: torch.Tensor,
    table_keys: torch.Tensor,
    table_values: torch.Tensor,
    matches: torch.Tensor,
    capacity: int,
    max_probes: int,
    config: TritonHashJoinConfig,
) -> None:
    _hash_join_probe_kernel[(left_key.numel(),)](
        left_key,
        states,
        table_keys,
        table_values,
        matches,
        left_key.numel(),
        capacity,
        GROUP_SIZE=config.probe_group_size,
        MAX_PROBES=max_probes,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _mix64(x):
        x = x.to(tl.uint64)
        x = x ^ (x >> 33)
        x = x * 0xff51afd7ed558ccd
        x = x ^ (x >> 33)
        x = x * 0xc4ceb9fe1a85ec53
        x = x ^ (x >> 33)
        return x

    @triton.jit
    def _hash_join_build_kernel(
        build_keys,
        states,
        table_keys,
        table_values,
        failures,
        n_elements: tl.constexpr,
        capacity: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_PROBES: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        keys = tl.load(build_keys + offsets, mask=valid, other=0)
        mixed_a = _mix64(keys)
        mixed_b = _mix64(keys ^ 0x9e3779b97f4a7c15)
        mask = capacity - 1
        slot = mixed_a & mask
        step = (mixed_b | 1) & mask
        inserted = tl.full((BLOCK_SIZE,), False, tl.int1)
        for probe in tl.static_range(0, MAX_PROBES):
            candidate = (slot + probe * step) & mask
            active = valid & ~inserted
            old = tl.atomic_cas(states + candidate, 0, 1, sem="relaxed", mask=active)
            won = active & (old == 0)
            tl.store(table_keys + candidate, keys, mask=won)
            tl.store(table_values + candidate, offsets, mask=won)
            inserted = inserted | won
        tl.store(failures + offsets, 1, mask=valid & ~inserted)

    @triton.jit
    def _hash_join_probe_kernel(
        probe_keys,
        states,
        table_keys,
        table_values,
        matches,
        n_elements: tl.constexpr,
        capacity: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        MAX_PROBES: tl.constexpr,
    ):
        key_index = tl.program_id(0)
        lanes = tl.arange(0, GROUP_SIZE)
        key = tl.load(probe_keys + key_index)
        mixed_a = _mix64(key)
        mixed_b = _mix64(key ^ 0x9e3779b97f4a7c15)
        mask = capacity - 1
        slot = mixed_a & mask
        step = (mixed_b | 1) & mask
        found_value = tl.full((), -1, tl.int64)
        active = key_index < n_elements
        for base_probe in tl.static_range(0, MAX_PROBES, GROUP_SIZE):
            candidate = (slot + (base_probe + lanes) * step) & mask
            occupied = tl.load(states + candidate, mask=active, other=0)
            stored_keys = tl.load(table_keys + candidate, mask=active & (occupied != 0), other=0)
            stored_values = tl.load(table_values + candidate, mask=active & (occupied != 0), other=-1)
            hit_values = tl.where((occupied != 0) & (stored_keys == key), stored_values, -1)
            group_hit = tl.max(hit_values, axis=0)
            found_value = tl.where((found_value < 0) & (group_hit >= 0), group_hit, found_value)
        tl.store(matches + key_index, found_value, mask=active)
