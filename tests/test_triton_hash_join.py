import importlib.util
import pathlib
import sysconfig

import pytest
import torch

from tpch_torch.backend.triton_hash_join import (
    TritonHashJoinConfig,
    triton_hash_join_indices,
)
from tpch_torch.record_batch import ColumnMeta


def _can_compile_triton_cuda_driver() -> bool:
    include_dir = sysconfig.get_paths().get("include")
    return include_dir is not None and pathlib.Path(include_dir, "Python.h").exists()


def test_triton_hash_join_config_uses_double_hash_and_four_thread_probe_group():
    config = TritonHashJoinConfig()

    assert config.hash_function_count == 2
    assert config.probe_group_size == 4


def test_triton_hash_join_requires_cuda_inputs_without_cpu_fallback():
    left = torch.tensor([1, 2, 3], dtype=torch.int64)
    right = torch.tensor([2, 3], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="CUDA"):
        triton_hash_join_indices(left, right)


def test_triton_hash_join_for_physical_values_reuses_comparable_key_validation_on_cpu():
    from tpch_torch.backend.physical_hash_join import triton_hash_join_indices_for_values
    from tpch_torch.backend.physical_types import PhysicalValue

    left = PhysicalValue(
        torch.tensor([100, 125, 130], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=2),
    )
    right = PhysicalValue(
        torch.tensor([10, 13], dtype=torch.int64),
        meta=ColumnMeta.decimal(precision=12, scale=1),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        triton_hash_join_indices_for_values(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton hash join")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is not installed")
@pytest.mark.skipif(not _can_compile_triton_cuda_driver(), reason="Python.h is required for Triton CUDA driver JIT")
def test_triton_hash_join_indices_match_unique_build_reference_on_cuda():
    left = torch.tensor([9, 4, 7, 4, 1, 12, 7], dtype=torch.int64, device="cuda")
    right = torch.tensor([7, 9, 1, 4], dtype=torch.int64, device="cuda")

    left_rows, right_rows = triton_hash_join_indices(left, right)

    assert left_rows.cpu().tolist() == [0, 1, 2, 3, 4, 6]
    assert right_rows.cpu().tolist() == [1, 3, 0, 3, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton hash join")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton is not installed")
@pytest.mark.skipif(not _can_compile_triton_cuda_driver(), reason="Python.h is required for Triton CUDA driver JIT")
def test_triton_hash_join_rejects_duplicate_build_keys_on_cuda():
    left = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    right = torch.tensor([2, 2, 3], dtype=torch.int64, device="cuda")

    with pytest.raises(ValueError, match="unique build keys"):
        triton_hash_join_indices(left, right)
