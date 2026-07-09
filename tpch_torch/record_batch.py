"""Typed tensor record batches used as the columnar metadata substrate."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from tpch_torch.record_batch_storage import ColumnStorage
from tpch_torch.record_batch_types import (
    AllocationOwner,
    ColumnMeta,
    ColumnType,
    LogicalDType,
    StorageKind,
)


@dataclass(frozen=True)
class BatchMeta:
    row_count: int
    chunk_size: int
    chunk_index: int
    source_offset: int
    device: torch.device
    schema_version: int = 2

    def with_row_count(self, row_count: int) -> "BatchMeta":
        return BatchMeta(
            row_count=row_count,
            chunk_size=self.chunk_size,
            chunk_index=self.chunk_index,
            source_offset=self.source_offset,
            device=self.device,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class TensorRecordBatch:
    """Columnar tensor batch plus per-column type, storage, and validity metadata."""

    columns: Mapping[str, torch.Tensor]
    meta: Mapping[str, ColumnMeta]
    validity: Mapping[str, torch.Tensor | None] | None = None
    types: Mapping[str, ColumnType] | None = None
    storage: Mapping[str, ColumnStorage] | None = None
    batch_meta: BatchMeta | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_batch(self)
        object.__setattr__(self, "columns", MappingProxyType(normalized.columns))
        object.__setattr__(self, "meta", MappingProxyType(normalized.meta))
        object.__setattr__(self, "validity", MappingProxyType(normalized.validity))
        object.__setattr__(self, "types", MappingProxyType(normalized.types))
        object.__setattr__(self, "storage", MappingProxyType(normalized.storage))
        object.__setattr__(self, "batch_meta", normalized.batch_meta)

    @classmethod
    def from_storages(
        cls,
        *,
        columns: Mapping[str, ColumnStorage],
        types: Mapping[str, ColumnType],
        batch_meta: BatchMeta,
    ) -> "TensorRecordBatch":
        meta = {
            name: types[name].to_column_meta(storage.dictionary)
            for name, storage in columns.items()
        }
        tensors = {name: storage.data for name, storage in columns.items()}
        validity = {name: storage.validity for name, storage in columns.items()}
        return cls(tensors, meta, validity, types=types, storage=columns, batch_meta=batch_meta)

    @property
    def row_count(self) -> int:
        return int(self.batch_meta.row_count)

    def filter(self, mask: torch.Tensor) -> "TensorRecordBatch":
        if mask.dtype is not torch.bool:
            raise TypeError("record batch filter mask must be boolean")
        if mask.numel() != self.row_count:
            raise ValueError("record batch filter mask must match row count")
        row_count = int(mask.sum().cpu().item())
        return self._with_storage(
            {name: column.filter(mask) for name, column in self.storage.items()},
            self.batch_meta.with_row_count(row_count),
        )

    def gather(self, indices: torch.Tensor) -> "TensorRecordBatch":
        if indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.int64)
        return self._with_storage(
            {name: column.gather(indices) for name, column in self.storage.items()},
            self.batch_meta.with_row_count(int(indices.numel())),
        )

    def project(self, names: Sequence[str]) -> "TensorRecordBatch":
        return self._with_storage(
            {name: self.storage[name] for name in names},
            self.batch_meta,
            types={name: self.types[name] for name in names},
        )

    def _with_storage(
        self,
        storage: Mapping[str, ColumnStorage],
        batch_meta: BatchMeta,
        *,
        types: Mapping[str, ColumnType] | None = None,
    ) -> "TensorRecordBatch":
        selected_types = types or {name: self.types[name] for name in storage}
        return TensorRecordBatch.from_storages(
            columns=storage,
            types=selected_types,
            batch_meta=batch_meta,
        )


@dataclass(frozen=True)
class _NormalizedBatch:
    columns: dict[str, torch.Tensor]
    meta: dict[str, ColumnMeta]
    validity: dict[str, torch.Tensor | None]
    types: dict[str, ColumnType]
    storage: dict[str, ColumnStorage]
    batch_meta: BatchMeta


def _normalize_batch(batch: TensorRecordBatch) -> _NormalizedBatch:
    columns = dict(batch.columns)
    meta = dict(batch.meta)
    validity = dict(batch.validity or {})
    if batch.storage is not None:
        return _normalize_storage_batch(batch)
    _validate_tensor_batch(columns, meta, validity)
    storage = _storage_from_v1(columns, meta, validity)
    types = dict(batch.types or _types_from_meta(meta))
    batch_meta = batch.batch_meta or _infer_batch_meta(columns)
    _validate_storage_batch(storage, types, batch_meta)
    return _NormalizedBatch(columns, meta, validity, types, storage, batch_meta)


def _normalize_storage_batch(batch: TensorRecordBatch) -> _NormalizedBatch:
    storage = dict(batch.storage or {})
    types = dict(batch.types or {})
    if batch.batch_meta is None:
        raise ValueError("batch_meta is required when storage is provided")
    _validate_storage_batch(storage, types, batch.batch_meta)
    columns = {name: column.data for name, column in storage.items()}
    validity = {name: column.validity for name, column in storage.items()}
    meta = {name: types[name].to_column_meta(storage[name].dictionary) for name in storage}
    return _NormalizedBatch(columns, meta, validity, types, storage, batch.batch_meta)


def _storage_from_v1(
    columns: Mapping[str, torch.Tensor],
    meta: Mapping[str, ColumnMeta],
    validity: Mapping[str, torch.Tensor | None],
) -> dict[str, ColumnStorage]:
    result: dict[str, ColumnStorage] = {}
    for name, tensor in columns.items():
        column_meta = meta[name]
        valid = validity.get(name)
        if column_meta.logical_dtype == LogicalDType.DECIMAL:
            result[name] = ColumnStorage.decimal64(tensor, validity=valid)
        elif column_meta.logical_dtype == LogicalDType.STRING_DICT:
            result[name] = ColumnStorage.dictionary_ids(tensor, column_meta.dictionary or (), validity=valid)
        else:
            result[name] = ColumnStorage.fixed(tensor, validity=valid)
    return result


def _types_from_meta(meta: Mapping[str, ColumnMeta]) -> dict[str, ColumnType]:
    return {name: ColumnType.from_column_meta(name, column_meta) for name, column_meta in meta.items()}


def _infer_batch_meta(columns: Mapping[str, torch.Tensor]) -> BatchMeta:
    row_count = 0 if not columns else int(next(iter(columns.values())).shape[0])
    device = torch.device("cpu") if not columns else next(iter(columns.values())).device
    return BatchMeta(row_count, row_count, 0, 0, device)


def _validate_tensor_batch(
    columns: Mapping[str, torch.Tensor],
    meta: Mapping[str, ColumnMeta],
    validity: Mapping[str, torch.Tensor | None],
) -> None:
    missing = set(columns) - set(meta)
    if missing:
        raise ValueError(f"missing ColumnMeta for: {', '.join(sorted(missing))}")
    row_count = None
    for name, tensor in columns.items():
        if tensor.ndim == 0:
            raise ValueError(f"record batch column must be at least 1-D: {name}")
        row_count = int(tensor.shape[0]) if row_count is None else row_count
        if int(tensor.shape[0]) != row_count:
            raise ValueError("record batch columns must have equal row count")
        _validate_validity_shape(name, validity.get(name), int(tensor.shape[0]))


def _validate_storage_batch(
    storage: Mapping[str, ColumnStorage],
    types: Mapping[str, ColumnType],
    batch_meta: BatchMeta,
) -> None:
    if set(storage) != set(types):
        raise ValueError("record batch storage and types must have identical columns")
    for name, column in storage.items():
        if column.row_count != batch_meta.row_count:
            raise ValueError(f"storage row count mismatch for column: {name}")
        if column.device != batch_meta.device:
            raise ValueError(f"storage device mismatch for column: {name}")
        _validate_validity_shape(name, column.validity, column.row_count)


def _validate_validity_shape(name: str, valid: torch.Tensor | None, row_count: int) -> None:
    if valid is None:
        return
    if valid.dtype is not torch.bool or valid.ndim != 1 or valid.shape[0] != row_count:
        raise ValueError(f"invalid validity mask for column: {name}")
