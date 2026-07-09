"""Typed TensorRecordBatch aggregate helpers."""

from __future__ import annotations

import torch

from tpch_torch.record_batch import BatchMeta, ColumnStorage, LogicalDType, TensorRecordBatch


def grouped_sum_batch(
    batch: TensorRecordBatch,
    *,
    group_keys: tuple[str, ...],
    sum_columns: tuple[str, ...],
) -> TensorRecordBatch:
    if len(group_keys) != 1:
        raise NotImplementedError("first typed batch aggregate supports one group key")
    key_name = group_keys[0]
    unique_keys, inverse = torch.unique(batch.columns[key_name], sorted=True, return_inverse=True)
    storages = {key_name: _storage_like(batch, key_name, unique_keys)}
    types = {key_name: batch.types[key_name]}
    for column in sum_columns:
        output_name = f"sum_{column}"
        summed = _sum_by_inverse(batch.columns[column], inverse, int(unique_keys.numel()))
        storages[output_name] = _storage_like(batch, column, summed)
        types[output_name] = _rename_sum_type(batch.types[column], output_name)
    meta = BatchMeta(
        row_count=int(unique_keys.numel()),
        chunk_size=batch.batch_meta.chunk_size,
        chunk_index=batch.batch_meta.chunk_index,
        source_offset=batch.batch_meta.source_offset,
        device=batch.batch_meta.device,
        schema_version=batch.batch_meta.schema_version,
    )
    return TensorRecordBatch.from_storages(columns=storages, types=types, batch_meta=meta)


def _sum_by_inverse(values: torch.Tensor, inverse: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    return result.index_add(0, inverse, values)


def _storage_like(batch: TensorRecordBatch, column: str, data: torch.Tensor) -> ColumnStorage:
    if batch.types[column].logical_dtype == LogicalDType.DECIMAL:
        return ColumnStorage.decimal64(data)
    if batch.storage[column].dictionary is not None:
        return ColumnStorage.dictionary_ids(data, batch.storage[column].dictionary or ())
    return ColumnStorage.fixed(data)


def _rename_sum_type(column_type, name: str):
    return type(column_type)(
        name=name,
        duckdb_type_id=column_type.duckdb_type_id,
        duckdb_type_repr=column_type.duckdb_type_repr,
        logical_dtype=column_type.logical_dtype,
        nullable=column_type.nullable,
        precision=column_type.precision,
        scale=column_type.scale,
    )
