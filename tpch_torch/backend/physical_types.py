"""Tensor table types used by the DuckDB physical-plan interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from tpch_torch.relational import yyyymmdd_to_iso
from tpch_torch.record_batch import (
    BatchMeta,
    ColumnMeta,
    ColumnStorage,
    ColumnType,
    LogicalDType,
    TensorRecordBatch,
)


@dataclass(frozen=True)
class PhysicalValue:
    """A tensor column or scalar literal plus optional decoding metadata."""

    tensor: torch.Tensor | None = None
    dictionary: tuple[str, ...] | None = None
    is_date: bool = False
    literal: int | float | str | bool | None = None
    valid: torch.Tensor | None = None
    sorted_non_decreasing: bool = False
    unique: bool = False
    meta: ColumnMeta | None = None

    @property
    def is_literal(self) -> bool:
        return self.tensor is None

    def require_tensor(self) -> torch.Tensor:
        if self.tensor is None:
            raise TypeError("physical value is a scalar literal, not a tensor column")
        return self.tensor

    def gather(self, indices: torch.Tensor) -> "PhysicalValue":
        valid = None if self.valid is None else self.valid.index_select(0, indices)
        return PhysicalValue(
            tensor=self.require_tensor().index_select(0, indices),
            dictionary=self.dictionary,
            is_date=self.is_date,
            valid=valid,
            meta=self.meta,
        )

    def gather_optional(self, indices: torch.Tensor, valid: torch.Tensor) -> "PhysicalValue":
        if valid.dtype is not torch.bool:
            raise TypeError("optional gather validity mask must be boolean")
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
        base_valid = self.valid.index_select(0, safe_indices) if self.valid is not None else valid
        return PhysicalValue(
            tensor=self.require_tensor().index_select(0, safe_indices),
            dictionary=self.dictionary,
            is_date=self.is_date,
            valid=base_valid & valid,
            meta=self.meta,
        )

    def filter(self, mask: torch.Tensor) -> "PhysicalValue":
        valid = None if self.valid is None else self.valid[mask]
        return PhysicalValue(
            tensor=self.require_tensor()[mask],
            dictionary=self.dictionary,
            is_date=self.is_date,
            valid=valid,
            sorted_non_decreasing=self.sorted_non_decreasing,
            unique=self.unique,
            meta=self.meta,
        )

    def with_metadata(
        self,
        *,
        sorted_non_decreasing: bool | None = None,
        unique: bool | None = None,
    ) -> "PhysicalValue":
        return PhysicalValue(
            tensor=self.tensor,
            dictionary=self.dictionary,
            is_date=self.is_date,
            literal=self.literal,
            valid=self.valid,
            sorted_non_decreasing=(
                self.sorted_non_decreasing
                if sorted_non_decreasing is None
                else sorted_non_decreasing
            ),
            unique=self.unique if unique is None else unique,
            meta=self.meta,
        )

    def cell(self, index: int) -> Any:
        if self.valid is not None and not bool(self.valid[index].cpu().item()):
            return None
        raw = self.require_tensor()[index].cpu().item()
        if self.dictionary is not None:
            return self.dictionary[int(raw)]
        if self.meta is not None and self.meta.logical_dtype == LogicalDType.DECIMAL:
            return self.meta.decode_scalar(int(raw))
        if self.is_date:
            return yyyymmdd_to_iso(int(raw))
        if isinstance(raw, float):
            return float(raw)
        if isinstance(raw, bool):
            return bool(raw)
        return int(raw)


@dataclass(frozen=True)
class PhysicalTable:
    """A relation flowing between interpreted DuckDB physical nodes."""

    name: str
    columns: Mapping[str, PhysicalValue]
    order: tuple[str, ...]
    row_count: int
    batch: TensorRecordBatch | None = None

    def __post_init__(self) -> None:
        columns = dict(self.columns)
        order = tuple(self.order)
        object.__setattr__(self, "columns", MappingProxyType(columns))
        object.__setattr__(self, "order", order)
        if self.batch is None:
            object.__setattr__(self, "batch", _batch_from_ordered_values(columns, order, self.row_count))

    @classmethod
    def from_batch(
        cls,
        name: str,
        batch: TensorRecordBatch,
        *,
        order: Sequence[str] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> "PhysicalTable":
        table_order = tuple(order or batch.columns.keys())
        columns = _physical_values_from_batch(batch, table_order)
        for alias, source in (aliases or {}).items():
            columns[alias] = columns[source]
        return cls(name, columns, table_order, batch.row_count, batch)

    def value_at(self, index: int) -> PhysicalValue:
        try:
            return self.columns[self.order[index]]
        except IndexError as exc:
            raise KeyError(f"projection index out of range: #{index}") from exc

    def value_named(self, name: str) -> PhysicalValue:
        candidates = _name_candidates(name)
        for candidate in candidates:
            if candidate in self.columns:
                return self.columns[candidate]
        unique_match = _unique_base_match(self.columns, candidates)
        if unique_match is not None:
            return self.columns[unique_match]
        raise KeyError(f"unknown physical column: {name}")

    def filter(self, mask: torch.Tensor, name: str | None = None) -> "PhysicalTable":
        if mask.dtype is not torch.bool:
            raise TypeError("physical filter mask must be boolean")
        if mask.ndim != 1 or mask.numel() != self.row_count:
            raise ValueError("physical filter mask must match row count")
        row_count = int(mask.sum().cpu().item())
        return PhysicalTable(
            name or self.name,
            _transform_unique_values(self.columns, lambda value: value.filter(mask)),
            self.order,
            row_count,
            self.batch.filter(mask) if self.batch is not None else None,
        )

    def gather(self, indices: torch.Tensor, name: str | None = None) -> "PhysicalTable":
        if indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.int64)
        return PhysicalTable(
            name or self.name,
            _transform_unique_values(self.columns, lambda value: value.gather(indices)),
            self.order,
            int(indices.numel()),
            self.batch.gather(indices) if self.batch is not None else None,
        )

    @classmethod
    def projected(
        cls,
        name: str,
        items: Sequence[tuple[str, PhysicalValue, Sequence[str]]],
        row_count: int,
    ) -> "PhysicalTable":
        columns: dict[str, PhysicalValue] = {}
        order: list[str] = []
        for index, (raw_name, value, aliases) in enumerate(items):
            column_name = _unique_name(raw_name, columns, index)
            columns[column_name] = value
            order.append(column_name)
            for alias in aliases:
                if _is_projection_position(alias):
                    columns[alias] = value
                else:
                    columns.setdefault(alias, value)
        return cls(name, columns, tuple(order), row_count)


def _batch_from_ordered_values(
    columns: Mapping[str, PhysicalValue],
    order: tuple[str, ...],
    row_count: int,
) -> TensorRecordBatch | None:
    storages: dict[str, ColumnStorage] = {}
    types: dict[str, ColumnType] = {}
    device: torch.device | None = None
    for name in order:
        value = columns.get(name)
        if value is None or value.tensor is None:
            return None
        tensor = value.require_tensor()
        device = tensor.device if device is None else device
        if tensor.device != device:
            return None
        storages[name] = _storage_from_physical_value(value)
        types[name] = _type_from_physical_value(name, value)
    batch_meta = BatchMeta(row_count, row_count, 0, 0, device or torch.device("cpu"))
    return TensorRecordBatch.from_storages(columns=storages, types=types, batch_meta=batch_meta)


def _physical_values_from_batch(
    batch: TensorRecordBatch,
    order: tuple[str, ...],
) -> dict[str, PhysicalValue]:
    return {name: _physical_value_from_batch(batch, name) for name in order}


def _physical_value_from_batch(batch: TensorRecordBatch, name: str) -> PhysicalValue:
    storage = batch.storage[name]
    meta = batch.meta[name]
    return PhysicalValue(
        tensor=storage.data,
        dictionary=storage.dictionary,
        is_date=meta.logical_dtype == LogicalDType.DATE,
        valid=storage.validity,
        meta=meta,
    )


def _storage_from_physical_value(value: PhysicalValue) -> ColumnStorage:
    tensor = value.require_tensor()
    if value.dictionary is not None:
        return ColumnStorage.dictionary_ids(tensor, value.dictionary, validity=value.valid)
    if value.meta is not None and value.meta.logical_dtype == LogicalDType.DECIMAL:
        return ColumnStorage.decimal64(tensor, validity=value.valid)
    return ColumnStorage.fixed(tensor, validity=value.valid)


def _type_from_physical_value(name: str, value: PhysicalValue) -> ColumnType:
    if value.meta is not None:
        return ColumnType.from_column_meta(name, value.meta)
    tensor = value.require_tensor()
    if tensor.dtype == torch.float32:
        return ColumnType.fp32(name)
    if tensor.dtype == torch.float64:
        return ColumnType.fp64(name)
    if tensor.dtype == torch.bool:
        return ColumnType.boolean(name)
    return ColumnType.date(name) if value.is_date else ColumnType.int64(name)


def materialize_literal(value: PhysicalValue, row_count: int, device: torch.device) -> PhysicalValue:
    """Broadcast a scalar literal into a tensor column."""

    if value.tensor is not None:
        return value
    if isinstance(value.literal, bool):
        tensor = torch.full((row_count,), bool(value.literal), dtype=torch.bool, device=device)
    elif isinstance(value.literal, int):
        tensor = torch.full((row_count,), int(value.literal), dtype=torch.int64, device=device)
    elif isinstance(value.literal, float):
        tensor = torch.full((row_count,), float(value.literal), dtype=torch.float64, device=device)
    else:
        raise TypeError(f"cannot materialize non-numeric literal: {value.literal!r}")
    return PhysicalValue(tensor=tensor)


def table_device(table: PhysicalTable) -> torch.device:
    for value in table.columns.values():
        if value.tensor is not None:
            return value.tensor.device
    return torch.device("cpu")


def _transform_unique_values(
    columns: Mapping[str, PhysicalValue],
    transform,
) -> dict[str, PhysicalValue]:
    transformed: dict[int, PhysicalValue] = {}
    result: dict[str, PhysicalValue] = {}
    for name, value in columns.items():
        key = id(value)
        if key not in transformed:
            transformed[key] = transform(value)
        result[name] = transformed[key]
    return result


def _name_candidates(name: str) -> tuple[str, ...]:
    stripped = name.strip()
    unquoted = stripped.replace('"', "")
    candidates = [stripped, unquoted]
    if "." in unquoted:
        candidates.append(unquoted.rsplit(".", 1)[-1])
    return tuple(dict.fromkeys(candidates))


def _unique_name(name: str, columns: Mapping[str, PhysicalValue], index: int) -> str:
    if name not in columns:
        return name
    return f"{name}__{index}"


def _unique_base_match(columns: Mapping[str, PhysicalValue], candidates: tuple[str, ...]) -> str | None:
    matches = [
        name
        for name in columns
        if any(_base_name(name) == _base_name(candidate) for candidate in candidates)
    ]
    return matches[0] if len(matches) == 1 else None


def _base_name(name: str) -> str:
    return _strip_unique_suffix(name.replace('"', "").strip().rsplit(".", 1)[-1])


def _strip_unique_suffix(name: str) -> str:
    base, separator, suffix = name.rpartition("__")
    return base if separator and suffix.isdigit() else name


def _is_projection_position(name: str) -> bool:
    return name.startswith("#") and name[1:].isdigit()
