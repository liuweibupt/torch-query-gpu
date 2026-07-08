"""Typed tensor record batches used as the P1 column metadata substrate."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import torch


class LogicalDType(str, Enum):
    INT64 = "int64"
    FP32 = "fp32"
    FP64 = "fp64"
    DECIMAL = "decimal"
    STRING_DICT = "string_dict"
    BOOL = "bool"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ColumnMeta:
    """Logical and physical metadata for one tensor column."""

    logical_dtype: LogicalDType
    torch_dtype: torch.dtype
    nullable: bool = False
    scale: int | None = None
    precision: int | None = None
    dictionary: tuple[str, ...] | None = None

    @classmethod
    def int64(cls, *, nullable: bool = False) -> "ColumnMeta":
        return cls(LogicalDType.INT64, torch.int64, nullable=nullable)

    @classmethod
    def fp32(cls, *, nullable: bool = False) -> "ColumnMeta":
        return cls(LogicalDType.FP32, torch.float32, nullable=nullable)

    @classmethod
    def fp64(cls, *, nullable: bool = False) -> "ColumnMeta":
        return cls(LogicalDType.FP64, torch.float64, nullable=nullable)

    @classmethod
    def boolean(cls, *, nullable: bool = False) -> "ColumnMeta":
        return cls(LogicalDType.BOOL, torch.bool, nullable=nullable)

    @classmethod
    def date(cls, *, nullable: bool = False) -> "ColumnMeta":
        return cls(LogicalDType.DATE, torch.int32, nullable=nullable)

    @classmethod
    def decimal(
        cls,
        *,
        precision: int,
        scale: int,
        nullable: bool = False,
    ) -> "ColumnMeta":
        if precision <= 0 or scale < 0:
            raise ValueError("decimal precision must be positive and scale non-negative")
        return cls(
            LogicalDType.DECIMAL,
            torch.int64,
            nullable=nullable,
            precision=precision,
            scale=scale,
        )

    @classmethod
    def string_dict(
        cls,
        dictionary: Sequence[str] = (),
        *,
        nullable: bool = False,
    ) -> "ColumnMeta":
        return cls(
            LogicalDType.STRING_DICT,
            torch.int64,
            nullable=nullable,
            dictionary=tuple(dictionary),
        )

    def decode_scalar(self, value: int | float | bool) -> int | float | bool | str | Decimal:
        if self.logical_dtype == LogicalDType.DECIMAL:
            return Decimal(int(value)).scaleb(-int(self.scale or 0))
        if self.logical_dtype == LogicalDType.STRING_DICT and self.dictionary is not None:
            return self.dictionary[int(value)]
        return value


@dataclass(frozen=True)
class TensorRecordBatch:
    """Columnar tensor batch plus per-column metadata and validity masks."""

    columns: Mapping[str, torch.Tensor]
    meta: Mapping[str, ColumnMeta]
    validity: Mapping[str, torch.Tensor | None] | None = None

    def __post_init__(self) -> None:
        columns = dict(self.columns)
        meta = dict(self.meta)
        validity = dict(self.validity or {})
        _validate_batch(columns, meta, validity)
        object.__setattr__(self, "columns", MappingProxyType(columns))
        object.__setattr__(self, "meta", MappingProxyType(meta))
        object.__setattr__(self, "validity", MappingProxyType(validity))

    @property
    def row_count(self) -> int:
        if not self.columns:
            return 0
        return int(next(iter(self.columns.values())).shape[0])

    def filter(self, mask: torch.Tensor) -> "TensorRecordBatch":
        if mask.dtype is not torch.bool:
            raise TypeError("record batch filter mask must be boolean")
        return TensorRecordBatch(
            columns={name: tensor[mask] for name, tensor in self.columns.items()},
            meta=self.meta,
            validity=_transform_validity(self.validity, lambda valid: valid[mask]),
        )

    def gather(self, indices: torch.Tensor) -> "TensorRecordBatch":
        if indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.int64)
        return TensorRecordBatch(
            columns={name: tensor.index_select(0, indices) for name, tensor in self.columns.items()},
            meta=self.meta,
            validity=_transform_validity(self.validity, lambda valid: valid.index_select(0, indices)),
        )

    def project(self, names: Sequence[str]) -> "TensorRecordBatch":
        return TensorRecordBatch(
            columns={name: self.columns[name] for name in names},
            meta={name: self.meta[name] for name in names},
            validity={name: self.validity.get(name) for name in names},
        )


def _validate_batch(
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
        valid = validity.get(name)
        if valid is not None and (valid.dtype is not torch.bool or valid.shape[0] != tensor.shape[0]):
            raise ValueError(f"invalid validity mask for column: {name}")


def _transform_validity(validity: Mapping[str, torch.Tensor | None], transform) -> dict[str, torch.Tensor | None]:
    return {name: None if valid is None else transform(valid) for name, valid in validity.items()}
