"""Type metadata for typed tensor record batches."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal, Sequence

import torch


class LogicalDType(str, Enum):
    INT64 = "int64"
    FP32 = "fp32"
    FP64 = "fp64"
    DECIMAL = "decimal"
    STRING_DICT = "string_dict"
    STRING = "string"
    BOOL = "bool"
    DATE = "date"
    UNKNOWN = "unknown"


class StorageKind(str, Enum):
    FIXED = "fixed"
    DECIMAL64 = "decimal64"
    DICTIONARY = "dictionary"
    UTF8_OFFSETS = "utf8_offsets"


@dataclass(frozen=True)
class AllocationOwner:
    """Ownership metadata for tensor buffers."""

    kind: Literal["torch", "external", "dlpack", "cudf"]
    handle: object | None = None
    stream: object | None = None
    memory_resource: str | None = None

    @classmethod
    def torch(cls) -> "AllocationOwner":
        return cls(kind="torch")


@dataclass(frozen=True)
class ColumnMeta:
    """v1 compatibility metadata used by existing physical operators."""

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
class ColumnType:
    """DuckDB logical type plus TQP lowering type for one column."""

    name: str
    duckdb_type_id: str
    duckdb_type_repr: str
    logical_dtype: LogicalDType
    nullable: bool
    precision: int | None = None
    scale: int | None = None

    @classmethod
    def int64(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "BIGINT", "BIGINT", LogicalDType.INT64, nullable)

    @classmethod
    def fp32(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "FLOAT", "FLOAT", LogicalDType.FP32, nullable)

    @classmethod
    def fp64(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "DOUBLE", "DOUBLE", LogicalDType.FP64, nullable)

    @classmethod
    def boolean(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "BOOLEAN", "BOOLEAN", LogicalDType.BOOL, nullable)

    @classmethod
    def date(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "DATE", "DATE", LogicalDType.DATE, nullable)

    @classmethod
    def decimal(
        cls,
        name: str,
        *,
        precision: int,
        scale: int,
        nullable: bool = False,
    ) -> "ColumnType":
        return cls(
            name,
            "DECIMAL",
            f"DECIMAL({precision},{scale})",
            LogicalDType.DECIMAL,
            nullable,
            precision=precision,
            scale=scale,
        )

    @classmethod
    def varchar(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "VARCHAR", "VARCHAR", LogicalDType.STRING, nullable)

    @classmethod
    def string_dict(cls, name: str, *, nullable: bool = False) -> "ColumnType":
        return cls(name, "VARCHAR", "VARCHAR", LogicalDType.STRING_DICT, nullable)

    @classmethod
    def from_column_meta(cls, name: str, meta: ColumnMeta) -> "ColumnType":
        if meta.logical_dtype == LogicalDType.DECIMAL:
            return cls.decimal(
                name,
                precision=int(meta.precision or 0),
                scale=int(meta.scale or 0),
                nullable=meta.nullable,
            )
        if meta.logical_dtype == LogicalDType.FP32:
            return cls.fp32(name, nullable=meta.nullable)
        if meta.logical_dtype == LogicalDType.FP64:
            return cls.fp64(name, nullable=meta.nullable)
        if meta.logical_dtype == LogicalDType.BOOL:
            return cls.boolean(name, nullable=meta.nullable)
        if meta.logical_dtype == LogicalDType.DATE:
            return cls.date(name, nullable=meta.nullable)
        if meta.logical_dtype == LogicalDType.STRING_DICT:
            return cls.string_dict(name, nullable=meta.nullable)
        return cls.int64(name, nullable=meta.nullable)

    def to_column_meta(self, dictionary: tuple[str, ...] | None = None) -> ColumnMeta:
        if self.logical_dtype == LogicalDType.DECIMAL:
            return ColumnMeta.decimal(
                precision=int(self.precision or 0),
                scale=int(self.scale or 0),
                nullable=self.nullable,
            )
        if self.logical_dtype == LogicalDType.FP32:
            return ColumnMeta.fp32(nullable=self.nullable)
        if self.logical_dtype == LogicalDType.FP64:
            return ColumnMeta.fp64(nullable=self.nullable)
        if self.logical_dtype == LogicalDType.BOOL:
            return ColumnMeta.boolean(nullable=self.nullable)
        if self.logical_dtype == LogicalDType.DATE:
            return ColumnMeta.date(nullable=self.nullable)
        if self.logical_dtype in {LogicalDType.STRING, LogicalDType.STRING_DICT}:
            return ColumnMeta.string_dict(dictionary or (), nullable=self.nullable)
        return ColumnMeta.int64(nullable=self.nullable)
