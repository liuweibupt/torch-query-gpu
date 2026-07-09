"""DuckDB type mapping and encoding helpers for typed tensor columns."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable

import numpy as np
import torch

from tpch_torch.record_batch import ColumnMeta, ColumnType, LogicalDType

DECIMAL_BASE = 10
_DECIMAL_RE = re.compile(r"DECIMAL\((\d+),(\d+)\)", re.I)


def column_meta_from_duckdb_type(duckdb_type: str, *, nullable: bool = False) -> ColumnMeta:
    normalized = duckdb_type.strip().upper()
    decimal_match = _DECIMAL_RE.fullmatch(normalized)
    if decimal_match is not None:
        return ColumnMeta.decimal(
            precision=int(decimal_match.group(1)),
            scale=int(decimal_match.group(2)),
            nullable=nullable,
        )
    if normalized in {"BIGINT", "INTEGER", "INT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER"}:
        return ColumnMeta.int64(nullable=nullable)
    if normalized in {"FLOAT", "REAL"}:
        return ColumnMeta.fp32(nullable=nullable)
    if normalized in {"DOUBLE", "DOUBLE PRECISION"}:
        return ColumnMeta.fp64(nullable=nullable)
    if normalized == "BOOLEAN":
        return ColumnMeta.boolean(nullable=nullable)
    if normalized == "DATE":
        return ColumnMeta.date(nullable=nullable)
    if normalized in {"VARCHAR", "TEXT", "CHAR"} or normalized.startswith("VARCHAR"):
        return ColumnMeta.string_dict(nullable=nullable)
    return ColumnMeta(LogicalDType.UNKNOWN, torch.float64, nullable=nullable)


def column_type_from_duckdb_type(
    name: str,
    duckdb_type: str,
    *,
    nullable: bool = False,
) -> ColumnType:
    """Return the v2 logical type while preserving the DuckDB type spelling."""

    normalized = duckdb_type.strip().upper()
    decimal_match = _DECIMAL_RE.fullmatch(normalized)
    if decimal_match is not None:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        return ColumnType.decimal(name, precision=precision, scale=scale, nullable=nullable)
    if normalized in {"BIGINT", "INTEGER", "INT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER"}:
        return ColumnType.int64(name, nullable=nullable)
    if normalized in {"FLOAT", "REAL"}:
        return ColumnType.fp32(name, nullable=nullable)
    if normalized in {"DOUBLE", "DOUBLE PRECISION"}:
        return ColumnType.fp64(name, nullable=nullable)
    if normalized == "BOOLEAN":
        return ColumnType.boolean(name, nullable=nullable)
    if normalized == "DATE":
        return ColumnType.date(name, nullable=nullable)
    if normalized in {"VARCHAR", "TEXT", "CHAR"} or normalized.startswith("VARCHAR"):
        return ColumnType.varchar(name, nullable=nullable)
    return ColumnType(name, normalized or "UNKNOWN", normalized or "UNKNOWN", LogicalDType.UNKNOWN, nullable)


def encode_decimal_array(values: Iterable[object] | np.ndarray, meta: ColumnMeta, device: str) -> torch.Tensor:
    if meta.logical_dtype != LogicalDType.DECIMAL:
        raise TypeError("encode_decimal_array requires decimal ColumnMeta")
    encoded = [_decimal_to_scaled_int(value, int(meta.scale or 0)) for value in list(values)]
    return torch.tensor(encoded, dtype=torch.int64, device=device)


def encode_strings_dynamic(
    values: Iterable[object] | np.ndarray,
    device: str,
    *,
    existing: ColumnMeta | None = None,
) -> tuple[torch.Tensor, ColumnMeta]:
    value_strings = [str(value) for value in list(values)]
    vocabulary = _expanded_string_vocabulary(value_strings, existing)
    ids = {literal: index for index, literal in enumerate(vocabulary)}
    tensor = torch.tensor([ids[value] for value in value_strings], dtype=torch.int64, device=device)
    return tensor, ColumnMeta.string_dict(vocabulary, nullable=existing.nullable if existing else False)


def align_decimal_tensors(
    left: torch.Tensor,
    left_meta: ColumnMeta,
    right: torch.Tensor,
    right_meta: ColumnMeta,
) -> tuple[torch.Tensor, torch.Tensor, ColumnMeta]:
    scale = max(int(left_meta.scale or 0), int(right_meta.scale or 0))
    left_aligned = rescale_decimal_tensor(left, int(left_meta.scale or 0), scale)
    right_aligned = rescale_decimal_tensor(right, int(right_meta.scale or 0), scale)
    precision = max(int(left_meta.precision or 0), int(right_meta.precision or 0))
    return left_aligned, right_aligned, ColumnMeta.decimal(precision=precision, scale=scale)


def rescale_decimal_tensor(values: torch.Tensor, from_scale: int, to_scale: int) -> torch.Tensor:
    if to_scale < from_scale:
        divisor = DECIMAL_BASE ** (from_scale - to_scale)
        return torch.div(values, divisor, rounding_mode="trunc")
    multiplier = DECIMAL_BASE ** (to_scale - from_scale)
    return values * multiplier


def decimal_literal_scale(value: object) -> int:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    exponent = int(decimal_value.as_tuple().exponent)
    return max(0, -exponent)


def encode_decimal_literal(value: object, scale: int) -> int:
    return _decimal_to_scaled_int(value, scale)


def _expanded_string_vocabulary(
    values: list[str],
    existing: ColumnMeta | None,
) -> tuple[str, ...]:
    if existing is None:
        return tuple(sorted(set(values)))
    vocabulary = list(existing.dictionary or ())
    seen = set(vocabulary)
    for value in values:
        if value in seen:
            continue
        vocabulary.append(value)
        seen.add(value)
    return tuple(vocabulary)


def _decimal_to_scaled_int(value: object, scale: int) -> int:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    scaled = decimal_value.scaleb(scale)
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(f"decimal value cannot be represented at scale {scale}: {value}")
    return int(integral)
