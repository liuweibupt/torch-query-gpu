"""Columnar tensor storage utilities for TPC-H tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from numbers import Integral
from typing import Any, Iterable, Mapping

import numpy as np
import torch

STRING_COLUMNS = frozenset({"l_returnflag", "l_linestatus"})
DATE_COLUMNS = frozenset({"l_shipdate", "l_commitdate", "l_receiptdate"})


@dataclass(frozen=True)
class TensorTable:
    """A table represented as one tensor per column plus optional vocabularies."""

    columns: Mapping[str, torch.Tensor]
    dictionaries: Mapping[str, tuple[str, ...]]

    def __len__(self) -> int:
        first_column = next(iter(self.columns.values()), None)
        if first_column is None:
            return 0
        return int(first_column.numel())

    def decode_value(self, column: str, encoded_value: int) -> str:
        vocabulary = self.dictionaries[column]
        return vocabulary[int(encoded_value)]

    def require_columns(self, required_columns: Iterable[str]) -> None:
        missing = [name for name in required_columns if name not in self.columns]
        if missing:
            raise KeyError(", ".join(missing))


def table_from_rows(rows: Iterable[Mapping[str, Any]], device: str | torch.device = "cpu") -> TensorTable:
    """Build a `TensorTable` from Python row dictionaries."""

    materialized_rows = list(rows)
    if not materialized_rows:
        return TensorTable(columns={}, dictionaries={})

    column_names = tuple(materialized_rows[0].keys())
    columns: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}

    for column_name in column_names:
        values = [row[column_name] for row in materialized_rows]
        tensor, vocabulary = _encode_column(column_name, values, device)
        columns[column_name] = tensor
        if vocabulary is not None:
            dictionaries[column_name] = vocabulary

    return TensorTable(columns=columns, dictionaries=dictionaries)


def table_from_columnar(
    columnar: Mapping[str, Iterable[Any]], device: str | torch.device = "cpu"
) -> TensorTable:
    """Build a `TensorTable` from a mapping of column names to Python iterables."""

    columns: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}
    for column_name, values_iterable in columnar.items():
        tensor, vocabulary = _encode_column(column_name, values_iterable, device)
        columns[column_name] = tensor
        if vocabulary is not None:
            dictionaries[column_name] = vocabulary
    return TensorTable(columns=columns, dictionaries=dictionaries)


def _encode_column(
    column_name: str, values: Iterable[Any], device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if isinstance(values, np.ndarray):
        return _encode_numpy_column(column_name, values, device)

    materialized_values = list(values)
    if column_name in STRING_COLUMNS:
        return _encode_string_column(materialized_values, device)
    if column_name in DATE_COLUMNS:
        return _encode_date_column(materialized_values, device), None
    return _encode_numeric_column(materialized_values, device), None


def _encode_numpy_column(
    column_name: str, values: np.ndarray, device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...] | None]:
    if column_name in STRING_COLUMNS:
        vocabulary, inverse = np.unique(values.astype(str), return_inverse=True)
        tensor = torch.as_tensor(inverse, dtype=torch.int64, device=device)
        return tensor, tuple(str(value) for value in vocabulary)
    if column_name in DATE_COLUMNS:
        if values.dtype == np.dtype("O"):
            return _encode_date_column(values.tolist(), device), None
        tensor = torch.as_tensor(values, dtype=torch.int32, device=device)
        return tensor, None
    if values.dtype == np.dtype("O"):
        return _encode_numeric_column(values.tolist(), device), None
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    return tensor, None


def _encode_string_column(
    values: list[Any], device: str | torch.device
) -> tuple[torch.Tensor, tuple[str, ...]]:
    vocabulary = tuple(sorted({str(value) for value in values}))
    ids = {value: index for index, value in enumerate(vocabulary)}
    encoded = [ids[str(value)] for value in values]
    return torch.tensor(encoded, dtype=torch.int64, device=device), vocabulary


def _encode_date_column(values: list[Any], device: str | torch.device) -> torch.Tensor:
    encoded = [_date_to_yyyymmdd(value) for value in values]
    return torch.tensor(encoded, dtype=torch.int32, device=device)


def _encode_numeric_column(values: list[Any], device: str | torch.device) -> torch.Tensor:
    normalized = [float(value) if isinstance(value, Decimal) else value for value in values]
    return torch.tensor(normalized, dtype=torch.float64, device=device)


def _date_to_yyyymmdd(value: Any) -> int:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return (value.year * 10_000) + (value.month * 100) + value.day
    if isinstance(value, str):
        return int(value.replace("-", ""))
    if isinstance(value, Integral):
        return int(value)
    raise TypeError(f"unsupported date value: {value!r}")
