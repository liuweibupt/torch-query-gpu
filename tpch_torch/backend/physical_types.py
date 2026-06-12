"""Tensor table types used by the DuckDB physical-plan interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from tpch_torch.relational import yyyymmdd_to_iso


@dataclass(frozen=True)
class PhysicalValue:
    """A tensor column or scalar literal plus optional decoding metadata."""

    tensor: torch.Tensor | None = None
    dictionary: tuple[str, ...] | None = None
    is_date: bool = False
    literal: int | float | str | bool | None = None

    @property
    def is_literal(self) -> bool:
        return self.tensor is None

    def require_tensor(self) -> torch.Tensor:
        if self.tensor is None:
            raise TypeError("physical value is a scalar literal, not a tensor column")
        return self.tensor

    def gather(self, indices: torch.Tensor) -> "PhysicalValue":
        return PhysicalValue(
            tensor=self.require_tensor().index_select(0, indices),
            dictionary=self.dictionary,
            is_date=self.is_date,
        )

    def filter(self, mask: torch.Tensor) -> "PhysicalValue":
        return PhysicalValue(
            tensor=self.require_tensor()[mask],
            dictionary=self.dictionary,
            is_date=self.is_date,
        )

    def cell(self, index: int) -> Any:
        raw = self.require_tensor()[index].cpu().item()
        if self.dictionary is not None:
            return self.dictionary[int(raw)]
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "order", tuple(self.order))

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
        raise KeyError(f"unknown physical column: {name}")

    def filter(self, mask: torch.Tensor, name: str | None = None) -> "PhysicalTable":
        if mask.dtype is not torch.bool:
            raise TypeError("physical filter mask must be boolean")
        if mask.ndim != 1 or mask.numel() != self.row_count:
            raise ValueError("physical filter mask must match row count")
        return PhysicalTable(
            name or self.name,
            _transform_unique_values(self.columns, lambda value: value.filter(mask)),
            self.order,
            int(mask.sum().cpu().item()),
        )

    def gather(self, indices: torch.Tensor, name: str | None = None) -> "PhysicalTable":
        if indices.dtype != torch.int64:
            indices = indices.to(dtype=torch.int64)
        return PhysicalTable(
            name or self.name,
            _transform_unique_values(self.columns, lambda value: value.gather(indices)),
            self.order,
            int(indices.numel()),
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
                columns.setdefault(alias, value)
        return cls(name, columns, tuple(order), row_count)


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
