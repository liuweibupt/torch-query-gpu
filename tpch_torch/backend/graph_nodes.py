"""Reusable PyTorch graph nodes for TPC-H relational execution.

These nodes are intentionally small wrappers around tensor operators.  TPC-H
query recipes compose them explicitly so joins, semi/anti subqueries, scalar
subqueries, grouping, and scans are visible in the backend graph path instead
of being hidden inside whole-query template executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import duckdb
import torch

from tpch_torch.operators import grouped_max, grouped_mean, grouped_min
from tpch_torch.relational import (
    aggregate_count_by_keys as _rel_aggregate_count_by_keys,
    aggregate_sum_by_keys as _rel_aggregate_sum_by_keys,
    build_lookup_index,
    composite_key,
    lookup_values_from_index,
)
from tpch_torch.storage import TensorTable


@dataclass(frozen=True)
class GraphTable:
    """A tensor table produced by a graph scan or relational node."""

    name: str
    tensor_table: TensorTable

    @property
    def columns(self):
        return self.tensor_table.columns

    @property
    def dictionaries(self):
        return self.tensor_table.dictionaries

    def __len__(self) -> int:
        return len(self.tensor_table)

    def row_ids(self) -> torch.Tensor:
        device = _table_device(self.tensor_table)
        return torch.arange(len(self), dtype=torch.int64, device=device)

    def filter(self, mask: torch.Tensor, name: str | None = None) -> "GraphTable":
        return FilterNode(self, mask, name=name).execute()


@dataclass(frozen=True)
class GroupedAggregateResult:
    """Keys and values returned by a grouped reduction node."""

    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class ScanNode:
    """Columnar DuckDB table scan into tensors."""

    table_name: str
    columns: tuple[str, ...]

    def execute(self, con: duckdb.DuckDBPyConnection, device: str = "cpu") -> GraphTable:
        return GraphTable(self.table_name, _fetch_graph_tensor_table(con, self.table_name, self.columns, device))


@dataclass(frozen=True)
class FilterNode:
    """Boolean mask selection over all columns of a graph table."""

    source: GraphTable
    mask: torch.Tensor
    name: str | None = None

    def execute(self) -> GraphTable:
        if self.mask.dtype is not torch.bool:
            raise TypeError("graph filter mask must be boolean")
        if self.mask.ndim != 1 or self.mask.numel() != len(self.source):
            raise ValueError("graph filter mask must match source row count")
        filtered = {column: values[self.mask] for column, values in self.source.columns.items()}
        return GraphTable(self.name or self.source.name, TensorTable(filtered, self.source.dictionaries))


@dataclass(frozen=True)
class LookupJoinNode:
    """Many-to-one lookup/hash join projection from build keys to probe rows."""

    build: GraphTable
    build_key: str
    build_value: str
    probe_keys: torch.Tensor

    def execute(self, missing_value: int | float = -1) -> torch.Tensor:
        index = build_lookup_index(self.build.columns[self.build_key], self.build.columns[self.build_value])
        return lookup_values_from_index(index, self.probe_keys, missing_value=missing_value)


def lookup_tensor_values(
    build_keys: torch.Tensor,
    build_values: torch.Tensor,
    probe_keys: torch.Tensor,
    missing_value: int | float = -1,
) -> torch.Tensor:
    """Map probe keys to values using an explicit tensor lookup index."""

    return lookup_values_from_index(
        build_lookup_index(build_keys, build_values),
        probe_keys,
        missing_value=missing_value,
    )


@dataclass(frozen=True)
class CompositeLookupJoinNode:
    """Lookup join for two-column equi-join keys."""

    build_left: torch.Tensor
    build_right: torch.Tensor
    build_values: torch.Tensor
    probe_left: torch.Tensor
    probe_right: torch.Tensor
    multiplier: int

    def execute(self, missing_value: int | float = -1) -> torch.Tensor:
        build_key = composite_key(self.build_left, self.build_right, self.multiplier)
        probe_key = composite_key(self.probe_left, self.probe_right, self.multiplier)
        return lookup_tensor_values(build_key, self.build_values, probe_key, missing_value)


@dataclass(frozen=True)
class SemiJoinNode:
    """Boolean mask for probe rows whose key exists in build rows."""

    probe_keys: torch.Tensor
    build_keys: torch.Tensor

    def execute(self) -> torch.Tensor:
        return torch.isin(self.probe_keys, torch.unique(self.build_keys))


@dataclass(frozen=True)
class AntiJoinNode:
    """Boolean mask for probe rows whose key does not exist in build rows."""

    probe_keys: torch.Tensor
    build_keys: torch.Tensor

    def execute(self) -> torch.Tensor:
        return ~SemiJoinNode(self.probe_keys, self.build_keys).execute()


@dataclass(frozen=True)
class MaterializedCTENode:
    """Materialize a named CTE result as a reusable graph table."""

    name: str
    columns: dict[str, torch.Tensor]
    dictionaries: dict[str, tuple[str, ...]] | None = None

    def execute(self) -> GraphTable:
        return GraphTable(self.name, TensorTable(dict(self.columns), dict(self.dictionaries or {})))


@dataclass(frozen=True)
class ScalarSubqueryNode:
    """Scalar aggregate node used by TPC-H correlated/CTE subqueries."""

    values: torch.Tensor
    op: str
    multiplier: float = 1.0

    @classmethod
    def sum(cls, values: torch.Tensor, multiplier: float = 1.0) -> "ScalarSubqueryNode":
        return cls(values=values, op="sum", multiplier=multiplier)

    @classmethod
    def max(cls, values: torch.Tensor) -> "ScalarSubqueryNode":
        return cls(values=values, op="max")

    @classmethod
    def min(cls, values: torch.Tensor) -> "ScalarSubqueryNode":
        return cls(values=values, op="min")

    @classmethod
    def avg(cls, values: torch.Tensor, multiplier: float = 1.0) -> "ScalarSubqueryNode":
        return cls(values=values, op="avg", multiplier=multiplier)

    def execute(self) -> torch.Tensor:
        if self.values.numel() == 0:
            return torch.tensor(float("nan"), dtype=torch.float64, device=self.values.device)
        if self.op == "sum":
            return self.values.sum() * self.multiplier
        if self.op == "max":
            return self.values.max()
        if self.op == "min":
            return self.values.min()
        if self.op == "avg":
            return self.values.mean() * self.multiplier
        raise ValueError(f"unsupported scalar subquery aggregate: {self.op}")


class AggregateNode:
    """Factory for grouped tensor aggregate nodes."""

    @staticmethod
    def grouped_sum(keys: Sequence[torch.Tensor], values: torch.Tensor) -> GroupedAggregateResult:
        group_keys, sums = _rel_aggregate_sum_by_keys(tuple(keys), values)
        return GroupedAggregateResult(group_keys, sums)

    @staticmethod
    def grouped_count(keys: Sequence[torch.Tensor]) -> GroupedAggregateResult:
        group_keys, counts = _rel_aggregate_count_by_keys(tuple(keys))
        return GroupedAggregateResult(group_keys, counts)

    @staticmethod
    def grouped_mean(keys: Sequence[torch.Tensor], values: torch.Tensor) -> GroupedAggregateResult:
        group_keys, sums = _rel_aggregate_sum_by_keys(tuple(keys), values)
        _, counts = _rel_aggregate_count_by_keys(tuple(keys))
        return GroupedAggregateResult(group_keys, sums / counts.to(dtype=sums.dtype))

    @staticmethod
    def grouped_min(keys: Sequence[torch.Tensor], values: torch.Tensor) -> GroupedAggregateResult:
        group_ids, unique_keys = _group_ids(keys)
        return GroupedAggregateResult(unique_keys, grouped_min(values, group_ids, int(unique_keys.shape[0])))

    @staticmethod
    def grouped_max(keys: Sequence[torch.Tensor], values: torch.Tensor) -> GroupedAggregateResult:
        group_ids, unique_keys = _group_ids(keys)
        return GroupedAggregateResult(unique_keys, grouped_max(values, group_ids, int(unique_keys.shape[0])))

    @staticmethod
    def grouped_count_distinct(keys: Sequence[torch.Tensor], values: torch.Tensor) -> GroupedAggregateResult:
        if values.ndim != 1:
            raise ValueError("distinct aggregate values must be 1-D")
        materialized: dict[tuple[int, ...], set[int]] = {}
        host_keys = torch.stack(tuple(key.to(dtype=torch.int64) for key in keys), dim=1).cpu().tolist()
        host_values = values.to(dtype=torch.int64).cpu().tolist()
        for key, value in zip(host_keys, host_values):
            materialized.setdefault(tuple(int(part) for part in key), set()).add(int(value))
        sorted_keys = sorted(materialized)
        device = values.device
        key_tensor = torch.tensor(sorted_keys, dtype=torch.int64, device=device)
        counts = torch.tensor([len(materialized[key]) for key in sorted_keys], dtype=torch.int64, device=device)
        return GroupedAggregateResult(key_tensor, counts)


@dataclass(frozen=True)
class GroupedScalarSubqueryNode:
    """Per-key scalar subquery aggregate with lookup back to outer rows."""

    keys: torch.Tensor
    values: torch.Tensor

    @classmethod
    def sum(cls, keys: Sequence[torch.Tensor], values: torch.Tensor) -> "GroupedScalarSubqueryNode":
        result = AggregateNode.grouped_sum(keys, values)
        return cls(result.keys, result.values)

    @classmethod
    def mean(cls, keys: Sequence[torch.Tensor], values: torch.Tensor) -> "GroupedScalarSubqueryNode":
        result = AggregateNode.grouped_mean(keys, values)
        return cls(result.keys, result.values)

    @classmethod
    def min(cls, keys: Sequence[torch.Tensor], values: torch.Tensor) -> "GroupedScalarSubqueryNode":
        result = AggregateNode.grouped_min(keys, values)
        return cls(result.keys, result.values)

    @classmethod
    def max(cls, keys: Sequence[torch.Tensor], values: torch.Tensor) -> "GroupedScalarSubqueryNode":
        result = AggregateNode.grouped_max(keys, values)
        return cls(result.keys, result.values)

    def lookup(self, probe_keys: Sequence[torch.Tensor], missing_value: int | float = -1) -> torch.Tensor:
        probes = tuple(probe_keys)
        if len(probes) != int(self.keys.shape[1]):
            raise ValueError("probe key count must match grouped key count")
        build_columns = tuple(self.keys[:, index] for index in range(int(self.keys.shape[1])))
        build_key, probe_key = _packed_lookup_keys(build_columns, probes)
        return lookup_tensor_values(build_key, self.values, probe_key, missing_value)


@dataclass(frozen=True)
class SortLimitNode:
    """Host-side final ORDER BY/LIMIT node for already aggregated result rows."""

    rows: tuple[dict, ...]
    order_by: tuple[tuple[str, bool], ...]
    limit: int | None = None

    def execute(self) -> list[dict]:
        sorted_rows = list(self.rows)
        for column, descending in reversed(self.order_by):
            sorted_rows = sorted(sorted_rows, key=lambda row: row[column], reverse=descending)
        if self.limit is None:
            return sorted_rows
        return sorted_rows[: self.limit]


def row_ids(table: GraphTable) -> torch.Tensor:
    return table.row_ids()


def _packed_lookup_keys(
    build_keys: Sequence[torch.Tensor],
    probe_keys: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    build = tuple(key.to(dtype=torch.int64) for key in build_keys)
    probe = tuple(key.to(dtype=torch.int64) for key in probe_keys)
    if not build:
        raise ValueError("at least one key column is required")
    if len(build) != len(probe):
        raise ValueError("build and probe key counts must match")
    bounds = tuple(_shared_key_bound(left, right) for left, right in zip(build, probe))
    return _pack_key_columns(build, bounds), _pack_key_columns(probe, bounds)


def _shared_key_bound(build_key: torch.Tensor, probe_key: torch.Tensor) -> tuple[int, int]:
    values = tuple(tensor for tensor in (build_key, probe_key) if tensor.numel() > 0)
    if not values:
        return 0, 1
    min_value = min(int(value.min().cpu().item()) for value in values)
    max_value = max(int(value.max().cpu().item()) for value in values)
    return min_value, max(max_value - min_value + 1, 1)


def _pack_key_columns(keys: Sequence[torch.Tensor], bounds: Sequence[tuple[int, int]]) -> torch.Tensor:
    packed = keys[0] - bounds[0][0]
    for key, (min_value, width) in zip(keys[1:], bounds[1:]):
        packed = packed * width + (key - min_value)
    return packed


def _group_ids(keys: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    from tpch_torch.operators import composite_group_ids

    return composite_group_ids(tuple(key.to(dtype=torch.int64) for key in keys))


def _table_device(table: TensorTable) -> torch.device:
    first = next(iter(table.columns.values()), None)
    if first is None:
        return torch.device("cpu")
    return first.device


def _fetch_graph_tensor_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: tuple[str, ...],
    device: str,
) -> TensorTable:
    from tpch_torch.backend.generic import _encode_generic_column

    select_list = ", ".join(columns)
    columnar = con.execute(f"select {select_list} from {table_name}").fetchnumpy()
    tensors: dict[str, torch.Tensor] = {}
    dictionaries: dict[str, tuple[str, ...]] = {}
    for column, values in columnar.items():
        tensor, vocabulary = _encode_generic_column(values, device)
        tensors[column] = tensor
        if vocabulary is not None:
            dictionaries[column] = vocabulary
    return TensorTable(tensors, dictionaries)

# Thin adapters used by TPC-H graph recipes. They keep query files declarative
# while routing scans/lookups/aggregations through the common graph-node layer.
def fetch_tensor_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: Sequence[str],
    device: str | torch.device = "cpu",
) -> TensorTable:
    return ScanNode(table_name, tuple(columns)).execute(con, device=str(device)).tensor_table


def lookup_values(
    dimension_keys: torch.Tensor,
    dimension_values: torch.Tensor,
    fact_keys: torch.Tensor,
    missing_value: int | float = -1,
) -> torch.Tensor:
    return lookup_tensor_values(dimension_keys, dimension_values, fact_keys, missing_value)


def lookup_row_indices(
    dimension_keys: torch.Tensor,
    fact_keys: torch.Tensor,
    missing_value: int = -1,
) -> torch.Tensor:
    rows = torch.arange(dimension_keys.numel(), dtype=torch.int64, device=dimension_keys.device)
    return lookup_tensor_values(dimension_keys, rows, fact_keys, missing_value)


def aggregate_sum_by_keys(
    key_columns: Sequence[torch.Tensor], value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    result = AggregateNode.grouped_sum(tuple(key_columns), value)
    return result.keys, result.values


def aggregate_count_by_keys(key_columns: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    result = AggregateNode.grouped_count(tuple(key_columns))
    return result.keys, result.values


def decode(table: TensorTable, column: str, encoded: int | torch.Tensor) -> str:
    raw = int(encoded.cpu().item()) if isinstance(encoded, torch.Tensor) else int(encoded)
    return table.dictionaries[column][raw]


def string_eq(table: TensorTable, column: str, value: str) -> torch.Tensor:
    return _string_match(table, column, lambda item: item == value)


def string_ne(table: TensorTable, column: str, value: str) -> torch.Tensor:
    return ~string_eq(table, column, value)


def string_in(table: TensorTable, column: str, values: Sequence[str]) -> torch.Tensor:
    accepted = set(values)
    return _string_match(table, column, lambda item: item in accepted)


def string_startswith(table: TensorTable, column: str, prefix: str) -> torch.Tensor:
    return _string_match(table, column, lambda item: item.startswith(prefix))


def string_contains(table: TensorTable, column: str, needle: str) -> torch.Tensor:
    return _string_match(table, column, lambda item: needle in item)


def string_not_like_special_requests(table: TensorTable, column: str) -> torch.Tensor:
    def accepts(item: str) -> bool:
        return not ("special" in item and "requests" in item and item.index("special") < item.rindex("requests"))

    return _string_match(table, column, accepts)


def yyyymmdd_to_year(values: torch.Tensor) -> torch.Tensor:
    return values.to(dtype=torch.int64) // 10_000


def yyyymmdd_to_iso(value: int) -> str:
    raw = int(value)
    year = raw // 10_000
    month = (raw // 100) % 100
    day = raw % 100
    return f"{year:04d}-{month:02d}-{day:02d}"


def _string_match(table: TensorTable, column: str, predicate) -> torch.Tensor:
    values = table.columns[column]
    matching = [index for index, item in enumerate(table.dictionaries[column]) if predicate(item)]
    if not matching:
        return torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    accepted = torch.tensor(matching, dtype=values.dtype, device=values.device)
    return torch.isin(values, accepted)
