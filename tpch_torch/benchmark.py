"""Cold/hot benchmark helpers for end-to-end TQP query execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter_ns
from typing import Literal

import duckdb
import torch

from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.ir import FrontendName
from tpch_torch.relational import QueryResult
from tpch_torch.runner import run_sql_with_frontend

BenchmarkMode = Literal["cold", "hot"]
ConnectionFactory = Callable[[Path], duckdb.DuckDBPyConnection]
QueryRunner = Callable[..., QueryResult]
Clock = Callable[[], int]
Synchronizer = Callable[[], None]

NANOSECONDS_PER_MILLISECOND = 1_000_000.0
DEFAULT_COLD_RUNS = 1
DEFAULT_WARMUP_RUNS = 3
DEFAULT_HOT_RUNS = 10


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for cold/hot end-to-end query timing."""

    db_path: Path
    sql: str
    device: str = "cpu"
    frontend: FrontendName = "sirius"
    cold_runs: int = DEFAULT_COLD_RUNS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    hot_runs: int = DEFAULT_HOT_RUNS
    use_compressed_masks: bool = False
    partition_config: PartitionConfig | None = None

    def __post_init__(self) -> None:
        _validate_non_negative("cold_runs", self.cold_runs)
        _validate_non_negative("warmup_runs", self.warmup_runs)
        _validate_non_negative("hot_runs", self.hot_runs)
        if self.cold_runs == 0 and self.hot_runs == 0:
            raise ValueError("at least one of cold_runs or hot_runs must be positive")


@dataclass(frozen=True)
class TimingSample:
    """One measured query execution sample."""

    mode: BenchmarkMode
    iteration: int
    elapsed_ms: float
    query_id: int | None
    row_count: int


@dataclass(frozen=True)
class TimingSummary:
    """Summary statistics for a benchmark mode."""

    count: int
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    stdev_ms: float | None


@dataclass(frozen=True)
class BenchmarkReport:
    """Cold/hot benchmark results."""

    config: BenchmarkConfig
    cold: TimingSummary
    hot: TimingSummary
    samples: tuple[TimingSample, ...]


@dataclass(frozen=True)
class _BenchmarkRuntime:
    connect: ConnectionFactory
    runner: QueryRunner
    clock_ns: Clock
    sync: Synchronizer


@dataclass(frozen=True)
class _BenchmarkContext:
    config: BenchmarkConfig
    runtime: _BenchmarkRuntime


@dataclass(frozen=True)
class _Measurement:
    mode: BenchmarkMode
    iteration: int
    con: duckdb.DuckDBPyConnection


def benchmark_sql(
    config: BenchmarkConfig,
    *,
    connect: ConnectionFactory = connect_database,
    runner: QueryRunner = run_sql_with_frontend,
    clock_ns: Clock = perf_counter_ns,
    synchronizer: Synchronizer | None = None,
) -> BenchmarkReport:
    """Measure cold and hot end-to-end execution for one SQL query."""

    runtime = _BenchmarkRuntime(connect, runner, clock_ns, _synchronizer(config.device, synchronizer))
    context = _BenchmarkContext(config=config, runtime=runtime)
    cold_samples = _measure_cold(context)
    hot_samples = _measure_hot(context)
    samples = cold_samples + hot_samples
    return BenchmarkReport(
        config=config,
        cold=summarize_samples(cold_samples),
        hot=summarize_samples(hot_samples),
        samples=tuple(samples),
    )


def summarize_samples(samples: Sequence[TimingSample]) -> TimingSummary:
    """Return summary statistics for measured samples."""

    if not samples:
        return TimingSummary(0, None, None, None, None, None, None)
    values = [sample.elapsed_ms for sample in samples]
    return TimingSummary(
        count=len(values),
        min_ms=min(values),
        median_ms=median(values),
        mean_ms=mean(values),
        p95_ms=_percentile(values, 0.95),
        max_ms=max(values),
        stdev_ms=stdev(values) if len(values) > 1 else 0.0,
    )


def _measure_cold(context: _BenchmarkContext) -> list[TimingSample]:
    samples: list[TimingSample] = []
    for iteration in range(context.config.cold_runs):
        con = context.runtime.connect(context.config.db_path)
        try:
            measurement = _Measurement("cold", iteration, con)
            samples.append(_measure_one(context, measurement))
        finally:
            con.close()
    return samples


def _measure_hot(context: _BenchmarkContext) -> list[TimingSample]:
    if context.config.hot_runs == 0:
        return []
    con = context.runtime.connect(context.config.db_path)
    try:
        for _ in range(context.config.warmup_runs):
            _run_query(context, con)
            context.runtime.sync()
        return [
            _measure_one(context, _Measurement("hot", iteration, con))
            for iteration in range(context.config.hot_runs)
        ]
    finally:
        con.close()


def _measure_one(context: _BenchmarkContext, measurement: _Measurement) -> TimingSample:
    context.runtime.sync()
    start_ns = context.runtime.clock_ns()
    result = _run_query(context, measurement.con)
    context.runtime.sync()
    elapsed_ms = (context.runtime.clock_ns() - start_ns) / NANOSECONDS_PER_MILLISECOND
    return TimingSample(
        measurement.mode,
        measurement.iteration,
        elapsed_ms,
        result.query_id,
        len(result.rows),
    )


def _run_query(context: _BenchmarkContext, con: duckdb.DuckDBPyConnection) -> QueryResult:
    config = context.config
    return context.runtime.runner(
        con,
        config.sql,
        device=config.device,
        frontend=config.frontend,
        use_compressed_masks=config.use_compressed_masks,
        partition_config=config.partition_config,
    )


def _synchronizer(device: str, synchronizer: Synchronizer | None) -> Synchronizer:
    if synchronizer is not None:
        return synchronizer
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
        return torch.cuda.synchronize
    return _noop


def _noop() -> None:
    return None


def _percentile(values: Sequence[float], quantile: float) -> float:
    sorted_values = sorted(values)
    index = max(ceil(len(sorted_values) * quantile) - 1, 0)
    return sorted_values[index]


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
