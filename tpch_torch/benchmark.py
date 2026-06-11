"""Cold/hot benchmark helpers for end-to-end TQP query execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from math import ceil
from statistics import mean, median, stdev
from time import perf_counter_ns
from typing import Literal

import duckdb
import torch

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


def benchmark_sql(
    config: BenchmarkConfig,
    *,
    connect: ConnectionFactory = connect_database,
    runner: QueryRunner = run_sql_with_frontend,
    clock_ns: Clock = perf_counter_ns,
    synchronizer: Synchronizer | None = None,
) -> BenchmarkReport:
    """Measure cold and hot end-to-end execution for one SQL query."""

    sync = _synchronizer(config.device, synchronizer)
    cold_samples = _measure_cold(config, connect, runner, clock_ns, sync)
    hot_samples = _measure_hot(config, connect, runner, clock_ns, sync)
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


def _measure_cold(
    config: BenchmarkConfig,
    connect: ConnectionFactory,
    runner: QueryRunner,
    clock_ns: Clock,
    sync: Synchronizer,
) -> list[TimingSample]:
    samples: list[TimingSample] = []
    for iteration in range(config.cold_runs):
        con = connect(config.db_path)
        try:
            samples.append(_measure_one("cold", iteration, con, config, runner, clock_ns, sync))
        finally:
            con.close()
    return samples


def _measure_hot(
    config: BenchmarkConfig,
    connect: ConnectionFactory,
    runner: QueryRunner,
    clock_ns: Clock,
    sync: Synchronizer,
) -> list[TimingSample]:
    if config.hot_runs == 0:
        return []
    con = connect(config.db_path)
    try:
        for _ in range(config.warmup_runs):
            _run_query(con, config, runner)
            sync()
        return [_measure_one("hot", iteration, con, config, runner, clock_ns, sync) for iteration in range(config.hot_runs)]
    finally:
        con.close()


def _measure_one(
    mode: BenchmarkMode,
    iteration: int,
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    runner: QueryRunner,
    clock_ns: Clock,
    sync: Synchronizer,
) -> TimingSample:
    sync()
    start_ns = clock_ns()
    result = _run_query(con, config, runner)
    sync()
    elapsed_ms = (clock_ns() - start_ns) / NANOSECONDS_PER_MILLISECOND
    return TimingSample(mode, iteration, elapsed_ms, result.query_id, len(result.rows))


def _run_query(con: duckdb.DuckDBPyConnection, config: BenchmarkConfig, runner: QueryRunner) -> QueryResult:
    return runner(
        con,
        config.sql,
        device=config.device,
        frontend=config.frontend,
        use_compressed_masks=config.use_compressed_masks,
    )


def _synchronizer(device: str, synchronizer: Synchronizer | None) -> Synchronizer:
    if synchronizer is not None:
        return synchronizer
    if device == "cuda":
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
