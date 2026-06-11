from pathlib import Path

from tpch_torch.benchmark import BenchmarkConfig, benchmark_sql
from tpch_torch.relational import QueryResult


class FakeConnection:
    def __init__(self, connection_id: int, events: list[tuple[str, int]]):
        self.connection_id = connection_id
        self.events = events

    def close(self):
        self.events.append(("close", self.connection_id))


class IncrementingClock:
    def __init__(self, step_ns: int = 10_000_000):
        self.current = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        value = self.current
        self.current += self.step_ns
        return value


def test_benchmark_uses_new_connections_for_cold_and_reuses_one_hot_connection():
    events: list[tuple[str, int]] = []
    runner_calls: list[tuple[int, bool]] = []

    def connect(path: Path):
        connection_id = len([event for event in events if event[0] == "connect"])
        events.append(("connect", connection_id))
        return FakeConnection(connection_id, events)

    def runner(con, sql, *, device, frontend, use_compressed_masks):
        runner_calls.append((con.connection_id, use_compressed_masks))
        return QueryResult(query_id=6, rows=[{"revenue": 1.0}])

    report = benchmark_sql(
        BenchmarkConfig(
            db_path=Path("tpch.duckdb"),
            sql="select 1",
            device="cpu",
            frontend="sirius",
            cold_runs=2,
            warmup_runs=1,
            hot_runs=2,
            use_compressed_masks=True,
        ),
        connect=connect,
        runner=runner,
        clock_ns=IncrementingClock(),
    )

    assert events == [("connect", 0), ("close", 0), ("connect", 1), ("close", 1), ("connect", 2), ("close", 2)]
    assert runner_calls == [(0, True), (1, True), (2, True), (2, True), (2, True)]
    assert [sample.mode for sample in report.samples] == ["cold", "cold", "hot", "hot"]
    assert report.cold.count == 2
    assert report.hot.count == 2
    assert report.cold.median_ms == 10.0
    assert report.hot.median_ms == 10.0


def test_cuda_benchmark_synchronizes_around_timed_runs():
    sync_events: list[str] = []

    def connect(path: Path):
        return FakeConnection(0, [])

    def runner(con, sql, *, device, frontend, use_compressed_masks):
        return QueryResult(query_id=None, rows=[{"n": 1}])

    benchmark_sql(
        BenchmarkConfig(
            db_path=Path("tpch.duckdb"),
            sql="select count(*) as n from t",
            device="cuda",
            frontend="sirius",
            cold_runs=1,
            warmup_runs=0,
            hot_runs=1,
        ),
        connect=connect,
        runner=runner,
        clock_ns=IncrementingClock(),
        synchronizer=lambda: sync_events.append("sync"),
    )

    assert sync_events == ["sync", "sync", "sync", "sync"]


def test_benchmark_config_rejects_negative_run_counts():
    try:
        BenchmarkConfig(db_path=Path("tpch.duckdb"), sql="select 1", cold_runs=-1)
    except ValueError as exc:
        assert "cold_runs" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_samples_uses_nearest_rank_p95():
    from tpch_torch.benchmark import TimingSample, summarize_samples

    samples = tuple(TimingSample("hot", index, float(index + 1), None, 1) for index in range(10))

    assert summarize_samples(samples).p95_ms == 10.0
