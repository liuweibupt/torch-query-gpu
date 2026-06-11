# Cold/Hot Query Timing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible benchmark command that reports cold and hot query timing for the existing SQL -> DuckDB/Sirius/Substrait frontend -> TQPPlan -> PyTorch backend path.

**Architecture:** Keep validation and execution unchanged. Add a small benchmark module that times the same `run_sql_with_frontend()` path using wall-clock `perf_counter_ns()` plus CUDA synchronization before and after the timed region. The CLI resolves SQL text once, then measures cold samples with a fresh DuckDB connection per sample and hot samples with one connection, warmups, and repeated measured runs.

**Tech Stack:** Python 3.12, DuckDB, PyTorch CUDA synchronization, pytest, existing `tpch_torch` package.

---

## Timing semantics

- Cold timing: SQL text is resolved before timing. Each measured sample opens a new DuckDB connection, executes once, synchronizes CUDA if requested, records wall-clock elapsed time, and closes the connection. It does not flush OS page cache and does not restart the Python process.
- Hot timing: one DuckDB connection is opened, warmup iterations run unrecorded, then measured iterations run with CUDA synchronization around each full query execution.
- Measured elapsed time is end-to-end process wall time for frontend compile/admission, tensor fetch/encoding, H2D transfer if CUDA tensors are built, PyTorch execution, and result materialization.
- DuckDB validation baseline is not executed by the benchmark command.

## Tasks

1. Add tests for benchmark cold and hot connection/warmup semantics.
2. Implement `tpch_torch/benchmark.py` with `BenchmarkConfig`, `TimingSample`, `TimingSummary`, and `benchmark_sql()`.
3. Add `scripts/benchmark_query.py` CLI and `tpch-torch-benchmark` entrypoint.
4. Add CLI parser tests and package entrypoint tests.
5. Document the timing method and commands in README and architecture docs.
6. Verify with pytest, compileall, and a real SF1 Q6 benchmark on CUDA.
