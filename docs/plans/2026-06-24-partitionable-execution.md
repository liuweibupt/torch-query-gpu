# Partitionable Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit CoddSpeed-style partitionable execution path for single-table aggregate physical graphs and benchmark it.

**Architecture:** The runner carries an optional `PartitionConfig` into the PyTorch backend. A new `physical_partitionable` module validates supported graph shapes, executes each row chunk through the existing physical interpreter with scan range injection, and merges partial aggregate rows on the host.

**Tech Stack:** Python 3.12, DuckDB, PyTorch, pytest, existing TQP physical graph executor.

---

### Task 1: RED tests for partition helpers and unsupported shapes

**Files:**
- Create: `tests/test_partitionable_execution.py`

**Step 1: Write failing tests**

Add tests for `row_ranges`, invalid chunk size, and explicit unsupported non-aggregate graph behavior.

**Step 2: Run tests to verify RED**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_partitionable_execution.py -q
```

Expected: import fails because `tpch_torch.backend.physical_partitionable` does not exist.

### Task 2: GREEN partition helper module

**Files:**
- Create: `tpch_torch/backend/physical_partitionable.py`

**Step 1: Implement minimal helper API**

Add `PartitionConfig`, `row_ranges`, graph analysis shell, and unsupported errors.

**Step 2: Run tests**

Run the same pytest command and expect helper tests to pass or move to the next missing behavior.

### Task 3: RED/GREEN Q6 partitionable correctness

**Files:**
- Modify: `tests/test_partitionable_execution.py`
- Modify: `tpch_torch/backend/physical.py`
- Modify: `tpch_torch/backend/graph.py`
- Modify: `tpch_torch/backend/pytorch.py`
- Modify: `tpch_torch/runner.py`

**Step 1: Write failing test**

Use `create_lineitem_fixture`, compile Q6 SQL through Sirius, run with `PartitionConfig(table="lineitem", chunk_size=2)`, and compare with DuckDB/default PyTorch result.

**Step 2: Implement scan range injection and backend plumbing**

- Add optional scan range and `enable_fusion` parameters to `PhysicalPlanExecutor`.
- Add `execute_partitionable_physical_plan`.
- Thread `partition_config` from runner to backend.

**Step 3: Verify**

Run targeted test and existing Q6/runner tests.

### Task 4: RED/GREEN Q1 grouped aggregate merge

**Files:**
- Modify: `tests/test_partitionable_execution.py`
- Modify: `tpch_torch/backend/physical_partitionable.py`

**Step 1: Write failing test**

Use `TPC_H_Q1_SQL` and a fixture spanning multiple chunks. Assert partitionable rows match default run and DuckDB validation.

**Step 2: Implement grouped merge and average weighted merge**

Map output aliases from `DESCRIBE <sql>` to group/aggregate positions and merge partial rows by group key.

**Step 3: Verify**

Run partition tests, Q1 tests, physical tests.

### Task 5: CLI/benchmark support and docs

**Files:**
- Modify: `tpch_torch/benchmark.py`
- Modify: `scripts/benchmark_query.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_scripts.py`
- Create: `docs/partitionable-execution.zh.md`
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/papers/reading-notes/coddspeed-sigmod-2026.zh.md`

**Step 1: Write failing parser/config tests**

Check benchmark parser accepts partition flags and runner receives config.

**Step 2: Implement flags and report output**

Add `partition_config` to `BenchmarkConfig` and print it in human/JSON report.

**Step 3: Update docs**

Document CoddSpeed mapping, supported queries, command examples, and performance interpretation.

### Task 6: Verification, benchmark, merge, push

**Files:** all changed files.

**Step 1: Run verification**

```bash
git diff --check
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
```

**Step 2: Benchmark if data exists**

Use `/work/torch-query-gpu/data/tpch_sf1.duckdb` if present:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python scripts/benchmark_query.py --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 6 --device cpu --cold-runs 0 --warmup-runs 1 --hot-runs 3
timeout 60 /work/torch-query-gpu/.venv/bin/python scripts/benchmark_query.py --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 6 --device cpu --cold-runs 0 --warmup-runs 1 --hot-runs 3 --partition-table lineitem --partition-chunk-size 100000
```

**Step 3: Commit, merge to main, push, cleanup worktree**

Follow repository policy: commit branch, merge into `main`, push `origin main`, remove merged branch/worktree.
