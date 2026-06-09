# DuckDB SQL to Substrait to PyTorch/GPU Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a direct SQL-driven execution path that exports real DuckDB Substrait plans and runs supported TPC-H queries on PyTorch tensors, with CPU/GPU validation against DuckDB.

**Architecture:** Keep DuckDB responsible for SQL parsing and Substrait export, then dispatch exported TPC-H query shapes to correctness-first PyTorch executors. JSON remains optional debugging input; the primary CLI path accepts `--query`, `--sql`, or `--sql-file` and performs SQL→Substrait→PyTorch in one command.

**Tech Stack:** Python 3.12, DuckDB Substrait extension, PyTorch tensors, pytest, optional CUDA.

---

### Task 1: Generic SQL loading and CLI skeleton

**Files:**
- Modify: `tpch_torch/sql.py`
- Modify: `tpch_torch/duckdb_bridge.py`
- Create: `tpch_torch/runner.py`
- Create: `scripts/run_query.py`
- Create: `scripts/validate_query.py`
- Modify: `pyproject.toml`
- Test: `tests/test_runner_cli.py`

**Steps:**
1. Write failing parser tests for `tpch-torch-run --query 6`, `--sql`, and `--sql-file`.
2. Run targeted pytest and confirm failure.
3. Implement SQL loading helpers and generic CLI parsers.
4. Add console scripts `tpch-torch-run` and `tpch-torch-validate`.
5. Run targeted pytest and confirm pass.
6. Commit.

### Task 2: Direct SQL→Substrait dispatch API preserving Q1

**Files:**
- Modify: `tpch_torch/runner.py`
- Modify: `tpch_torch/duckdb_bridge.py`
- Test: `tests/test_runner.py`

**Steps:**
1. Write failing tests that run canonical Q1 by SQL text through `run_sql`, requiring `export_substrait_json` to be called and returning Q1 rows.
2. Run targeted pytest and confirm failure.
3. Implement `run_sql(con, sql, device)` and `validate_sql(con, sql, device)` with Q1 dispatch.
4. Keep existing Q1 commands working as wrappers.
5. Run targeted pytest and confirm pass.
6. Commit.

### Task 3: Q6 PyTorch executor through direct SQL path

**Files:**
- Modify: `tpch_torch/sql.py`
- Create: `tpch_torch/queries/q06.py`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_q06.py`
- Test: `tests/test_runner.py`

**Steps:**
1. Write failing Q6 executor test on a small lineitem fixture.
2. Write failing direct `validate_sql(...Q6...)` test against DuckDB fixture.
3. Run targeted tests and confirm failure.
4. Implement Q6 tensor executor and dispatch.
5. Run targeted tests and confirm pass.
6. Commit.

### Task 4: Single-table/query-specific support for Q12, Q14, Q15, Q19

**Files:**
- Create: `tpch_torch/queries/q12.py`
- Create: `tpch_torch/queries/q14.py`
- Create: `tpch_torch/queries/q15.py`
- Create: `tpch_torch/queries/q19.py`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_supported_tpch.py`

**Steps:**
1. Write failing end-to-end tests on small DuckDB fixtures for each query.
2. Run targeted tests and confirm failure.
3. Implement correctness-first tensor logic for each query.
4. Run targeted tests and confirm pass.
5. Commit.

### Task 5: Dimension lookup join helpers

**Files:**
- Create: `tpch_torch/join.py`
- Modify: `tpch_torch/storage.py`
- Test: `tests/test_join.py`

**Steps:**
1. Write failing tests for unique-key tensor lookup joins and composite-key lookups.
2. Run targeted tests and confirm failure.
3. Implement lookup helpers using sorted torch tensors and explicit duplicate-key errors.
4. Run targeted tests and confirm pass.
5. Commit.

### Task 6: Join query executors for Q3, Q5, Q7, Q8, Q9, Q10, Q11, Q13, Q18

**Files:**
- Create: query modules under `tpch_torch/queries/`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_supported_tpch.py`

**Steps:**
1. Add one failing end-to-end validation test per query using small generated TPC-H data.
2. Run each test and confirm failure.
3. Implement query-specific correctness-first tensor executor.
4. Run each targeted test and confirm pass.
5. Commit after each query or small batch.

### Task 7: Batch support matrix and CLI validation

**Files:**
- Modify: `README.md`
- Test: `tests/test_supported_tpch.py`

**Steps:**
1. Write failing test that enumerates DuckDB-exportable TPC-H queries and asserts supported queries validate through `validate_sql`.
2. Run and confirm failure for remaining unsupported exported queries.
3. Fill dispatcher support list until all exported queries pass or have documented DuckDB export failures.
4. Update README support matrix and direct SQL CLI examples.
5. Run full tests and compileall.
6. Run SF1 CUDA validation for representative queries, including Q1 and Q6.
7. Commit and push.
